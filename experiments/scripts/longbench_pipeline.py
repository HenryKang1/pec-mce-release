"""
LongBench QA evaluation for CTKS and pointer-preserved evidence variants.

Per question, LongBench provides a long `context` (5-20k+ tokens) containing
multiple titled passages. Since this exceeds the SLM's 2048-token context
window, raw retrieval or compression is mandatory. This is the setting
where compile-time structuring should actually help.

Variants evaluated (same reader model across all):
  - raw_trunc    : truncate context to fit, single call (baseline)
  - raw_topk     : split into passages, FAISS retrieve top-k raw passages
  - summary      : LLM-generated 1-sentence summary per passage (pure abstractive)
  - anchors      : verbatim entities/dates/numbers per passage (pure extractive)
  - anchored     : summary + anchors concatenated per passage   <-- ours
  - pec_card     : pointer-preserved evidence cards (verbatim sentence cards)
  - pec_hydrate  : retrieve evidence cards, then hydrate their source sentences
  - pec_adaptive : use hydrated cards, fall back to raw_topk when evidence is weak
  - pec_bridge   : top-2 raw passages plus hydrated evidence cards

Usage:
  python longbench_pipeline.py --task hotpotqa --model lfm2.5-1.2b-instruct --variant anchored
  python longbench_pipeline.py --task hotpotqa --model lfm2.5-1.2b-instruct --variant pec_hydrate
  python longbench_pipeline.py --task all --model lfm2.5-1.2b-instruct --variant all
"""
import argparse
import json
import re
import sys
import string
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from utils.config import DATASETS_DIR, TOPIC6_DIR, MODELS_DIR, GGUF_MODELS
from rag_pipeline import ChunkIndex, LLMGenerator


TASKS = ["hotpotqa", "2wikimqa", "musique"]
# hybrid2 = top-2 raw passages + anchored notes for the rest (rescue variant)
# hybrid3 = top-3 raw + anchored rest
# pec_* = pointer-preserved evidence cards; no abstractive generation.
VARIANTS = ["raw_trunc", "raw_topk", "summary", "anchors", "anchored",
            "hybrid2", "hybrid3", "pec_card", "pec_hydrate", "pec_adaptive",
            "pec_bridge", "pec_bridge_k3", "pec_hop", "pec_query_expand",
            # PEC novelty-defense ablations:
            "pec_hop_no_anchor",     # 2-step retrieval, but step 2 uses original q
            "pec_hop_fact_only",     # cards = bare FACT sentence (no TITLE/ANCHORS/PTR)
            "pec_hop_no_hydration",  # full cards, no +/-1 sentence window at read time
            # Hydration window sweep:
            "pec_hop_w0",            # window=0: just the source sentence, no neighbors
            "pec_hop_w2",            # window=2: wider local discourse
            # Dynamic fallback for capacity-threshold rescue:
            "pec_hop_dynamic",       # if evidence_is_weak: PEC-Bridge prompt, else PEC-Hop
            # Token-budget matched Raw RAG (same retrieval as raw_topk, ctx budget=840):
            "raw_topk_b840",
            # === New ARR-defense variants ===
            "sentence_only",         # bare-sentence retrieval, no schema, no hop, no hydration
            "pec_router",            # heuristic adaptive: pec_hop / pec_bridge / pec_bridge_k3
            "pec_hop_extractive",    # pec_hop + answer-only extractive decoding prompt (5-word cap)
            "pec_hop_short15",       # pec_hop + extractive prompt with 15-word cap
            "pec_hop_span",          # pec_hop + extractive "answer span" prompt, no length cap
            "pec_hop_concise",       # pec_hop + soft "be concise" prompt, no length cap
            "pec_hop_relations",     # pec_hop with RELATIONS field; relation-aware hop-2
            # Corruption stress test (negative controls for the schema):
            "pec_hop_shuffle_ptr",   # hydrate at random sentence indices (wrong neighborhoods)
            "pec_hop_random_anchor", # replace ANCHORS with random tokens before hop-2
            # === Cross-prompt baselines (disentangle prompt vs representation effect) ===
            # Same retrieval as raw_topk / raw_topk_b840 / sentence_only, but with the
            # decoding prompts from pec_hop_*. Critical for P0 reviewer concern that
            # the headline gain is a prompt artefact rather than a card-representation effect.
            "raw_topk_extractive", "raw_topk_short15", "raw_topk_concise",
            "raw_topk_b840_extractive", "raw_topk_b840_short15", "raw_topk_b840_concise",
            "sentence_only_extractive", "sentence_only_short15", "sentence_only_concise",
            # === MCE-RAG primitive: bridge-aware minimal copyable evidence ===
            "bridge_sentence",
            "bridge_sentence_extractive", "bridge_sentence_short15", "bridge_sentence_concise",
            # === External compression baselines (ARR head-to-head) ===
            # Both take the same top-k raw passages as raw_topk, then compress
            # before feeding the reader. llmlingua2 uses token-level keep/drop;
            # provence uses encoder-based sentence pruning.
            "llmlingua2", "llmlingua2_extractive",
            "provence", "provence_extractive",
            # PEC-Hop + sentence distillation: PEC retrieval narrows to the
            # right 3-5 passages, then bridge-sentence picker keeps only the
            # top 2-4 answer-relevant sentences (~150 tokens). Closes the
            # context-size gap vs Provence while staying non-parametric.
            "pec_hop_distill", "pec_hop_distill_extractive",
            # === Upset-rate boosters ===
            # rerank: retrieve top-2k seeds, rerank by dense + question-anchor
            # overlap + answer-type cue, take top-k. Same hydration as pec_hop.
            "pec_hop_rerank", "pec_hop_rerank_extractive",
            "pec_hop_rerank_fewextractive",
            # fewshot: same retrieval as pec_hop, fewextractive prompt only.
            "pec_hop_fewextractive",
            ]


# === Variant parser ===
# Maps a (possibly compound) variant name to (retrieval_variant, prompt_kind).
# prompt_kind is one of: 'default' | 'extractive' | 'short15' | 'concise' |
# 'fewextractive' (2-shot extractive) or 'oneextractive' (1-shot extractive).
# Longer suffixes must come first so the greedy match doesn't consume a substring.
PROMPT_SUFFIXES = ("_fewextractive", "_oneextractive", "_extractive", "_short15", "_concise")


def parse_variant(variant: str) -> tuple[str, str]:
    """Decompose <base>_<promptKind> -> (base, promptKind).

    Variants without a prompt suffix retain their full name as the base and
    return prompt_kind='default'. This lets us add cross-prompt baselines
    (e.g. raw_topk_extractive, sentence_only_concise) without duplicating
    the per-base dispatch logic.
    """
    for suf in PROMPT_SUFFIXES:
        if variant.endswith(suf):
            return variant[: -len(suf)], suf[1:]
    return variant, "default"


QUERY_EXPAND_PROMPT = (
    "Rewrite the question into a search query that mentions the people, "
    "places, dates, or organizations needed to answer it. "
    "Output one short query only.\n\n"
    "Question: {question}\n"
    "Search query:"
)


SUMMARY_PROMPT = (
    "Summarize this passage in 2-3 factual sentences. "
    "Preserve names, dates, numbers, and roles verbatim.\n\n"
    "Title: {title}\n"
    "Passage: {body}\n\n"
    "Summary:"
)

# Multi-word capitalized phrases on a single line; allow v. / & / ' inside
ENT_RX = re.compile(
    r"\b[A-Z][a-zA-Z0-9'\-]*(?:[ \t]+(?:[A-Z][a-zA-Z0-9'\-]*|v\.|&))+\b"
)
NUM_RX = re.compile(r"\b\d{2,4}(?:\.\d+)?\b")
YEAR_RX = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")
PROPER_RX = re.compile(r"(?<![a-zA-Z])[A-Z][a-zA-Z'\-]+(?:[ \t]+[A-Z][a-zA-Z'\-]+)*")
SENT_RX = re.compile(r"(?<=[.!?])\s+(?=(?:[A-Z0-9\"'(\[]|$))")
WORD_RX = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")

QUESTION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "his", "in", "is",
    "it", "its", "of", "on", "or", "she", "that", "the", "their", "they",
    "this", "to", "was", "were", "what", "when", "where", "which", "who",
    "whom", "whose", "why", "with",
}


# ============================================================
# Passage parsing
# ============================================================
PASSAGE_HEADER = re.compile(r"\n?Passage\s+\d+:\s*\n", re.IGNORECASE)


def split_passages(context: str) -> list[tuple[str, str]]:
    """Split LongBench context into (title, body) pairs.

    Context format:
        Passage 1:
        Title line
        Body ...

        Passage 2:
        ...
    """
    parts = PASSAGE_HEADER.split(context)
    out = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.split("\n", 1)
        if len(lines) == 1:
            title = lines[0][:80]
            body = lines[0]
        else:
            title, body = lines[0].strip(), lines[1].strip()
        if not body:
            continue
        out.append((title[:120], body))
    return out


def split_sentences(body: str) -> list[str]:
    """Lightweight sentence splitter for LongBench Wikipedia-style passages."""
    text = re.sub(r"\s+", " ", body).strip()
    if not text:
        return []
    sentences = [s.strip() for s in SENT_RX.split(text) if s.strip()]
    merged = []
    for sent in sentences:
        if merged and len(WORD_RX.findall(sent)) < 4:
            merged[-1] = f"{merged[-1]} {sent}".strip()
        else:
            merged.append(sent)
    return merged


def query_terms(question: str) -> set[str]:
    """Content terms used only for evidence confidence, not for card creation."""
    terms = set()
    for tok in WORD_RX.findall(question.lower()):
        if len(tok) < 3 or tok in QUESTION_STOPWORDS:
            continue
        terms.add(tok)
    return terms


# ============================================================
# Anchor extraction (no LLM, grounded)
# ============================================================
STOPWORDS_SHORT = {"The", "A", "An", "This", "That", "These", "Those", "It", "Its",
                   "Passage", "Title", "He", "She", "They", "We", "I", "You"}


def extract_anchors(title: str, body: str, max_items: int = 30) -> list[str]:
    """Pull verbatim entities/dates/numbers from `body` (and the title).

    Zero hallucination: every item must appear as-is in the source text.
    Priority: title > named-entity phrases > years > single proper nouns > numbers.
    """
    seen = set()
    title_anchors = []
    ent_anchors = []
    year_anchors = []
    single_anchors = []
    num_anchors = []

    # 1. Title itself (always as the first anchor)
    t = title.strip()
    if t:
        seen.add(t)
        title_anchors.append(t)

    # 2. Multi-word capitalized phrases (named entities)
    for m in ENT_RX.finditer(body):
        ent = m.group(0).strip()
        if ent in seen:
            continue
        toks = ent.split()
        if all(t in STOPWORDS_SHORT for t in toks):
            continue
        seen.add(ent)
        ent_anchors.append(ent)

    # 3. Years
    for m in YEAR_RX.finditer(body):
        y = m.group(0)
        if y in seen:
            continue
        seen.add(y)
        year_anchors.append(y)

    # 4. Single-token proper nouns NOT already absorbed
    for m in PROPER_RX.finditer(body):
        p = m.group(0).strip()
        if p in seen or p in STOPWORDS_SHORT or " " in p:
            continue
        if len(p) < 3:
            continue
        if any(p in s and s != p for s in seen):
            continue
        seen.add(p)
        single_anchors.append(p)

    # 5. Other numbers
    for m in NUM_RX.finditer(body):
        n = m.group(0)
        if n in seen:
            continue
        seen.add(n)
        num_anchors.append(n)

    # Combine by priority
    combined = title_anchors + ent_anchors + year_anchors + single_anchors + num_anchors
    return combined[:max_items]


def build_pec_cards(passages: list[tuple[str, str]],
                    max_cards_per_passage: int = 12) -> tuple[list[str], list[dict]]:
    """Build pointer-preserved evidence cards from verbatim source sentences.

    Each card is a compact, searchable object that keeps the exact source
    sentence and a pointer back to (passage, sentence). This avoids CTKS's
    main failure mode: abstractive summaries deleting answer spans.
    """
    cards = []
    meta = []
    for p_idx, (title, body) in enumerate(passages):
        sentences = split_sentences(body)
        kept = 0
        for s_idx, sent in enumerate(sentences):
            words = WORD_RX.findall(sent)
            if len(words) < 5:
                continue
            anchors = extract_anchors(title, sent, max_items=10)
            has_signal = anchors or YEAR_RX.search(sent) or NUM_RX.search(sent)
            if not has_signal and s_idx > 1:
                continue
            short_sent = " ".join(sent.split()[:80])
            card = (
                f"TITLE: {title}\n"
                f"ANCHORS: {', '.join(anchors)}\n"
                f"FACT: {short_sent}\n"
                f"PTR: p={p_idx};s={s_idx}"
            )
            cards.append(card)
            meta.append({
                "passage_idx": p_idx,
                "sentence_idx": s_idx,
                "title": title,
                "sentence": sent,
                "anchors": anchors,
            })
            kept += 1
            if kept >= max_cards_per_passage:
                break
    return cards, meta


def build_fact_only_cards(passages: list[tuple[str, str]],
                          max_cards_per_passage: int = 12) -> tuple[list[str], list[dict]]:
    """Bare-sentence cards: no TITLE/ANCHORS/PTR markup.

    Used by the pec_hop_fact_only ablation to isolate the contribution of the
    structured card schema. Sentence selection (length, signal heuristics) is
    identical to build_pec_cards so the only thing that changes is what gets
    embedded and presented to the reader.
    """
    cards: list[str] = []
    meta: list[dict] = []
    for p_idx, (title, body) in enumerate(passages):
        sentences = split_sentences(body)
        kept = 0
        for s_idx, sent in enumerate(sentences):
            words = WORD_RX.findall(sent)
            if len(words) < 5:
                continue
            anchors = extract_anchors(title, sent, max_items=10)
            has_signal = anchors or YEAR_RX.search(sent) or NUM_RX.search(sent)
            if not has_signal and s_idx > 1:
                continue
            short_sent = " ".join(sent.split()[:80])
            cards.append(short_sent)
            meta.append({
                "passage_idx": p_idx,
                "sentence_idx": s_idx,
                "title": title,
                "sentence": sent,
                "anchors": anchors,
            })
            kept += 1
            if kept >= max_cards_per_passage:
                break
    return cards, meta


def hydrate_cards(passages: list[tuple[str, str]],
                  card_meta: list[dict],
                  window: int = 1) -> list[str]:
    """Recover source sentences around retrieved cards using their pointers."""
    by_ptr = []
    seen = set()
    sentence_cache = {}
    for m in card_meta:
        p_idx = m["passage_idx"]
        s_idx = m["sentence_idx"]
        key = (p_idx, s_idx)
        if key in seen:
            continue
        seen.add(key)
        if p_idx not in sentence_cache:
            sentence_cache[p_idx] = split_sentences(passages[p_idx][1])
        sentences = sentence_cache[p_idx]
        lo = max(0, s_idx - window)
        hi = min(len(sentences), s_idx + window + 1)
        title = passages[p_idx][0]
        evidence = " ".join(sentences[lo:hi])
        by_ptr.append(f"{title}: {evidence}")
    return by_ptr


def evidence_is_weak(question: str,
                     retrieved_meta: list[dict],
                     scores: list[float],
                     min_titles: int = 2) -> bool:
    """Heuristic fallback gate for on-device use: cheap, answer-agnostic."""
    if not retrieved_meta:
        return True
    titles = {m["title"] for m in retrieved_meta}
    terms = query_terms(question)
    anchor_text = " ".join(
        [m["title"] for m in retrieved_meta] +
        [" ".join(m.get("anchors", [])) for m in retrieved_meta] +
        [m.get("sentence", "") for m in retrieved_meta[:3]]
    ).lower()
    overlap = sum(1 for t in terms if t in anchor_text)
    top_score = scores[0] if scores else 0.0
    return len(titles) < min_titles or overlap < 2 or top_score < 0.28


def retrieve_texts(retriever: ChunkIndex, query: str, texts: list[str],
                   top_k: int) -> tuple[list[str], list[int], list[float]]:
    """Build a transient FAISS index and return texts plus source indices."""
    retriever.build_from_chunks(texts, batch_size=128)
    retrieved, scores, _ = retriever.search(query, top_k=top_k)
    used_indices = []
    taken = set()
    for r in retrieved:
        idx = None
        for i, text in enumerate(texts):
            if i in taken:
                continue
            if text == r:
                idx = i
                break
        if idx is not None:
            taken.add(idx)
            used_indices.append(idx)
    return retrieved, used_indices, scores


# ============================================================
# Relation extraction (for pec_hop_relations variant)
# ============================================================
# Light-weight relation phrases: e.g. "directed by", "born in", "located in".
# We DO NOT need a parser; we look for verb-particle patterns where the verb
# is a typical biographical / structural relation. Lower-cased, copied verbatim
# from the source so it remains pointer-faithful.
REL_VERBS = (
    "directed", "produced", "written", "composed", "founded", "started",
    "released", "performed", "born", "died", "married", "located", "based",
    "headquartered", "owned", "acquired", "developed", "designed", "edited",
    "published", "translated", "starring", "starred", "managed", "operated",
    "served", "trained", "studied", "graduated", "elected", "appointed",
    "named", "called", "known", "set", "filmed", "shot",
)
REL_PREPS = ("by", "in", "at", "on", "of", "for", "from", "as", "to", "with")
# Pattern matches "<verb> <prep>" with optional adverb in between.
REL_RX = re.compile(
    r"\b(" + "|".join(REL_VERBS) + r")\b"
    r"(?:[ \t]+(?:[a-z]+ly|out|up|down|over|together))?"
    r"[ \t]+\b(" + "|".join(REL_PREPS) + r")\b",
    re.IGNORECASE,
)


def extract_relations(sent: str, max_items: int = 6) -> list[str]:
    """Extract verb-preposition relation phrases verbatim from a sentence."""
    out: list[str] = []
    seen: set[str] = set()
    for m in REL_RX.finditer(sent):
        phrase = m.group(0).strip().lower()
        # Normalize whitespace
        phrase = " ".join(phrase.split())
        if phrase in seen:
            continue
        seen.add(phrase)
        out.append(phrase)
        if len(out) >= max_items:
            break
    return out


def build_pec_cards_with_relations(
    passages: list[tuple[str, str]],
    max_cards_per_passage: int = 12,
) -> tuple[list[str], list[dict]]:
    """Same as build_pec_cards but adds a RELATIONS line and stores them in meta."""
    cards: list[str] = []
    meta: list[dict] = []
    for p_idx, (title, body) in enumerate(passages):
        sentences = split_sentences(body)
        kept = 0
        for s_idx, sent in enumerate(sentences):
            words = WORD_RX.findall(sent)
            if len(words) < 5:
                continue
            anchors = extract_anchors(title, sent, max_items=10)
            relations = extract_relations(sent, max_items=6)
            has_signal = anchors or YEAR_RX.search(sent) or NUM_RX.search(sent) or relations
            if not has_signal and s_idx > 1:
                continue
            short_sent = " ".join(sent.split()[:80])
            card = (
                f"TITLE: {title}\n"
                f"ANCHORS: {', '.join(anchors)}\n"
                f"RELATIONS: {', '.join(relations)}\n"
                f"FACT: {short_sent}\n"
                f"PTR: p={p_idx};s={s_idx}"
            )
            cards.append(card)
            meta.append({
                "passage_idx": p_idx,
                "sentence_idx": s_idx,
                "title": title,
                "sentence": sent,
                "anchors": anchors,
                "relations": relations,
            })
            kept += 1
            if kept >= max_cards_per_passage:
                break
    return cards, meta


# ============================================================
# Bridge-aware Minimal Copyable Evidence (MCE-RAG primitive)
# ============================================================
def question_anchors(question: str, max_items: int = 8) -> list[str]:
    """Extract entity-like anchors from the question itself.

    Uses the same regexes as passage anchor extraction. Capitalized phrases,
    years, numeric thresholds, single capitalized tokens.
    """
    seen: set[str] = set()
    out: list[str] = []

    for m in ENT_RX.finditer(question):
        ent = m.group(0).strip()
        if ent in seen:
            continue
        seen.add(ent)
        out.append(ent)

    for m in YEAR_RX.finditer(question):
        y = m.group(0)
        if y in seen:
            continue
        seen.add(y)
        out.append(y)

    for m in PROPER_RX.finditer(question):
        p = m.group(0).strip()
        if p in seen or p in STOPWORDS_SHORT or " " in p:
            continue
        if len(p) < 3:
            continue
        if any(p in s and s != p for s in seen):
            continue
        seen.add(p)
        out.append(p)

    for m in NUM_RX.finditer(question):
        n = m.group(0)
        if n in seen:
            continue
        seen.add(n)
        out.append(n)

    return out[:max_items]


# Question word -> answer-type bias for sentence selection
ANSWER_TYPE_CUES = {
    "when": ("year", "date"),
    "what year": ("year",),
    "how many": ("number",),
    "how much": ("number",),
    "where": ("location",),
    "who": ("person",),
    "which": ("entity",),
}


def question_answer_type(question: str) -> tuple[str, ...]:
    q = question.lower().strip()
    for cue, types in ANSWER_TYPE_CUES.items():
        if cue in q:
            return types
    return ()


def sentence_signal_score(sent: str, anchors: list[str], rel_overlap: int,
                          ans_types: tuple[str, ...]) -> float:
    """Lightweight signal score added on top of dense similarity.

    - +0.10 per question-entity contained verbatim
    - +0.05 if sentence contains a year and answer type wants year/date
    - +0.05 if sentence contains a number and answer type wants number
    - +0.02 per relation match
    """
    sent_lc = sent.lower()
    score = 0.0
    for a in anchors:
        if not a:
            continue
        if len(a) <= 3:
            # tiny anchor: require word-boundary
            if re.search(r"\b" + re.escape(a) + r"\b", sent, re.IGNORECASE):
                score += 0.10
        else:
            if a.lower() in sent_lc:
                score += 0.10
    if ans_types:
        if ("year" in ans_types or "date" in ans_types) and YEAR_RX.search(sent):
            score += 0.05
        if "number" in ans_types and NUM_RX.search(sent):
            score += 0.05
    score += 0.02 * rel_overlap
    return score


def build_bridge_sentence_evidence(
    passages: list[tuple[str, str]],
    question: str,
    retriever: ChunkIndex,
    max_sents: int = 4,
    max_words: int = 60,
) -> list[str]:
    """Construct a minimal copyable evidence set with bridge constraints.

    Pipeline:
      1. Split passages into sentence atoms (>= 5 words, has any signal).
      2. Score each sentence by dense FAISS sim + question-anchor overlap +
         answer-type cues.
      3. Greedy selection with title diversity and bridge constraints:
         - first sentence = top scorer
         - subsequent sentences must add a new title OR share an entity with
           an already-selected sentence (bridge link), and prefer sentences
           that bring new question-anchors into coverage.
      4. Cap at `max_sents` sentences and `max_words` words total.
    """
    # Step 1: atoms
    atoms: list[tuple[int, int, str, str, list[str]]] = []
    for p_idx, (title, body) in enumerate(passages):
        sents = split_sentences(body)
        for s_idx, sent in enumerate(sents):
            words = WORD_RX.findall(sent)
            if len(words) < 5:
                continue
            anchors = extract_anchors(title, sent, max_items=8)
            atoms.append((p_idx, s_idx, title, sent, anchors))

    if not atoms:
        return []

    # Step 2: dense scoring
    sent_texts = [" ".join(a[3].split()[:80]) for a in atoms]
    retriever.build_from_chunks(sent_texts, batch_size=128)
    # Get all scores via top_k = len(sent_texts) so we can sort uniformly.
    retrieved_sents, dense_scores, _ = retriever.search(
        question, top_k=min(len(sent_texts), 50)
    )
    # Build a map from sentence text -> score (best match)
    score_map: dict[str, float] = {}
    for s, sc in zip(retrieved_sents, dense_scores):
        if s not in score_map or sc > score_map[s]:
            score_map[s] = sc

    q_anchors = question_anchors(question)
    q_anchor_set = {a.lower() for a in q_anchors if a}
    ans_types = question_answer_type(question)

    scored: list[tuple[float, int]] = []
    for i, a in enumerate(atoms):
        text = sent_texts[i]
        d = score_map.get(text, -1.0)  # not retrieved -> low score
        anchors_lc = {x.lower() for x in a[4] if x}
        rel_overlap = len(q_anchor_set & anchors_lc)
        signal = sentence_signal_score(a[3], q_anchors, rel_overlap, ans_types)
        scored.append((d + signal, i))
    scored.sort(reverse=True)

    # Step 3: greedy selection
    selected: list[int] = []
    used_titles: set[str] = set()
    used_entities: set[str] = set()
    used_words = 0
    covered_q_anchors: set[str] = set()
    title_count: dict[str, int] = {}

    def add_sentence(idx: int) -> None:
        nonlocal used_words
        a = atoms[idx]
        selected.append(idx)
        used_titles.add(a[2])
        title_count[a[2]] = title_count.get(a[2], 0) + 1
        used_entities.update(x.lower() for x in a[4] if x)
        for qa in q_anchors:
            if qa and qa.lower() in a[3].lower():
                covered_q_anchors.add(qa.lower())
        used_words += len(a[3].split())

    # First: highest scorer that fits the evidence budget.
    for _, idx0 in scored:
        if len(atoms[idx0][3].split()) <= max_words:
            add_sentence(idx0)
            break

    # Greedy adds with relaxed bridge / diversity constraints.
    # The strict bridge rule starved evidence selection on tight budgets.
    # Revised policy:
    #  - Up to `max_sents` sentences, capped by `max_words` budget.
    #  - Penalize same-title sentences after the first 1 from that title
    #    (soft diversity, not hard exclusion).
    #  - Slight bonus for sentences that share an entity with already-
    #    selected ones (bridge link) or introduce a new question anchor.
    while len(selected) < max_sents:
        best: Optional[tuple[float, int]] = None
        for sc, idx in scored:
            if idx in selected:
                continue
            a = atoms[idx]
            sent_words = len(a[3].split())
            if used_words + sent_words > max_words:
                continue
            title = a[2]
            sent_anchors = {x.lower() for x in a[4] if x}
            bridge = bool(sent_anchors & used_entities)
            new_q_anchors = {qa for qa in q_anchors
                             if qa and qa.lower() in a[3].lower()
                             and qa.lower() not in covered_q_anchors}
            adj = sc
            adj += 0.05 if bridge else 0.0
            adj += 0.05 * len(new_q_anchors)
            if title_count.get(title, 0) >= 1:
                adj -= 0.05 * title_count.get(title, 0)
            if best is None or adj > best[0]:
                best = (adj, idx)
        if best is None:
            break
        add_sentence(best[1])

    # If every sentence exceeded the budget, still return a bounded excerpt
    # from the top sentence rather than falling back to the full context.
    if not selected and scored:
        _, idx0 = scored[0]
        a0 = atoms[idx0]
        words = a0[3].split()
        clipped = " ".join(words[:max_words])
        return [f"{a0[2]}: {clipped}"]

    for _, idx in scored:
        if len(selected) >= max_sents:
            break
        if len(selected) >= 2:
            break
        a = atoms[idx]
        sent_words = len(a[3].split())
        if used_words + sent_words > max_words:
            continue
        add_sentence(idx)

    # Fallback: ensure at least 2 sentences if any available
    if len(selected) < 2:
        for _, idx in scored:
            if idx in selected:
                continue
            a = atoms[idx]
            sent_words = len(a[3].split())
            if used_words + sent_words > max_words:
                continue
            selected.append(idx)
            used_words += sent_words
            if len(selected) >= 2:
                break

    # Build output strings: TITLE: sentence
    out: list[str] = []
    for idx in selected:
        a = atoms[idx]
        out.append(f"{a[2]}: {a[3]}")
    return out


# ============================================================
# Corruption helpers (negative controls for the card schema)
# ============================================================
import random as _rand_mod


def hydrate_cards_shuffled(passages: list[tuple[str, str]],
                           card_meta: list[dict],
                           window: int = 1,
                           seed: int = 0) -> list[str]:
    """Hydrate at RANDOM sentence indices instead of true PTR.

    Used by pec_hop_shuffle_ptr to test whether pointer-preserved hydration is
    actually load-bearing or whether any sentence window from the same title
    would do.
    """
    rng = _rand_mod.Random(seed)
    by_ptr = []
    seen = set()
    sentence_cache = {}
    for m in card_meta:
        p_idx = m["passage_idx"]
        true_s_idx = m["sentence_idx"]
        if p_idx not in sentence_cache:
            sentence_cache[p_idx] = split_sentences(passages[p_idx][1])
        sentences = sentence_cache[p_idx]
        if not sentences:
            continue
        # Pick a random sentence from the same passage that is NOT the true one.
        candidates = [i for i in range(len(sentences)) if i != true_s_idx]
        if not candidates:
            candidates = list(range(len(sentences)))
        bad_s = rng.choice(candidates)
        key = (p_idx, bad_s)
        if key in seen:
            continue
        seen.add(key)
        lo = max(0, bad_s - window)
        hi = min(len(sentences), bad_s + window + 1)
        title = passages[p_idx][0]
        evidence = " ".join(sentences[lo:hi])
        by_ptr.append(f"{title}: {evidence}")
    return by_ptr


def random_anchor_meta(card_meta: list[dict], all_words: list[str],
                       seed: int = 0) -> list[dict]:
    """Return a copy of meta where each card's anchors are replaced by random
    tokens from the full passage vocabulary. This is the negative control for
    anchor-driven step-2 retrieval (pec_hop_random_anchor).
    """
    rng = _rand_mod.Random(seed)
    if not all_words:
        return card_meta
    out = []
    for m in card_meta:
        n_anchors = max(1, len(m.get("anchors") or []))
        random_anchors = rng.sample(all_words, k=min(n_anchors, len(all_words)))
        new_m = dict(m)
        new_m["anchors"] = random_anchors
        out.append(new_m)
    return out


def collect_passage_vocab(passages: list[tuple[str, str]]) -> list[str]:
    """Tokens used as the random-anchor pool. Lower-cased, deduplicated, length>=4."""
    vocab: list[str] = []
    seen: set[str] = set()
    for _, body in passages:
        for w in WORD_RX.findall(body):
            if len(w) < 4:
                continue
            wl = w.lower()
            if wl in seen:
                continue
            seen.add(wl)
            vocab.append(w)
    return vocab


# ============================================================
# Per-passage compilation
# ============================================================
def compile_passage(title: str, body: str, variant: str,
                    llm_generator=None) -> str:
    """Produce the compiled string for one passage under a given variant."""
    if variant == "raw_topk":
        # Keep raw body but truncate to ~150 words to fit many in context
        words = body.split()
        return f"{title}: {' '.join(words[:200])}"

    if variant == "anchors":
        anchors = extract_anchors(title, body)
        return f"{title}: ANCHORS: {', '.join(anchors)}"

    if variant in ("pec_card", "pec_hydrate", "pec_adaptive"):
        raise ValueError("PEC variants are compiled at the passage-set level")

    if variant in ("summary", "anchored"):
        words = body.split()
        truncated_body = " ".join(words[:300])
        prompt = SUMMARY_PROMPT.format(title=title, body=truncated_body)
        output = llm_generator.llm(
            prompt,
            max_tokens=120,
            temperature=0.0,
            echo=False,
            stop=["\n\n", "Title:", "Passage:", "<|im_end|>"],
            repeat_penalty=1.1,
        )
        summary = output["choices"][0]["text"].strip().replace("\n", " ")
        if not summary or len(summary) < 5:
            summary = " ".join(words[:25])
        if variant == "summary":
            return f"{title}: {summary}"
        anchors = extract_anchors(title, body)
        return f"{title}: {summary} | ANCHORS: {', '.join(anchors)}"

    raise ValueError(f"Unknown variant: {variant}")


# ============================================================
# Evaluation metrics
# ============================================================
def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = " ".join(s.split())
    return s


def em_score(pred: str, answers: list[str]) -> int:
    p = normalize_answer(pred)
    return int(any(normalize_answer(a) == p for a in answers))


def f1_score(pred: str, answers: list[str]) -> float:
    p = normalize_answer(pred).split()
    if not p:
        return 0.0
    best = 0.0
    for a in answers:
        g = normalize_answer(a).split()
        if not g:
            continue
        common = set(p) & set(g)
        if not common:
            continue
        prec = len(common) / len(p)
        rec = len(common) / len(g)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def answer_in_pred(pred: str, answers: list[str]) -> int:
    p = normalize_answer(pred)
    return int(any(normalize_answer(a) in p for a in answers if a))


# ============================================================
# Main run
# ============================================================
QA_PROMPT_TMPL = (
    "Answer the following question based on the provided documents. "
    "Give a short, direct answer.\n\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

# Extractive answer-only prompt: encourages the reader to copy a short answer
# span from the evidence. Avoids "unknown" fallback so the model still attempts
# an answer. Targets the 2WikiMQA EM verbosity issue.
QA_EXTRACTIVE_PROMPT_TMPL = (
    "Read the short evidence snippets and answer the question by copying the "
    "shortest span from the evidence that directly answers it. "
    "Reply with at most 5 words. Do not write a full sentence.\n\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Short answer:"
)

# Variant prompts to test whether the multifield F1 drop and Qwen collapse are
# driven by the hard 5-word cap. Each one keeps the "copy the shortest span"
# instruction but relaxes the length restriction differently.
QA_SHORT15_PROMPT_TMPL = (
    "Read the short evidence snippets and answer the question by copying the "
    "shortest span from the evidence that directly answers it. "
    "Keep the answer under 15 words and avoid full sentences when a span suffices.\n\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Short answer:"
)
QA_SPAN_PROMPT_TMPL = (
    "Read the short evidence snippets and answer the question by copying the "
    "answer span from the evidence. Output only the span, not a sentence.\n\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Answer span:"
)
QA_CONCISE_PROMPT_TMPL = (
    "Answer the following question based on the provided documents. "
    "Be concise: prefer the exact answer span over a full sentence.\n\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

# Single-shot extractive prompt: only the Einstein demonstration.
# Ablation control for whether the 2-shot pattern is necessary.
QA_ONEEXTRACTIVE_PROMPT_TMPL = (
    "Read the short evidence snippets and answer the question by copying the "
    "shortest span from the evidence that directly answers it. "
    "Reply with at most 5 words. Do not write a full sentence.\n\n"
    "Example:\n"
    "[Document 1 (HYDRATED)]: Albert Einstein: Albert Einstein was born in Ulm in 1879.\n"
    "[Document 2 (HYDRATED)]: Ulm: Ulm is a city in Baden-Wurttemberg, Germany.\n"
    "Question: In what country was Albert Einstein born?\n"
    "Short answer: Germany\n\n"
    "Now answer this question:\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Short answer:"
)

# Few-shot extractive prompt: two demonstrations of bridge-style multi-hop
# extraction with short, copy-only answers. Adds ~140 reader tokens but is
# designed to lift the EM/loose conversion rate that plain extractive leaves
# on the table (SLMs follow demonstrated answer shape).
QA_FEWEXTRACTIVE_PROMPT_TMPL = (
    "Read the short evidence snippets and answer the question by copying the "
    "shortest span from the evidence that directly answers it. "
    "Reply with at most 5 words. Do not write a full sentence.\n\n"
    "Example 1:\n"
    "[Document 1 (HYDRATED)]: Albert Einstein: Albert Einstein was born in Ulm in 1879.\n"
    "[Document 2 (HYDRATED)]: Ulm: Ulm is a city in Baden-Wurttemberg, Germany.\n"
    "Question: In what country was Albert Einstein born?\n"
    "Short answer: Germany\n\n"
    "Example 2:\n"
    "[Document 1 (HYDRATED)]: Inception: Inception is a 2010 film directed by Christopher Nolan.\n"
    "[Document 2 (HYDRATED)]: Christopher Nolan: Christopher Nolan was born on July 30, 1970.\n"
    "Question: When was the director of Inception born?\n"
    "Short answer: July 30, 1970\n\n"
    "Now answer this question:\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Short answer:"
)


# ============================================================
# Adaptive router (heuristic features, no training)
# ============================================================
RELATION_CUE_WORDS = (
    "directed", "produced", "written", "wrote", "founded", "released",
    "performed", "born", "died", "married", "located", "based",
    "owned", "acquired", "developed", "designed", "edited",
    "published", "starring", "starred",
    "spouse", "father", "mother", "son", "daughter", "brother", "sister",
    "parent", "child", "predecessor", "successor",
    "between", "before", "after", "than",
)


def question_relation_cue(question: str) -> bool:
    q = question.lower()
    return any(c in q for c in RELATION_CUE_WORDS)


def router_decision(question: str,
                    selected_meta: list[dict],
                    scores: list[float]) -> str:
    """Pick a PEC variant per query based on cheap features.

    Returns one of:
      - 'pec_hop'        (default; best on single-hop and clean evidence)
      - 'pec_bridge'     (weak / low-score signal -> mix in 2 raw passages)
      - 'pec_bridge_k3'  (multi-hop relation question -> mix in 3 raw passages)
    """
    if not selected_meta:
        return "pec_bridge"
    titles = {m["title"] for m in selected_meta}
    n_titles = len(titles)
    top_score = scores[0] if scores else 0.0
    score_gap = (scores[0] - scores[min(4, len(scores) - 1)]) if len(scores) >= 5 else 0.0

    # Strong relation cue + the top-k spreads across few titles -> need a wider
    # raw window to catch the bridge fact.
    if question_relation_cue(question) and n_titles <= 3:
        return "pec_bridge_k3"
    # Weak retrieval (existing heuristic, plus low score gap)
    if top_score < 0.30 or score_gap < 0.02 or n_titles < 2:
        return "pec_bridge"
    return "pec_hop"


def run_task(task: str, model_name: str, variant: str,
             top_k: int, max_context_tokens: int,
             n_samples: Optional[int] = None):
    data_path = DATASETS_DIR / "longbench" / "data" / f"{task}.jsonl"
    items = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    if n_samples and n_samples < len(items):
        items = items[:n_samples]

    out_dir = TOPIC6_DIR / "experiments" / "results" / "longbench"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_tag = f"_n{n_samples}" if n_samples is not None else ""
    topk_tag = f"_k{top_k}" if top_k != 5 else ""
    out_file = out_dir / f"{model_name}_{task}_{variant}{sample_tag}{topk_tag}.json"
    if out_file.exists():
        print(f"[Skip] {out_file} already exists")
        with open(out_file, encoding="utf-8") as f:
            cached = json.load(f)
        print(f"  EM={cached['metrics']['em']:.2f}% "
              f"F1={cached['metrics']['f1']:.2f}% "
              f"n={cached['n_samples']}")
        return cached

    # Load model
    model_info = GGUF_MODELS[model_name]
    model_path = str(MODELS_DIR / model_info["file"])
    print(f"[Model] Loading {model_name}")
    gen = LLMGenerator(model_path, n_ctx=2048, n_threads=4, n_gpu_layers=-1)

    # Resolve (retrieval_variant, prompt_kind) once. Cross-prompt compound
    # variants (e.g. raw_topk_extractive) reuse the base retrieval + a
    # different decoding prompt at the end of the per-item loop.
    retrieval_variant, prompt_kind = parse_variant(variant)

    # Separate (lightweight) index just for retrieval
    retriever = ChunkIndex() if retrieval_variant in (
        "raw_topk", "summary", "anchors", "anchored",
        "hybrid2", "hybrid3", "pec_card", "pec_hydrate", "pec_adaptive",
        "pec_bridge", "pec_bridge_k3", "pec_hop", "pec_query_expand",
        "pec_hop_no_anchor", "pec_hop_fact_only", "pec_hop_no_hydration",
        "pec_hop_w0", "pec_hop_w2", "pec_hop_dynamic",
        "raw_topk_b840",
        "sentence_only", "pec_router",
        "pec_hop_relations", "pec_hop_shuffle_ptr", "pec_hop_random_anchor",
        "bridge_sentence",
        # External compressors retrieve top-k like raw_topk, then compress.
        "llmlingua2", "provence",
        # PEC-Hop+distill: retrieval narrows passages, sentence picker distills.
        "pec_hop_distill",
        # PEC-Hop+rerank: anchor-overlap + answer-type rerank of seed cards.
        "pec_hop_rerank",
    ) else None

    # Lazy-load external compressor models. Both run on the same GPU as the
    # llama.cpp reader; ~570M (llmlingua2) and ~430M (provence) leave ample
    # headroom on a 24GB 3090 alongside the 1.2B Q4 reader.
    compressor = None
    if retrieval_variant == "llmlingua2":
        from llmlingua import PromptCompressor
        print("[Compressor] Loading microsoft/llmlingua-2-xlm-roberta-large-meetingbank")
        compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
            device_map="cuda",
        )
    elif retrieval_variant == "provence":
        from transformers import AutoModel
        import torch as _torch
        print("[Compressor] Loading naver/provence-reranker-debertav3-v1")
        compressor = AutoModel.from_pretrained(
            "naver/provence-reranker-debertav3-v1",
            trust_remote_code=True,
        ).to("cuda").eval()

    all_em, all_f1, all_loose = [], [], []
    records = []
    for item in tqdm(items, desc=f"{task}/{variant}"):
        question = item["input"]
        context = item["context"]
        answers = item["answers"] if isinstance(item["answers"], list) else [item["answers"]]

        t_pipeline = time.perf_counter()

        # Budget: n_ctx=2048, reserve ~200 for prompt, ~120 for answer, rest for ctx.
        # Use actual llama tokenizer to enforce the limit.
        # raw_topk_b840 budget-matches PEC-Hop's ~840-token ctx for fair comparison.
        MAX_CTX_TOKENS = 840 if retrieval_variant == "raw_topk_b840" else max_context_tokens
        fallback_used = False
        selected_titles = []

        # Use retrieval_variant for the dispatch tree; the original variant string
        # only governs prompt selection at the very end of the loop.
        rv = retrieval_variant

        if rv == "raw_trunc":
            ctx_str = context
        elif rv in ("hybrid2", "hybrid3"):
            n_raw = 2 if variant == "hybrid2" else 3
            passages = split_passages(context)
            raw_chunks = [f"{t}: {' '.join(b.split()[:200])}" for t, b in passages]
            retriever.build_from_chunks(raw_chunks, batch_size=128)
            top_raw_texts, _, _ = retriever.search(question, top_k=n_raw)
            used_idx = set()
            for r in top_raw_texts:
                for i, c in enumerate(raw_chunks):
                    if c == r:
                        used_idx.add(i)
                        break
            rest = [(i, passages[i][0], passages[i][1]) for i in range(len(passages))
                    if i not in used_idx]
            rest_compiled = []
            for _, t, b in rest:
                note = compile_passage(t, b, "anchored", llm_generator=gen)
                rest_compiled.append(note)
            parts = []
            for i, r in enumerate(top_raw_texts):
                parts.append(f"[Document {i+1} (RAW)]: {r}")
            for i, rc in enumerate(rest_compiled):
                parts.append(f"[Document {i+1+len(top_raw_texts)} (NOTE)]: {rc}")
            ctx_str = "\n\n".join(parts)
        elif rv == "sentence_only":
            # Bare-sentence retrieval baseline: split into sentences, single-stage
            # FAISS top-k, no schema, no anchor expansion, no hydration.
            # This is the strongest "novelty defense" baseline for PEC.
            passages = split_passages(context)
            sent_texts: list[str] = []
            for _, body in passages:
                for s in split_sentences(body):
                    if len(WORD_RX.findall(s)) >= 5:
                        sent_texts.append(" ".join(s.split()[:80]))
            if not sent_texts:
                ctx_str = context
                fallback_used = True
            else:
                retriever.build_from_chunks(sent_texts, batch_size=128)
                top_sents, _, _ = retriever.search(question, top_k=top_k)
                ctx_str = "\n\n".join(
                    f"[Document {i+1} (SENT)]: {c}" for i, c in enumerate(top_sents)
                )
        elif rv == "pec_hop_distill":
            # Two-stage minimal copyable evidence:
            #   (a) PEC card retrieval picks the right 3-5 passages out of 10
            #   (b) bridge-sentence picker keeps only top-3 answer-relevant
            #       sentences inside that subset (~150 reader tokens)
            # Goal: match Provence-sized prompt while remaining non-parametric.
            passages = split_passages(context)
            cards, card_meta = build_pec_cards(passages)
            if not cards:
                ctx_str = context
                fallback_used = True
            else:
                retrieved_cards, card_indices, card_scores = retrieve_texts(
                    retriever, question, cards, top_k=top_k
                )
                selected_meta = [card_meta[i] for i in card_indices]
                selected_titles = [m["title"] for m in selected_meta]
                # Restrict to passages those cards point at (typically 3-5/10)
                used_p: list[int] = []
                seen_p: set[int] = set()
                for m in selected_meta:
                    pi = m["passage_idx"]
                    if pi not in seen_p:
                        seen_p.add(pi)
                        used_p.append(pi)
                sub_passages = [passages[i] for i in used_p]
                evidence = build_bridge_sentence_evidence(
                    sub_passages, question, retriever,
                    max_sents=3, max_words=100,
                )
                if not evidence:
                    ctx_str = context
                    fallback_used = True
                else:
                    ctx_str = "\n\n".join(
                        f"[Document {i+1} (DISTILL)]: {c}" for i, c in enumerate(evidence)
                    )
        elif rv == "llmlingua2":
            # Retrieve top-k raw passages (same input as raw_topk), then
            # apply LLMLingua-2 token-level keep/drop compression.
            # Target rate ~0.6 keeps roughly the PEC-Hop budget (~840 reader
            # tokens vs raw_topk's ~1400) on average, so this is a budget-
            # matched extractive compressor head-to-head.
            passages = split_passages(context)
            raw_chunks = [f"{t}: {' '.join(b.split()[:200])}" for t, b in passages]
            if not raw_chunks:
                ctx_str = context
                fallback_used = True
            else:
                retriever.build_from_chunks(raw_chunks, batch_size=128)
                top_texts, _, _ = retriever.search(question, top_k=top_k)
                try:
                    comp = compressor.compress_prompt(
                        top_texts,
                        rate=0.6,
                        force_tokens=["\n", "?", ".", ","],
                    )
                    ctx_str = comp.get("compressed_prompt", "") or "\n".join(top_texts)
                except Exception as e:
                    # On rare tokenizer edge-cases fall back to uncompressed
                    print(f"[llmlingua2] fallback for question: {e}")
                    ctx_str = "\n\n".join(top_texts)
                    fallback_used = True
        elif rv == "provence":
            # Retrieve top-k raw passages, then prune each independently with
            # Provence (encoder-based sentence pruner). Concatenate pruned
            # outputs. Provence picks its own keep-rate per passage from the
            # learned cross-encoder; we report the resulting size as-is.
            passages = split_passages(context)
            raw_chunks = [f"{t}: {' '.join(b.split()[:200])}" for t, b in passages]
            if not raw_chunks:
                ctx_str = context
                fallback_used = True
            else:
                retriever.build_from_chunks(raw_chunks, batch_size=128)
                top_texts, _, _ = retriever.search(question, top_k=top_k)
                parts = []
                for i, txt in enumerate(top_texts):
                    try:
                        out = compressor.process(question, txt)
                        pruned = out.get("pruned_context", "") if isinstance(out, dict) else ""
                        if pruned and pruned.strip():
                            parts.append(f"[Document {i+1}]: {pruned.strip()}")
                    except Exception as e:
                        print(f"[provence] passage {i} fallback: {e}")
                        parts.append(f"[Document {i+1}]: {txt}")
                if not parts:
                    ctx_str = "\n\n".join(top_texts)
                    fallback_used = True
                else:
                    ctx_str = "\n\n".join(parts)
        elif rv == "bridge_sentence":
            # MCE-RAG primitive: bridge-aware minimal copyable evidence.
            # Dense + entity-overlap + answer-type scoring, with soft title
            # diversity and bridge bonuses; budget targeted at ~150 reader tokens.
            passages = split_passages(context)
            evidence = build_bridge_sentence_evidence(
                passages, question, retriever,
                max_sents=5, max_words=120,
            )
            if not evidence:
                ctx_str = context
                fallback_used = True
            else:
                ctx_str = "\n\n".join(
                    f"[Document {i+1} (BRIDGE)]: {c}" for i, c in enumerate(evidence)
                )
                selected_titles = [e.split(":", 1)[0] for e in evidence]
        elif rv in ("pec_card", "pec_hydrate", "pec_adaptive", "pec_bridge",
                          "pec_bridge_k3", "pec_hop", "pec_query_expand",
                          "pec_hop_no_anchor", "pec_hop_fact_only",
                          "pec_hop_no_hydration",
                          "pec_hop_w0", "pec_hop_w2", "pec_hop_dynamic",
                          "pec_router", "pec_hop_span",
                          "pec_hop_relations", "pec_hop_shuffle_ptr",
                          "pec_hop_random_anchor",
                          "pec_hop_rerank"):
            passages = split_passages(context)
            if rv == "pec_hop_fact_only":
                cards, card_meta = build_fact_only_cards(passages)
            elif rv == "pec_hop_relations":
                cards, card_meta = build_pec_cards_with_relations(passages)
            else:
                cards, card_meta = build_pec_cards(passages)
            if not cards:
                ctx_str = context
                fallback_used = True
            else:
                # pec_hop_rerank: pull a wider seed pool, then rerank by dense
                # similarity + question-anchor overlap + answer-type cue, take
                # top-k. Same downstream hydration / anchor expansion as pec_hop.
                if rv == "pec_hop_rerank":
                    seed_k = min(len(cards), top_k * 2)
                    retrieved_cards, card_indices, card_scores = retrieve_texts(
                        retriever, question, cards, top_k=seed_k
                    )
                    q_anchors_list = question_anchors(question)
                    q_anchors_set = {a.lower() for a in q_anchors_list if a}
                    ans_types = question_answer_type(question)
                    rescored: list[tuple[float, int, float]] = []
                    for rank, ci in enumerate(card_indices):
                        m = card_meta[ci]
                        card_anchor_set = {a.lower() for a in m.get("anchors", []) if a}
                        rel_overlap = len(q_anchors_set & card_anchor_set)
                        signal = sentence_signal_score(
                            m["sentence"], q_anchors_list, rel_overlap, ans_types,
                        )
                        rescored.append((
                            card_scores[rank] + signal, ci, card_scores[rank]
                        ))
                    rescored.sort(reverse=True)
                    card_indices = [t[1] for t in rescored[:top_k]]
                    card_scores = [t[2] for t in rescored[:top_k]]
                    retrieved_cards = [cards[i] for i in card_indices]
                else:
                    retrieved_cards, card_indices, card_scores = retrieve_texts(
                        retriever, question, cards, top_k=top_k
                    )
                selected_meta = [card_meta[i] for i in card_indices]
                selected_titles = [m["title"] for m in selected_meta]
                weak = evidence_is_weak(question, selected_meta, card_scores)

                # Dynamic variant: dispatch based on evidence strength.
                # Weak retrieval -> PEC-Bridge prompt (raw + cards). Strong -> PEC-Hop.
                if rv == "pec_hop_dynamic":
                    variant_eff = "pec_bridge" if weak else "pec_hop"
                elif rv == "pec_router":
                    # Heuristic adaptive router: hop / bridge / bridge_k3.
                    variant_eff = router_decision(question, selected_meta, card_scores)
                elif rv == "pec_hop_span":
                    variant_eff = "pec_hop"
                elif rv == "pec_hop_relations":
                    # Treat as pec_hop for the dispatch tree; the difference is
                    # that anchors (which now include relations) feed step-2.
                    variant_eff = "pec_hop"
                elif rv == "pec_hop_shuffle_ptr":
                    # Negative control: use pec_hop retrieval, but corrupt the
                    # final hydration to point at random sentences.
                    variant_eff = "pec_hop"
                elif rv == "pec_hop_random_anchor":
                    # Negative control: replace anchors with random tokens BEFORE
                    # step-2 query expansion, then run pec_hop.
                    variant_eff = "pec_hop"
                    pool = collect_passage_vocab(passages)
                    selected_meta = random_anchor_meta(selected_meta, pool, seed=0)
                elif rv == "pec_hop_rerank":
                    # Re-ranked seeds; downstream behavior identical to pec_hop.
                    variant_eff = "pec_hop"
                else:
                    variant_eff = rv

                # For pec_hop_relations, splice relation phrases into the anchor
                # list so they participate in the step-2 query string.
                if rv == "pec_hop_relations":
                    for m in selected_meta[:3]:
                        rel = m.get("relations") or []
                        m["anchors"] = list(m.get("anchors", [])) + list(rel)

                if variant_eff in ("pec_bridge", "pec_bridge_k3"):
                    n_raw = 2 if variant_eff == "pec_bridge" else 3
                    raw_chunks = [f"{t}: {' '.join(b.split()[:200])}" for t, b in passages]
                    raw_texts, raw_indices, _ = retrieve_texts(
                        retriever, question, raw_chunks, top_k=n_raw
                    )
                    raw_title_set = {passages[i][0] for i in raw_indices}
                    bridge_meta = [
                        m for m in selected_meta
                        if m["title"] not in raw_title_set
                    ]
                    hydrated = hydrate_cards(
                        passages, bridge_meta[:max(1, top_k - n_raw)], window=1
                    )
                    parts = []
                    for i, r in enumerate(raw_texts):
                        parts.append(f"[Document {i+1} (RAW)]: {r}")
                    offset = len(parts)
                    for i, h in enumerate(hydrated):
                        parts.append(f"[Document {offset+i+1} (HYDRATED)]: {h}")
                    ctx_str = "\n\n".join(parts)
                elif variant_eff in ("pec_hop", "pec_hop_no_anchor",
                                   "pec_hop_fact_only", "pec_hop_no_hydration",
                                   "pec_hop_w0", "pec_hop_w2"):
                    # PEC-Hop and ablations / hydration sweep variants.
                    use_anchor_expansion = variant_eff in (
                        "pec_hop", "pec_hop_no_hydration",
                        "pec_hop_w0", "pec_hop_w2"
                    )
                    use_hydration = variant_eff in (
                        "pec_hop", "pec_hop_no_anchor",
                        "pec_hop_w0", "pec_hop_w2"
                    )
                    # Hydration window varies by variant.
                    hyd_window = {
                        "pec_hop_w0": 0,
                        "pec_hop_w2": 2,
                    }.get(variant_eff, 1)
                    # Build step-2 query.
                    if use_anchor_expansion:
                        seed_anchors: list[str] = []
                        for m in selected_meta[:3]:
                            for a in m.get("anchors", [])[:4]:
                                seed_anchors.append(a)
                            seed_anchors.append(m["title"])
                        seen_a: set[str] = set()
                        dedup_anchors: list[str] = []
                        for a in seed_anchors:
                            if not a:
                                continue
                            key = a.lower()
                            if key in seen_a:
                                continue
                            seen_a.add(key)
                            dedup_anchors.append(a)
                        step2_query = (question + " " + " ".join(dedup_anchors[:8])).strip()
                    else:
                        step2_query = question
                    # Step 2 retrieval reuses the same card index.
                    expanded_results, _, _ = retriever.search(
                        step2_query, top_k=top_k * 2
                    )
                    expanded_meta = []
                    seen_idx = set(card_indices)
                    for r in expanded_results:
                        for i, c in enumerate(cards):
                            if i in seen_idx:
                                continue
                            if c == r:
                                expanded_meta.append(card_meta[i])
                                seen_idx.add(i)
                                break
                        if len(expanded_meta) >= top_k:
                            break
                    half_seed = max(2, top_k - len(expanded_meta))
                    final_meta = (selected_meta[:half_seed] + expanded_meta)[:top_k + 2]
                    # Final prompt construction.
                    if use_hydration:
                        if rv == "pec_hop_shuffle_ptr":
                            hydrated = hydrate_cards_shuffled(
                                passages, final_meta, window=hyd_window, seed=0
                            )
                        else:
                            hydrated = hydrate_cards(passages, final_meta, window=hyd_window)
                        tag = "HOP" if variant_eff == "pec_hop" else (
                            f"W{hyd_window}" if variant_eff in ("pec_hop_w0", "pec_hop_w2")
                            else "NO-ANCH"
                        )
                        ctx_str = "\n\n".join(
                            f"[Document {i+1} ({tag})]: {c}"
                            for i, c in enumerate(hydrated)
                        )
                    else:
                        # No hydration: present each card's stored sentence (or full
                        # structured card text for pec_hop_no_hydration; bare sentence
                        # for pec_hop_fact_only).
                        if variant_eff == "pec_hop_fact_only":
                            chunks = [m["sentence"] for m in final_meta]
                            tag = "FACT"
                        else:
                            # pec_hop_no_hydration: keep full structured card text
                            final_idx = []
                            for m in final_meta:
                                for i, mm in enumerate(card_meta):
                                    if mm is m:
                                        final_idx.append(i); break
                            chunks = [cards[i] for i in final_idx]
                            tag = "CARD"
                        ctx_str = "\n\n".join(
                            f"[Document {i+1} ({tag})]: {c}"
                            for i, c in enumerate(chunks)
                        )
                    selected_titles = [m["title"] for m in final_meta]
                elif variant == "pec_query_expand":
                    # Cheap LLM rewrite to surface hop-2 entities, then
                    # union retrievals from original + rewritten query.
                    out = gen.llm(
                        QUERY_EXPAND_PROMPT.format(question=question),
                        max_tokens=40, temperature=0.0, echo=False,
                        stop=["\n", "Question:", "<|im_end|>"],
                        repeat_penalty=1.1,
                    )
                    rewrite = out["choices"][0]["text"].strip()
                    expanded_query = (question + " " + rewrite).strip()
                    expanded_results, _, _ = retriever.search(
                        expanded_query, top_k=top_k * 2
                    )
                    expanded_meta = []
                    seen_idx = set(card_indices)
                    for r in expanded_results:
                        for i, c in enumerate(cards):
                            if i in seen_idx:
                                continue
                            if c == r:
                                expanded_meta.append(card_meta[i])
                                seen_idx.add(i)
                                break
                        if len(expanded_meta) >= top_k:
                            break
                    half_seed = top_k // 2 + 1
                    half_exp = top_k - half_seed + 2
                    final_meta = selected_meta[:half_seed] + expanded_meta[:half_exp]
                    hydrated = hydrate_cards(passages, final_meta, window=1)
                    ctx_str = "\n\n".join(
                        f"[Document {i+1} (EXPAND)]: {c}"
                        for i, c in enumerate(hydrated)
                    )
                elif variant == "pec_adaptive" and weak:
                    raw_chunks = [f"{t}: {' '.join(b.split()[:200])}" for t, b in passages]
                    raw_texts, _, _ = retrieve_texts(retriever, question, raw_chunks, top_k=top_k)
                    ctx_str = "\n\n".join(
                        f"[Document {i+1} (RAW-FALLBACK)]: {c}"
                        for i, c in enumerate(raw_texts)
                    )
                    fallback_used = True
                elif variant == "pec_card":
                    ctx_str = "\n\n".join(
                        f"[Document {i+1} (CARD)]: {c}"
                        for i, c in enumerate(retrieved_cards)
                    )
                else:
                    hydrated = hydrate_cards(passages, selected_meta, window=1)
                    ctx_str = "\n\n".join(
                        f"[Document {i+1} (HYDRATED)]: {c}"
                        for i, c in enumerate(hydrated)
                    )
        else:
            passages = split_passages(context)
            # raw_topk_b840 reuses raw_topk's compilation (truncate via budget below)
            compile_variant = "raw_topk" if rv == "raw_topk_b840" else rv
            compiled = [compile_passage(t, b, compile_variant, llm_generator=gen) for t, b in passages]
            # For short compiled forms, check if all fit within token budget.
            # If yes, use all (avoid lossy retrieval on tiny notes); else retrieve top-k.
            total_words = sum(len(c.split()) for c in compiled)
            est_tokens = int(total_words * 1.4)
            use_all = (rv in ("summary", "anchors", "anchored")
                       and est_tokens <= MAX_CTX_TOKENS - 100)
            if use_all:
                texts = compiled
            elif rv in ("raw_topk", "raw_topk_b840") or len(compiled) > top_k:
                retriever.build_from_chunks(compiled, batch_size=128)
                texts, _, _ = retriever.search(question, top_k=top_k)
            else:
                texts = compiled
            ctx_str = "\n\n".join(f"[Document {i+1}]: {c}" for i, c in enumerate(texts))

        # Hard token-level truncation using the actual llama tokenizer
        ctx_bytes = ctx_str.encode("utf-8", errors="ignore")
        ctx_tokens = gen.llm.tokenize(ctx_bytes)
        if len(ctx_tokens) > MAX_CTX_TOKENS:
            ctx_tokens = ctx_tokens[:MAX_CTX_TOKENS]
            try:
                ctx_str = gen.llm.detokenize(ctx_tokens).decode("utf-8", errors="ignore")
            except Exception:
                # Fallback: character-based cut
                char_ratio = len(ctx_str) // max(len(ctx_str.split()), 1)
                ctx_str = ctx_str[: MAX_CTX_TOKENS * char_ratio]

        # Final answer call. The decoding-prompt axis is now fully decoupled
        # from the retrieval-variant axis: any base (raw_topk, sentence_only,
        # pec_hop, ...) can be paired with any prompt suffix
        # (extractive / short15 / concise / default).
        prompt_tmpl = QA_PROMPT_TMPL
        if prompt_kind == "extractive":
            prompt_tmpl = QA_EXTRACTIVE_PROMPT_TMPL
        elif prompt_kind == "short15":
            prompt_tmpl = QA_SHORT15_PROMPT_TMPL
        elif prompt_kind == "concise":
            prompt_tmpl = QA_CONCISE_PROMPT_TMPL
        elif prompt_kind == "fewextractive":
            prompt_tmpl = QA_FEWEXTRACTIVE_PROMPT_TMPL
        elif prompt_kind == "oneextractive":
            prompt_tmpl = QA_ONEEXTRACTIVE_PROMPT_TMPL
        elif rv == "pec_hop_span":
            prompt_tmpl = QA_SPAN_PROMPT_TMPL
        prompt = prompt_tmpl.format(context=ctx_str, question=question)
        output = gen.llm(
            prompt,
            max_tokens=80,
            temperature=0.0,
            echo=False,
            stop=["\n", "\n\n", "Question:", "<|im_end|>"],
            repeat_penalty=1.1,
        )
        pred = output["choices"][0]["text"].strip().split("\n")[0]
        elapsed_ms = (time.perf_counter() - t_pipeline) * 1000

        em = em_score(pred, answers)
        f1 = f1_score(pred, answers)
        loose = answer_in_pred(pred, answers)
        all_em.append(em)
        all_f1.append(f1)
        all_loose.append(loose)

        records.append({
            "question": question,
            "prediction": pred,
            "answers": answers,
            "em": em,
            "f1": f1,
            "loose": loose,
            "latency_ms": round(elapsed_ms, 1),
            "fallback_used": fallback_used,
            "selected_titles": selected_titles,
            "context_tokens": len(ctx_tokens),
        })

    summary = {
        "task": task,
        "model": model_name,
        "variant": variant,
        "n_samples": len(records),
        "top_k": top_k,
        "metrics": {
            "em": round(sum(all_em) / len(all_em) * 100, 2),
            "f1": round(sum(all_f1) / len(all_f1) * 100, 2),
            "loose": round(sum(all_loose) / len(all_loose) * 100, 2),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in records) / len(records), 1),
            "fallback_rate": round(
                sum(1 for r in records if r.get("fallback_used")) / len(records) * 100, 2
            ),
            "avg_context_tokens": round(
                sum(r.get("context_tokens", 0) for r in records) / len(records), 1
            ),
        },
        "results": records,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    m = summary["metrics"]
    print(f"[Saved] {out_file}")
    print(f"  EM={m['em']:.2f}% F1={m['f1']:.2f}% loose={m['loose']:.2f}%"
          f" lat={m['avg_latency_ms']}ms")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="hotpotqa")
    parser.add_argument("--model", default="lfm2.5-1.2b-instruct")
    parser.add_argument("--variant", default="anchored")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-tokens", type=int, default=1500,
                        help="Context budget for raw_trunc and final QA call")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Subsample for quick testing")
    args = parser.parse_args()

    tasks = TASKS if args.task == "all" else [args.task]
    variants = VARIANTS if args.variant == "all" else [args.variant]

    rows = []
    for t in tasks:
        for v in variants:
            res = run_task(t, args.model, v,
                           top_k=args.top_k,
                           max_context_tokens=args.max_context_tokens,
                           n_samples=args.n_samples)
            if res:
                rows.append((t, v, res["metrics"]))

    print("\n=== Summary ===")
    print(f"{'Task':<12} {'Variant':<12} {'EM':>8} {'F1':>8} {'Loose':>8} {'Lat(ms)':>10}")
    for t, v, m in rows:
        print(f"{t:<12} {v:<12} {m['em']:>7.2f}% {m['f1']:>7.2f}% "
              f"{m['loose']:>7.2f}% {m['avg_latency_ms']:>9}")
