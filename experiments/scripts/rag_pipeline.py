"""
RAG Pipeline for Topic 6: On-Device SLM Inference Efficiency.

Components:
1. Document chunking + embedding
2. FAISS/HNSW retrieval
3. LLM generation via llama-cpp-python
4. Latency profiling at each stage
"""
import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from utils.config import (
    MODELS_DIR, EMBEDDING_MODEL, EMBEDDING_DIM,
    RAG_CONFIG, MOBILE_CONSTRAINTS, TOPIC6_DIR,
)


# ============================================================
# Performance Profiler
# ============================================================
@dataclass
class LatencyProfile:
    """Captures latency at each pipeline stage."""
    embedding_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    prefill_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    total_time_ms: float = 0.0
    tokens_generated: int = 0
    context_tokens: int = 0
    cache_hit: bool = False

    @property
    def ttft_ms(self) -> float:
        """Time to first token = embedding + retrieval + prefill."""
        return self.embedding_time_ms + self.retrieval_time_ms + self.prefill_time_ms

    @property
    def tokens_per_sec(self) -> float:
        if self.decode_time_ms <= 0:
            return 0.0
        return self.tokens_generated / (self.decode_time_ms / 1000.0)

    def to_dict(self) -> dict:
        return {
            "embedding_time_ms": round(self.embedding_time_ms, 2),
            "retrieval_time_ms": round(self.retrieval_time_ms, 2),
            "prefill_time_ms": round(self.prefill_time_ms, 2),
            "decode_time_ms": round(self.decode_time_ms, 2),
            "total_time_ms": round(self.total_time_ms, 2),
            "ttft_ms": round(self.ttft_ms, 2),
            "tokens_generated": self.tokens_generated,
            "tokens_per_sec": round(self.tokens_per_sec, 2),
            "context_tokens": self.context_tokens,
            "cache_hit": self.cache_hit,
        }


# ============================================================
# Retrieval Index
# ============================================================
class ChunkIndex:
    """FAISS-based chunk retrieval index."""

    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL):
        print(f"[Index] Loading embedding model: {embedding_model_name}")
        cache_dir = MODELS_DIR / "embeddings"
        self.encoder = SentenceTransformer(
            embedding_model_name,
            cache_folder=str(cache_dir) if cache_dir.exists() else None,
        )
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: list[str] = []
        self.chunk_embeddings: Optional[np.ndarray] = None

    def build_from_chunks(self, chunks: list[str], batch_size: int = 256):
        """Build FAISS index from text chunks."""
        print(f"[Index] Encoding {len(chunks)} chunks...")
        self.chunks = chunks
        self.chunk_embeddings = self.encoder.encode(
            chunks,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        # Inner product on normalized vectors = cosine similarity
        self.index = faiss.IndexFlatIP(self.chunk_embeddings.shape[1])
        self.index.add(self.chunk_embeddings.astype(np.float32))
        print(f"[Index] Built index with {self.index.ntotal} vectors, dim={self.chunk_embeddings.shape[1]}")

    def search(self, query: str, top_k: int = 5) -> tuple[list[str], list[float], float]:
        """Search for top-k chunks. Returns (chunks, scores, time_ms)."""
        t0 = time.perf_counter()
        q_emb = self.encoder.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(q_emb.astype(np.float32), top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        results = []
        result_scores = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
                result_scores.append(float(scores[0][i]))

        return results, result_scores, elapsed_ms

    def save(self, path: Path):
        """Save index and chunks to disk."""
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        np.save(str(path / "embeddings.npy"), self.chunk_embeddings)
        with open(path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)
        print(f"[Index] Saved to {path}")

    def load(self, path: Path):
        """Load index and chunks from disk."""
        self.index = faiss.read_index(str(path / "index.faiss"))
        self.chunk_embeddings = np.load(str(path / "embeddings.npy"))
        with open(path / "chunks.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        print(f"[Index] Loaded {len(self.chunks)} chunks from {path}")


# ============================================================
# LLM Generator (llama.cpp)
# ============================================================
LLAMA_CLI = str(Path.home() / "Desktop" / "llama_b8656" / "llama-cli.exe")
LLAMA_CLI_ENV = {"PATH": str(Path(LLAMA_CLI).parent) + ";" + __import__("os").environ.get("PATH", "")}


class LLMGeneratorCLI:
    """Subprocess wrapper around llama.cpp CLI — for models unsupported by llama-cpp-python (e.g. Gemma-4)."""

    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = -1, **kwargs):
        import os
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers if n_gpu_layers >= 0 else 99
        self.env = {**os.environ, **LLAMA_CLI_ENV}
        # Warm up: verify model loads (quick 1-token run)
        print(f"[LLM-CLI] Loading model: {model_path}")
        t0 = time.perf_counter()
        result = self._run("Hi", max_tokens=1)
        print(f"[LLM-CLI] Model ready in {(time.perf_counter()-t0)*1000:.0f}ms")

    def _run(self, prompt: str, max_tokens: int = 128, temperature: float = 0.0) -> str:
        import subprocess, re
        cmd = [
            LLAMA_CLI,
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(max_tokens),
            "--no-display-prompt",
            "-ngl", str(self.n_gpu_layers),
            "--temp", str(temperature),
            "-c", str(self.n_ctx),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, env=self.env, timeout=120)
        raw = result.stdout.strip()
        # Strip thinking block if present
        raw = re.sub(r'\[Start thinking\].*?\[End thinking\]', '', raw, flags=re.DOTALL).strip()
        # Strip prompt echo if any
        if prompt.strip() in raw:
            raw = raw.replace(prompt.strip(), "").strip()
        # Take first non-empty line
        for line in raw.split("\n"):
            line = line.strip()
            if line and not line.startswith(">"):
                return line
        return raw

    def generate(self, prompt: str, max_tokens: int = 128,
                 temperature: float = 0.0) -> tuple[str, float, float, int]:
        t_start = time.perf_counter()
        text = self._run(prompt, max_tokens=max_tokens, temperature=temperature)
        total_ms = (time.perf_counter() - t_start) * 1000
        n_tokens = len(text.split())
        decode_ms = total_ms * 0.8
        prefill_ms = total_ms * 0.2
        return text, prefill_ms, decode_ms, n_tokens


class LLMGenerator:
    """Wrapper around llama-cpp-python for inference with profiling.
    Automatically falls back to LLMGeneratorCLI for unsupported architectures."""

    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 4,
                 n_gpu_layers: int = -1, verbose: bool = False):
        from llama_cpp import Llama

        print(f"[LLM] Loading model: {model_path}")
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
            )
            self._cli = False
        except ValueError:
            print(f"[LLM] llama-cpp-python unsupported, falling back to CLI backend")
            self._delegate = LLMGeneratorCLI(model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
            self._cli = True
        self.model_path = model_path
        if not self._cli:
            print(f"[LLM] Model loaded. n_ctx={n_ctx}")

    def generate(self, prompt: str, max_tokens: int = 128,
                 temperature: float = 0.0) -> tuple[str, float, float, int]:
        """
        Generate response with timing.
        Returns: (text, prefill_ms, decode_ms, n_tokens)
        """
        if self._cli:
            return self._delegate.generate(prompt, max_tokens=max_tokens, temperature=temperature)

        # Prefill timing: measure time to first token
        t_start = time.perf_counter()

        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            echo=False,
            stop=["\n", "\n\n", "Question:", "Answer:", "\nQ:", "<|im_end|>"],
            repeat_penalty=1.1,
        )

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000

        raw = output["choices"][0]["text"].strip()
        # Take only first line / sentence to avoid runaway generation
        text = raw.split("\n")[0].strip()
        usage = output.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Approximate: prefill proportional to prompt tokens
        if completion_tokens > 0:
            decode_ms = total_ms * (completion_tokens / (prompt_tokens + completion_tokens))
            prefill_ms = total_ms - decode_ms
        else:
            prefill_ms = total_ms
            decode_ms = 0

        return text, prefill_ms, decode_ms, completion_tokens


# ============================================================
# RAG Pipeline
# ============================================================
class RAGPipeline:
    """Full RAG pipeline with profiling."""

    def __init__(self, index: ChunkIndex, generator: LLMGenerator,
                 top_k: int = RAG_CONFIG["top_k"]):
        self.index = index
        self.generator = generator
        self.top_k = top_k

    def build_prompt(self, question: str, contexts: list[str],
                     mode: str = "rag") -> str:
        """Build RAG prompt from question and retrieved contexts.

        mode:
          "rag" - standard RAG prompt
          "con" - Chain-of-Note: model writes per-document notes before answering
          "compress" - uses pre-compressed contexts (same prompt as rag)
        """
        if mode == "con":
            context_str = "\n\n".join(
                f"[Document {i+1}]: {ctx}" for i, ctx in enumerate(contexts)
            )
            prompt = (
                f"For each document, write one note about what it says relevant to the question.\n"
                f"Then write 'Final answer:' followed by a short direct answer.\n\n"
                f"{context_str}\n\n"
                f"Question: {question}\n\n"
                f"Doc 1 note:"
            )
        else:
            context_str = "\n\n".join(
                f"[Document {i+1}]: {ctx}" for i, ctx in enumerate(contexts)
            )
            prompt = (
                f"Answer the following question based on the provided documents. "
                f"Give a short, direct answer.\n\n"
                f"{context_str}\n\n"
                f"Question: {question}\n"
                f"Answer:"
            )
        return prompt

    def _compress_passages(self, question: str, passages: list[str]) -> tuple[list[str], float]:
        """Query-time compression: SLM summarizes each passage before answering.

        This is the query-time equivalent of CTKS compile-time compression.
        Same compression quality, but cost is paid per-query instead of amortized.

        Returns: (compressed_passages, compression_time_ms)
        """
        COMPRESS_PROMPT = (
            "Summarize this passage in one concise sentence, "
            "keeping only facts relevant to the question.\n\n"
            "Question: {question}\n"
            "Passage: {passage}\n"
            "Summary:"
        )

        compressed = []
        total_ms = 0.0
        for passage in passages:
            prompt = COMPRESS_PROMPT.format(question=question, passage=passage[:500])
            t0 = time.perf_counter()
            output = self.generator.llm(
                prompt,
                max_tokens=80,
                temperature=0.0,
                echo=False,
                stop=["\n", "\n\n", "Question:", "<|im_end|>"],
                repeat_penalty=1.1,
            )
            total_ms += (time.perf_counter() - t0) * 1000
            summary = output["choices"][0]["text"].strip()
            if summary:
                compressed.append(summary)
            else:
                compressed.append(passage[:100])  # fallback: truncate

        return compressed, total_ms

    def query(self, question: str, top_k: Optional[int] = None,
              mode: str = "rag") -> tuple[str, LatencyProfile]:
        """Run full RAG pipeline with profiling.

        mode: "rag" (standard), "con" (Chain-of-Note), or "compress" (query-time compression)
        """
        k = top_k or self.top_k
        profile = LatencyProfile()

        total_start = time.perf_counter()

        # 1. Retrieve
        chunks, scores, search_ms = self.index.search(question, top_k=k)
        profile.embedding_time_ms = search_ms * 0.7
        profile.retrieval_time_ms = search_ms * 0.3

        # 1.5. Query-time compression (if compress mode)
        compress_ms = 0.0
        if mode == "compress":
            chunks, compress_ms = self._compress_passages(question, chunks)

        # 2. Build prompt
        answer_mode = "rag" if mode == "compress" else mode
        prompt = self.build_prompt(question, chunks, mode=answer_mode)

        # 3. Generate
        if mode == "con":
            # Chain-of-Note: multi-line output needed, bypass single-line stop tokens
            t0 = time.perf_counter()
            output = self.generator.llm(
                prompt,
                max_tokens=300,
                temperature=0.0,
                echo=False,
                stop=["Question:", "\n\nQuestion", "<|im_end|>"],
                repeat_penalty=1.1,
            )
            total_ms = (time.perf_counter() - t0) * 1000
            raw = output["choices"][0]["text"]
            answer = self._extract_con_answer(raw)
            usage = output.get("usage", {})
            profile.prefill_time_ms = total_ms * 0.3
            profile.decode_time_ms = total_ms * 0.7
            profile.tokens_generated = usage.get("completion_tokens", 0)
        else:
            answer, prefill_ms, decode_ms, n_tokens = self.generator.generate(prompt)
            profile.prefill_time_ms = prefill_ms
            profile.decode_time_ms = decode_ms
            profile.tokens_generated = n_tokens

        profile.total_time_ms = (time.perf_counter() - total_start) * 1000

        # Track compression overhead separately (for compress mode analysis)
        if compress_ms > 0:
            profile.prefill_time_ms += compress_ms  # compression is part of "prefill" cost

        return answer, profile

    def _extract_con_answer(self, text: str) -> str:
        """Extract final answer from Chain-of-Note output."""
        import re
        # Look for "Final answer:" or "Answer:" after notes
        for marker in ["Final answer:", "Final Answer:", "Answer:"]:
            if marker in text:
                ans = text.split(marker)[-1].strip()
                return ans.split("\n")[0].strip()
        # Fallback: last non-empty line
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            # Skip lines that look like notes (start with "- Doc")
            for line in reversed(lines):
                if not line.startswith("- Doc") and not line.startswith("Doc"):
                    return line
        return text.split("\n")[0].strip()

    def query_vanilla(self, question: str) -> tuple[str, LatencyProfile]:
        """Run without RAG (vanilla inference) for baseline comparison."""
        profile = LatencyProfile()
        total_start = time.perf_counter()

        prompt = (
            f"Answer the following question. Give a short, direct answer.\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )

        answer, prefill_ms, decode_ms, n_tokens = self.generator.generate(prompt)
        profile.prefill_time_ms = prefill_ms
        profile.decode_time_ms = decode_ms
        profile.tokens_generated = n_tokens
        profile.total_time_ms = (time.perf_counter() - total_start) * 1000

        return answer, profile


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    print("RAG Pipeline module loaded successfully.")
    print(f"Models dir: {MODELS_DIR}")
    print(f"Topic 6 dir: {TOPIC6_DIR}")

    # Test embedding model only (no LLM needed)
    idx = ChunkIndex()
    test_chunks = [
        "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
        "Python is a high-level programming language.",
        "The human heart beats approximately 100,000 times per day.",
    ]
    idx.build_from_chunks(test_chunks)
    results, scores, ms = idx.search("What is the Eiffel Tower?", top_k=2)
    print(f"\nSearch test ({ms:.1f}ms):")
    for r, s in zip(results, scores):
        print(f"  [{s:.3f}] {r[:60]}...")
