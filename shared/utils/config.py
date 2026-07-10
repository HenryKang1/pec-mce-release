"""
Shared configuration: paths, model registry, retrieval settings.

Layout assumed by the release:
  <repo>/shared/...                  this package (config, download scripts)
  <repo>/experiments/scripts/...     evaluation pipeline and analysis scripts
  <repo>/experiments/results/...     per-question outputs and aggregates
"""
from pathlib import Path

# ============================================================
# Paths
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[2]  # repository root
SHARED_DIR = ROOT_DIR / "shared"
MODELS_DIR = SHARED_DIR / "models"
EMBEDDINGS_DIR = SHARED_DIR / "embeddings"
DATASETS_DIR = SHARED_DIR / "datasets"

# The experiment scripts reference the project directory via TOPIC6_DIR;
# in this release it is the repository root itself.
TOPIC6_DIR = ROOT_DIR

# ============================================================
# GGUF reader models (downloaded by shared/scripts/download_models.py).
# The paper uses: lfm2.5-1.2b-instruct, qwen3-4b, gemma-4-e4b.
# Exact artifact SHA-256 hashes are recorded in MODELS.md.
# ============================================================
GGUF_MODELS = {
    "qwen3-0.6b": {
        "repo": "unsloth/Qwen3-0.6B-GGUF",
        "file": "Qwen3-0.6B-Q4_K_M.gguf",
    },
    "deepseek-r1-1.5b": {
        "repo": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
        "file": "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
    },
    "qwen3-1.7b": {
        "repo": "unsloth/Qwen3-1.7B-GGUF",
        "file": "Qwen3-1.7B-Q4_K_M.gguf",
    },
    "qwen3-4b": {
        "repo": "unsloth/Qwen3-4B-GGUF",
        "file": "Qwen3-4B-Q4_K_M.gguf",
    },
    "lfm2.5-1.2b-thinking": {
        "repo": "LiquidAI/LFM2.5-1.2B-Thinking-GGUF",
        "file": "LFM2.5-1.2B-Thinking-Q4_K_M.gguf",
    },
    "lfm2.5-1.2b-instruct": {
        "repo": "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
        "file": "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
    },
    "gemma-4-e4b": {
        "repo": "bartowski/google_gemma-4-E4B-it-GGUF",
        "file": "google_gemma-4-E4B-it-Q4_K_M.gguf",
    },
}

# ============================================================
# Embedding model (frozen retrieval encoder)
# ============================================================
EMBEDDING_MODEL = "thenlper/gte-small"  # 384-dim
EMBEDDING_DIM = 384

# ============================================================
# QA datasets (external-retrieval stress test)
# ============================================================
QA_DATASETS = {
    "hotpotqa": {
        "hf_name": "hotpot_qa",
        "subset": "distractor",
        "split": "validation",
    },
    "triviaqa": {
        "hf_name": "trivia_qa",
        "subset": "rc.nocontext",
        "split": "validation",
    },
    "popqa": {
        "hf_name": "akariasai/PopQA",
        "split": "test",
    },
}

# ============================================================
# RAG settings
# ============================================================
RAG_CONFIG = {
    "chunk_size": 100,       # tokens per chunk
    "chunk_overlap": 50,     # overlap tokens
    "top_k": 5,              # default retrieval depth
    "max_context_tokens": 512,
    "similarity_metric": "cosine",
}

# ============================================================
# Inference settings (edge-deployment constraints)
# ============================================================
MOBILE_CONSTRAINTS = {
    "max_ram_mb": 2048,
    "max_ram_mb_relaxed": 4096,
    "max_kv_cache_mb": 512,
    "n_ctx": 2048,           # context window
    "n_batch": 512,
    "n_threads": 4,
}
