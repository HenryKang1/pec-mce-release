"""
Download GGUF models for Topic 6 inference experiments.
HF models for Topic 1 will be downloaded on-demand during training.
"""
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import GGUF_MODELS, MODELS_DIR, EMBEDDING_MODEL

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def download_gguf_models():
    """Download GGUF models for llama.cpp inference."""
    for name, info in GGUF_MODELS.items():
        out_path = MODELS_DIR / info["file"]
        if out_path.exists():
            print(f"[{name}] Already exists: {out_path}")
            continue

        print(f"[{name}] Downloading from {info['repo']}...")
        try:
            downloaded = hf_hub_download(
                repo_id=info["repo"],
                filename=info["file"],
                local_dir=str(MODELS_DIR),
                local_dir_use_symlinks=False,
            )
            print(f"[{name}] Saved to {downloaded}")
        except Exception as e:
            print(f"[{name}] ERROR: {e}")
            print(f"[{name}] You may need to manually download from: https://huggingface.co/{info['repo']}")


def download_embedding_model():
    """Pre-download the embedding model for sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    cache_dir = MODELS_DIR / "embeddings"
    cache_dir.mkdir(exist_ok=True)

    print(f"[Embedding] Downloading {EMBEDDING_MODEL}...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=str(cache_dir))
        # Quick test
        test_emb = model.encode(["test sentence"])
        print(f"[Embedding] OK! Dimension: {test_emb.shape[1]}")
    except Exception as e:
        print(f"[Embedding] ERROR: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Downloading models")
    print(f"Output directory: {MODELS_DIR}")
    print("=" * 60)

    # Embedding model first (small, fast)
    download_embedding_model()

    # GGUF models (large, takes time)
    download_gguf_models()

    print("\n" + "=" * 60)
    print("Model downloads complete!")
    print("=" * 60)
