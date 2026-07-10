"""
EM-Latency Pareto frontier plot across all models / tasks / variants.
Reads results/longbench/FRONTIER.json (updated by analyze_longbench.py).

Saves: paper/figs/frontier.pdf and frontier.png
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from utils.config import TOPIC6_DIR


MODEL_COLOR = {
    "lfm2.5-1.2b-instruct": "C0",
    "qwen3-0.6b": "C1",
    "qwen3-1.7b": "C2",
}
MODEL_LABEL = {
    "lfm2.5-1.2b-instruct": "LFM-Inst 1.2B",
    "qwen3-0.6b": "Qwen3-0.6B",
    "qwen3-1.7b": "Qwen3-1.7B",
}
VARIANT_MARKER = {
    "raw_trunc": "o",
    "raw_topk":  "s",
    "anchors":   "^",
    "summary":   "D",
    "anchored":  "P",
    "hybrid2":   "X",
    "hybrid3":   "*",
    "pec_card":  "v",
    "pec_hydrate": "h",
    "pec_adaptive": "8",
    "pec_bridge": "d",
}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rd = TOPIC6_DIR / "experiments" / "results" / "longbench"
    with open(rd / "FRONTIER.json") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    tasks = ["hotpotqa", "2wikimqa", "musique"]

    for ax, task in zip(axes, tasks):
        for r in data:
            if r["task"] != task:
                continue
            m = r["model"]
            v = r["variant"]
            x = r["latency_ms"]
            y = r["em"] * 100
            ax.scatter(x, y,
                       color=MODEL_COLOR.get(m, "gray"),
                       marker=VARIANT_MARKER.get(v, "."),
                       s=110, edgecolors="black", linewidths=0.5, alpha=0.85)
            ax.annotate(v, (x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("Latency per query (ms, log scale)")
        ax.set_ylabel("Exact Match (%)")
        ax.set_title(f"LongBench — {task}")
        ax.grid(alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    model_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                              markersize=10, label=MODEL_LABEL[m])
                     for m, c in MODEL_COLOR.items()]
    variant_handles = [Line2D([0], [0], marker=m, color='gray',
                                linestyle='None', markersize=10, label=v)
                       for v, m in VARIANT_MARKER.items()]
    axes[0].legend(handles=model_handles, loc="upper left",
                   fontsize=8, framealpha=0.9, title="Model")
    axes[-1].legend(handles=variant_handles, loc="upper left",
                    fontsize=8, framealpha=0.9, title="Variant",
                    ncol=2)

    fig.tight_layout()
    out_dir = TOPIC6_DIR / "paper" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"frontier.{ext}", dpi=150, bbox_inches="tight")
    print(f"[Saved] {out_dir / 'frontier.pdf'}")
    print(f"[Saved] {out_dir / 'frontier.png'}")


if __name__ == "__main__":
    main()
