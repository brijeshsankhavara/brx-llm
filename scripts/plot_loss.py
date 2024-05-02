"""
Plot train and eval loss from the log_history.json that brx.train writes.

  python scripts/plot_loss.py runs/brx-tiny/log_history.json assets/loss.png
"""

import json
import sys

import matplotlib.pyplot as plt


def main(log_path, out_path):
    with open(log_path) as f:
        history = json.load(f)

    train = [h for h in history if "loss" in h]
    evals = [h for h in history if "eval_loss" in h]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.plot([h["step"] for h in train], [h["loss"] for h in train], label="train", color="#1a5fb4", linewidth=1.2)
    if evals:
        ax.plot(
            [h["step"] for h in evals],
            [h["eval_loss"] for h in evals],
            label="validation",
            color="#e66100",
            marker="o",
            markersize=4,
            linewidth=1.2,
        )
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("BRX-tiny on TinyStories")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
