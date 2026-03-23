#!/usr/bin/env python3
"""
Generate benchmark figures from JSON logs produced by the other benchmark scripts.

Reads from benchmarks/logs/ and writes PNG figures to benchmarks/figures/.

Plots generated:
  1. Train Loss vs Epoch
  2. Test Accuracy vs Epoch
  3. Train Loss vs Wall-clock Time
  4. Per-step optimizer time comparison (bar chart)
"""

import glob
import json
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib is required. Install it with: pip install matplotlib")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
COLORS = {
    "FusedMuon": "#1f77b4",
    "VanillaMuon": "#ff7f0e",
    "AdamW": "#2ca02c",
}
MARKERS = {
    "FusedMuon": "o",
    "VanillaMuon": "s",
    "AdamW": "^",
}


def load_cifar_logs():
    """Load all *_cifar10.json logs. Returns dict[display_name -> log]."""
    logs = {}
    pattern = os.path.join(LOG_DIR, "*_cifar10.json")
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            data = json.load(f)
        name = data.get("optimizer", os.path.basename(path))
        logs[name] = data
    return logs


def _style(ax):
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)


# ---------------------------------------------------------------------------
# Plot 1: Train Loss vs Epoch
# ---------------------------------------------------------------------------
def plot_train_loss(logs):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, log in logs.items():
        epochs = [e["epoch"] for e in log["epochs"]]
        losses = [e["train_loss"] for e in log["epochs"]]
        ax.plot(epochs, losses, label=name,
                color=COLORS.get(name, None),
                marker=MARKERS.get(name, "."),
                markersize=4, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_title("Train Loss vs Epoch (CIFAR-10)")
    _style(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 2: Test Accuracy vs Epoch
# ---------------------------------------------------------------------------
def plot_test_accuracy(logs):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, log in logs.items():
        epochs = [e["epoch"] for e in log["epochs"]]
        accs = [e["test_accuracy"] * 100.0 for e in log["epochs"]]
        ax.plot(epochs, accs, label=name,
                color=COLORS.get(name, None),
                marker=MARKERS.get(name, "."),
                markersize=4, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Test Accuracy vs Epoch (CIFAR-10)")
    _style(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 3: Train Loss vs Wall-clock Time
# ---------------------------------------------------------------------------
def plot_loss_vs_time(logs):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, log in logs.items():
        cum_time = []
        t = 0.0
        for e in log["epochs"]:
            t += e["epoch_time_s"]
            cum_time.append(t)
        losses = [e["train_loss"] for e in log["epochs"]]
        ax.plot(cum_time, losses, label=name,
                color=COLORS.get(name, None),
                marker=MARKERS.get(name, "."),
                markersize=4, linewidth=1.5)
    ax.set_xlabel("Wall-clock Time (s)")
    ax.set_ylabel("Train Loss")
    ax.set_title("Train Loss vs Wall-clock Time (CIFAR-10)")
    _style(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 4: Per-step optimizer time (bar chart)
# ---------------------------------------------------------------------------
def plot_optimizer_time(logs):
    fig, ax = plt.subplots(figsize=(8, 5))
    names = []
    avg_times = []
    colors = []
    for name, log in logs.items():
        epochs = log["epochs"]
        avg_opt = sum(e["optimizer_step_time_ms"] for e in epochs) / len(epochs)
        names.append(name)
        avg_times.append(avg_opt)
        colors.append(COLORS.get(name, "#888888"))

    bars = ax.bar(names, avg_times, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, avg_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Avg Optimizer Step Time per Epoch (ms)")
    ax.set_title("Optimizer Step Time Comparison (CIFAR-10)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    logs = load_cifar_logs()
    if not logs:
        print(f"No CIFAR-10 log files found in {LOG_DIR}.")
        print("Run benchmarks/train_cifar10.py first.")
        sys.exit(1)

    print(f"Found logs for: {', '.join(logs.keys())}")

    plots = [
        ("train_loss_vs_epoch.png", plot_train_loss),
        ("test_accuracy_vs_epoch.png", plot_test_accuracy),
        ("train_loss_vs_time.png", plot_loss_vs_time),
        ("optimizer_step_time.png", plot_optimizer_time),
    ]

    for filename, plot_fn in plots:
        fig = plot_fn(logs)
        path = os.path.join(FIG_DIR, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
