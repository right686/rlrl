import csv
import shutil

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


task = "zsin_0"
root_path = Path(__file__).resolve().parent
project_root = root_path.parent
record_root = project_root / "output" / task / "record"
figure_root = project_root / "figure" / task / "train"


def numeric_children(path, prefix=None):
    children = []
    if not path.exists():
        return []
    for child in path.iterdir():
        if not child.is_dir():
            continue
        if prefix is not None:
            if not child.name.startswith(prefix):
                continue
            suffix = child.name[len(prefix):]
        else:
            suffix = child.name
        if suffix.isdigit():
            children.append((int(suffix), child))
    return [child for _, child in sorted(children)]


def load_training_data(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
    if data.size == 0:
        return None
    return np.atleast_1d(data)


def field(data, name):
    return np.atleast_1d(data[name]).astype(float)


def compute_energy(npz_path):
    if npz_path is None or not npz_path.exists():
        return None
    with np.load(npz_path) as data:
        if "actuator_force" not in data or "actuator_vel" not in data:
            return None
        force = np.asarray(data["actuator_force"], dtype=np.float64)
        velocity = np.asarray(data["actuator_vel"], dtype=np.float64)
    return float(np.sum(np.abs(force * velocity)) * 0.001)


def archive_flat_figures():
    legacy_dir = figure_root / "_legacy_flat"
    flat_pngs = sorted(figure_root.glob("*.png"))
    if not flat_pngs:
        return
    legacy_dir.mkdir(parents=True, exist_ok=True)
    for path in flat_pngs:
        target = legacy_dir / path.name
        if target.exists():
            target.unlink()
        shutil.move(str(path), str(target))


def plot_generation(generation_dir):
    generation = int(generation_dir.name)
    sample_dirs = numeric_children(generation_dir, prefix="sample_")
    if not sample_dirs:
        return None

    sample_data = []
    summary = []
    for sample_dir in sample_dirs:
        sample = int(sample_dir.name.split("_")[1])
        data = load_training_data(sample_dir / "train_information.csv")
        if data is None:
            continue

        npz_candidates = sorted(sample_dir.glob("test_generation_*.npz"))
        energy = compute_energy(npz_candidates[0]) if npz_candidates else None
        sample_data.append((sample, data, energy))
        summary.append(
            (
                sample,
                float(field(data, "mean_reward")[-1]),
                float(field(data, "loss")[-1]),
                energy,
            )
        )

    if not sample_data:
        return None

    generation_root = figure_root / f"generation_{generation:02d}"
    generation_root.mkdir(parents=True, exist_ok=True)

    metrics = (
        ("reward", "Reward", 1.0),
        ("mean_reward", "Mean Reward", 1.8),
        ("loss", "Loss", 1.2),
    )
    for metric, label, linewidth in metrics:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for sample, data, _ in sample_data:
            ax.plot(
                field(data, "episode"),
                field(data, metric),
                linewidth=linewidth,
                label=f"sample {sample}",
            )
        ax.set_title(f"Generation {generation:02d} - {label}")
        ax.set_xlabel("Episode")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(generation_root / f"{metric}.png", dpi=180)
        plt.close(fig)

    valid_energy = [item for item in sample_data if item[2] is not None]
    if valid_energy:
        best_sample, best_data, best_energy = min(valid_energy, key=lambda item: item[2])
    else:
        best_sample, best_data, _ = max(sample_data, key=lambda item: field(item[1], "mean_reward")[-1])
        best_energy = None

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    best_metrics = (
        ("reward", "Reward"),
        ("mean_reward", "Mean Reward"),
        ("loss", "Loss"),
    )
    for ax, (metric, label) in zip(axes, best_metrics):
        ax.plot(field(best_data, "episode"), field(best_data, metric), linewidth=1.8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[0].set_title(
        f"Generation {generation:02d} best sample: {best_sample}"
        + (f", energy={best_energy:.6g}" if best_energy is not None else "")
    )
    axes[-1].set_xlabel("Episode")
    fig.tight_layout()
    fig.savefig(generation_root / "best_sample.png", dpi=180)
    plt.close(fig)

    return generation_root, summary, (best_sample, best_energy)


def plot_summary(generation_summaries):
    if not generation_summaries:
        return []

    summary_root = figure_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)

    generations = np.asarray([item[0] for item in generation_summaries], dtype=float)
    best_reward = np.asarray([max(sample[1] for sample in item[1]) for item in generation_summaries])
    mean_reward = np.asarray([np.mean([sample[1] for sample in item[1]]) for item in generation_summaries])
    best_loss = np.asarray([min(sample[2] for sample in item[1]) for item in generation_summaries])
    mean_loss = np.asarray([np.mean([sample[2] for sample in item[1]]) for item in generation_summaries])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(generations, best_reward, marker="o", linewidth=2, label="best sample")
    axes[0].plot(generations, mean_reward, marker="s", linestyle="--", linewidth=1.6, label="generation mean")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Final mean reward")
    axes[0].set_title("Final Training Reward by Generation")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(generations, best_loss, marker="o", linewidth=2, label="best sample")
    axes[1].plot(generations, mean_loss, marker="s", linestyle="--", linewidth=1.6, label="generation mean")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Final loss")
    axes[1].set_title("Final Training Loss by Generation")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    summary_path = summary_root / "generation_summary.png"
    fig.savefig(summary_path, dpi=180)
    plt.close(fig)

    best_sample_ids = []
    best_energies = []
    best_rows = []
    for generation, summary, best_info in generation_summaries:
        best_sample, best_energy = best_info
        best_sample_ids.append(best_sample)
        best_energies.append(best_energy)
        best_row = next(row for row in summary if row[0] == best_sample)
        best_rows.append((generation, best_sample, best_energy, best_row[1], best_row[2]))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(generations, best_energies, marker="o", linewidth=2)
    for generation, sample, energy in zip(generations, best_sample_ids, best_energies):
        ax.annotate(f"s{sample}", (generation, energy), textcoords="offset points", xytext=(0, 7), ha="center")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best energy")
    ax.set_title("Best Sample Energy by Generation")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    best_energy_path = summary_root / "best_sample_energy.png"
    fig.savefig(best_energy_path, dpi=180)
    plt.close(fig)

    best_csv_path = summary_root / "best_samples.csv"
    with best_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "sample", "energy", "final_mean_reward", "final_loss"])
        writer.writerows(best_rows)

    return [summary_path, best_energy_path, best_csv_path]


def main():
    figure_root.mkdir(parents=True, exist_ok=True)
    archive_flat_figures()

    generation_summaries = []
    for generation_dir in numeric_children(record_root):
        result = plot_generation(generation_dir)
        if result is None:
            continue
        generation_root, summary, best_info = result
        generation_summaries.append((int(generation_dir.name), summary, best_info))
        print(f"saved: {generation_root}")

    for output_path in plot_summary(generation_summaries):
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
