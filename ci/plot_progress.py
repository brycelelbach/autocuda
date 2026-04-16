#!/usr/bin/env python3
"""Plot 'best so far' optimization progress over time from experiment logs."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path, help="Path to the experiment log CSV")
    p.add_argument(
        "--peak",
        type=float,
        default=None,
        help="Peak theoretical value for computing utilization %% "
        "(e.g. 7927 for GiB/s). If omitted, Y-axis shows raw metric values.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: <csv_stem>-progress.png next to CSV)",
    )
    p.add_argument(
        "--max-labels",
        type=int,
        default=6,
        help="Maximum number of improvement labels to show (default: 6)",
    )
    p.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU architecture name for the title (e.g. 'GB200')",
    )
    p.add_argument(
        "--kernel",
        type=str,
        default=None,
        help="Kernel name for the title (auto-detected from CSV path if omitted)",
    )
    p.add_argument(
        "--cutoff-hours",
        type=float,
        default=None,
        help="Only show experiments within this many hours from the start",
    )
    p.add_argument(
        "--harness",
        type=str,
        default=None,
        help="Harness name to show in subtitle (e.g. 'autocuda.py')",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to show in subtitle (e.g. 'Opus 4.6')",
    )
    return p.parse_args()


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["metric_value"] = pd.to_numeric(df["metric_value"], errors="coerce")
    return df


def detect_unit(df: pd.DataFrame) -> str:
    units = df["unit"].dropna().unique()
    return str(units[0]) if len(units) > 0 else ""


def detect_kernel_name(csv_path: Path) -> str:
    return csv_path.parent.name.replace("_", " ").title()


def compute_best_so_far(df: pd.DataFrame, peak: float | None) -> pd.DataFrame:
    """Build a dataframe of accepted improvements with running-best values."""
    accepted = df[df["status"].isin(["baseline", "accepted"])].copy()
    accepted = accepted.dropna(subset=["metric_value"]).reset_index(drop=True)

    if peak is not None:
        accepted["y_val"] = accepted["metric_value"] / peak * 100.0
    else:
        accepted["y_val"] = accepted["metric_value"]

    accepted["best_y"] = accepted["y_val"].cummax()
    accepted["prev_best"] = accepted["best_y"].shift(1, fill_value=0.0)
    accepted["delta"] = accepted["best_y"] - accepted["prev_best"]
    return accepted


def pick_labels(accepted: pd.DataFrame, max_labels: int) -> pd.DataFrame:
    """Select the top improvements by delta for labeling (excluding baseline)."""
    non_baseline = accepted[accepted["status"] != "baseline"]
    improvements = non_baseline[non_baseline["delta"] > 0].copy()
    return improvements.nlargest(max_labels, "delta")


def shorten_description(desc: str, max_len: int = 55) -> str:
    desc = desc.strip()
    if ";" in desc:
        desc = desc.split(";")[0].strip()
    elif "," in desc:
        parts = desc.split(",")
        desc = parts[0].strip()
        if len(parts) > 1 and len(desc) + len(parts[1]) + 2 < max_len:
            desc = desc + ", " + parts[1].strip()
    if len(desc) > max_len:
        desc = desc[: max_len - 3] + "..."
    return desc


def format_value(val: float, is_pct: bool) -> str:
    if is_pct:
        return f"{val:.1f}%"
    if val >= 1000:
        return f"{val:,.0f}"
    return f"{val:.1f}"


def format_delta(delta: float, is_pct: bool) -> str:
    if is_pct:
        return f"+{delta:.1f}pp"
    if delta >= 100:
        return f"+{delta:,.0f}"
    return f"+{delta:.1f}"


def plot(
    df: pd.DataFrame,
    accepted: pd.DataFrame,
    labels: pd.DataFrame,
    peak: float | None,
    unit: str,
    kernel_name: str,
    output: Path,
    gpu: str | None = None,
    harness: str | None = None,
    model: str | None = None,
):
    is_pct = peak is not None

    BG = "#1e1e2e"
    FG = "#cdd6f4"
    ACCENT = "#a6e3a1"
    ACCENT_DARK = "#40a050"
    GRID = "#45475a"
    REJECTED = "#585b70"
    LABEL_BG = "#2a2a3e"
    LABEL_BORDER = "#74c790"
    LABEL_TEXT = "#a6e3a1"

    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    t_start = df["timestamp"].min()

    valid = df.dropna(subset=["metric_value"]).copy()
    valid["elapsed_min"] = (valid["timestamp"] - t_start).dt.total_seconds() / 60.0
    if is_pct:
        valid["y_val"] = valid["metric_value"] / peak * 100.0
    else:
        valid["y_val"] = valid["metric_value"]

    accepted["elapsed_min"] = (accepted["timestamp"] - t_start).dt.total_seconds() / 60.0

    rejected = valid[~valid["status"].isin(["baseline", "accepted"])]
    ax.scatter(
        rejected["elapsed_min"],
        rejected["y_val"],
        c=REJECTED,
        s=8,
        alpha=0.4,
        zorder=2,
        label="Rejected / failed",
    )

    ax.scatter(
        accepted["elapsed_min"],
        accepted["y_val"],
        c=ACCENT,
        s=45,
        zorder=4,
        edgecolors="#1e1e2e",
        linewidths=0.5,
        label="Accepted",
    )

    elapsed_max = (df["timestamp"].max() - t_start).total_seconds() / 60.0
    step_times = list(accepted["elapsed_min"])
    step_vals = list(accepted["best_y"])
    step_times.append(elapsed_max)
    step_vals.append(step_vals[-1])
    ax.step(
        step_times,
        step_vals,
        where="post",
        color=ACCENT,
        linewidth=2.5,
        alpha=0.85,
        zorder=3,
        label="Best so far",
    )

    best_y = accepted["best_y"].max()
    baseline_y = accepted.iloc[0]["y_val"]

    y_range = best_y - baseline_y if best_y > baseline_y else best_y
    margin = max(y_range * 0.1, 1.0)
    ymin = max(0, baseline_y - margin)
    ymax = best_y + margin * 2
    if is_pct:
        ymax = min(100, ymax)
    ax.set_ylim(ymin, ymax)

    labels_sorted = labels.sort_values("timestamp").reset_index(drop=True)
    labels_sorted["elapsed_min"] = (
        (labels_sorted["timestamp"] - t_start).dt.total_seconds() / 60.0
    )

    for i, (_, row) in enumerate(labels_sorted.iterrows()):
        desc = shorten_description(str(row["description"]))
        val = row["best_y"]
        delta = row["delta"]
        elapsed = row["elapsed_min"]

        label_text = (
            f"{format_delta(delta, is_pct)} \u2192 {format_value(val, is_pct)}: {desc}"
        )

        if i % 2 == 0:
            x_off, y_off = 10, -(12 + 14 * (i // 2))
            va = "top"
        else:
            x_off, y_off = 10, 12 + 14 * (i // 2)
            va = "bottom"

        ax.annotate(
            label_text,
            xy=(elapsed, val),
            xytext=(x_off, y_off),
            textcoords="offset points",
            fontsize=8,
            color=LABEL_TEXT,
            fontweight="bold",
            ha="left",
            va=va,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=LABEL_BG,
                edgecolor=LABEL_BORDER,
                alpha=0.95,
                linewidth=0.8,
            ),
            zorder=10,
        )

    total_experiments = len(df)
    n_accepted = len(accepted) - 1
    duration = df["timestamp"].max() - df["timestamp"].min()
    hours = duration.total_seconds() / 3600

    tag_parts = []
    if gpu:
        tag_parts.append(gpu)
    if harness:
        tag_parts.append(harness)
    if model:
        tag_parts.append(model)
    tag_str = f" [{', '.join(tag_parts)}]" if tag_parts else ""

    baseline_str = format_value(baseline_y, is_pct)
    best_str = format_value(best_y, is_pct)

    ax.set_xlabel("Elapsed Time", fontsize=12, color=FG)

    if is_pct:
        ax.set_ylabel(f"{unit} Utilization (%)", fontsize=12, color=FG)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    else:
        ax.set_ylabel(unit, fontsize=12, color=FG)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1000 else f"{v:.0f}")
        )

    ax.set_title(
        f"{kernel_name} Kernel Optimization{tag_str}: "
        f"{baseline_str} \u2192 {best_str} "
        f"({total_experiments} experiments, {n_accepted} accepted, {hours:.1f}h)",
        fontsize=14,
        fontweight="bold",
        color=FG,
    )
    legend = ax.legend(
        loc="upper left", fontsize=9, framealpha=0.9,
        facecolor=BG, edgecolor=GRID, labelcolor=FG,
    )
    ax.grid(True, alpha=0.3, linestyle="--", color=GRID)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)

    def fmt_elapsed(x, _):
        h, m = divmod(int(x), 60)
        return f"{h}:{m:02d}" if h else f"{m}m"

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_elapsed))
    ax.set_xlim(left=0)

    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved to {output}")


def main():
    args = parse_args()
    if not args.csv.exists():
        print(f"Error: {args.csv} not found", file=sys.stderr)
        sys.exit(1)

    output = args.output or args.csv.with_name(args.csv.stem + "-progress.png")

    df = load_data(args.csv)

    if args.cutoff_hours is not None:
        cutoff = df["timestamp"].min() + pd.Timedelta(hours=args.cutoff_hours)
        df = df[df["timestamp"] <= cutoff].copy()

    unit = detect_unit(df)
    kernel_name = args.kernel or detect_kernel_name(args.csv)

    accepted = compute_best_so_far(df, args.peak)
    labels = pick_labels(accepted, args.max_labels)
    plot(
        df, accepted, labels, args.peak, unit, kernel_name, output,
        gpu=args.gpu, harness=args.harness, model=args.model,
    )


if __name__ == "__main__":
    main()
