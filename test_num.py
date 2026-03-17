import csv
import os
import re
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
HEAD_AWARE_PATH = ROOT / "scale_cache" / "head_aware" / "qwen3-vl_8b_w16a5_tmp3.pt"
MBQ_PATH = ROOT / "scale_cache" / "mbq" / "qwen3_vl_8b_w16a5.pt"
OUTPUT_DIR = ROOT / "compare_scale"
LAYER_DIR = OUTPUT_DIR / "layers_tmp3"
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
OVERVIEW_PNG = OUTPUT_DIR / "overview.png"

HEAD_AWARE_COLOR = "#0b7285"
MBQ_COLOR = "#e67700"
GAP_COLOR = "#ced4da"
GROUP_COLORS = {
    "attn_qkv": "#1c7ed6",
    "mlp_in": "#2f9e44",
    "mlp_out": "#d9480f",
}
GROUP_LABELS = {
    "attn_qkv": "Attention QKV",
    "mlp_in": "MLP Gate/Up",
    "mlp_out": "MLP Down",
}
GROUP_ORDER = ["attn_qkv", "mlp_in", "mlp_out"]


def configure_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#f8f9fa",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#dee2e6",
            "axes.labelcolor": "#343a40",
            "axes.titleweight": "bold",
            "grid.color": "#e9ecef",
            "grid.linewidth": 0.8,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.frameon": True,
            "legend.facecolor": "#ffffff",
            "legend.edgecolor": "#dee2e6",
            "savefig.facecolor": "#f8f9fa",
        }
    )


def load_scale_list(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if isinstance(payload, dict) and "scale" in payload:
        scale_list = payload["scale"]
    elif isinstance(payload, list):
        scale_list = payload
    else:
        raise TypeError(f"Unsupported scale payload type in {path}: {type(payload)}")

    normalized = []
    for prev_op, layer_names, scale in scale_list:
        normalized.append((prev_op, tuple(layer_names), torch.as_tensor(scale, dtype=torch.float32).flatten()))
    return normalized


def parse_layer_idx(prev_op):
    match = re.search(r"layers\.(\d+)\.", prev_op)
    if match is None:
        raise ValueError(f"Cannot parse layer index from {prev_op}")
    return int(match.group(1))


def classify_group(prev_op, layer_names):
    if prev_op.endswith("input_layernorm"):
        return "attn_qkv"
    if prev_op.endswith("post_attention_layernorm"):
        return "mlp_in"
    if prev_op.endswith("mlp.up_proj") and len(layer_names) == 1 and layer_names[0].endswith("mlp.down_proj"):
        return "mlp_out"
    return "unknown"


def compare_metrics(head_aware_scale, mbq_scale):
    diff = (head_aware_scale - mbq_scale).abs()
    rel = diff / mbq_scale.abs().clamp_min(1e-8)
    cosine = F.cosine_similarity(
        head_aware_scale.unsqueeze(0),
        mbq_scale.unsqueeze(0),
        dim=1,
    ).item()
    return {
        "channels": head_aware_scale.numel(),
        "mean_head_aware": head_aware_scale.mean().item(),
        "mean_mbq": mbq_scale.mean().item(),
        "std_head_aware": head_aware_scale.std().item(),
        "std_mbq": mbq_scale.std().item(),
        "mean_abs_diff": diff.mean().item(),
        "max_abs_diff": diff.max().item(),
        "mean_rel_diff": rel.mean().item(),
        "max_rel_diff": rel.max().item(),
        "cosine_similarity": cosine,
    }


def ensure_same_structure(head_aware_scales, mbq_scales):
    if len(head_aware_scales) != len(mbq_scales):
        raise ValueError(
            f"Scale list length mismatch: head-aware={len(head_aware_scales)} mbq={len(mbq_scales)}"
        )

    for index, (ha_item, mbq_item) in enumerate(zip(head_aware_scales, mbq_scales)):
        ha_key = (ha_item[0], ha_item[1])
        mbq_key = (mbq_item[0], mbq_item[1])
        if ha_key != mbq_key:
            raise ValueError(f"Scale item mismatch at index {index}: {ha_key} != {mbq_key}")


def build_records(head_aware_scales, mbq_scales):
    records_by_layer = defaultdict(dict)
    summary_rows = []

    for prev_op, layer_names, head_aware_scale in head_aware_scales:
        layer_idx = parse_layer_idx(prev_op)
        group_name = classify_group(prev_op, layer_names)
        records_by_layer[layer_idx][group_name] = {
            "prev_op": prev_op,
            "layer_names": layer_names,
            "head_aware": head_aware_scale,
        }

    for prev_op, layer_names, mbq_scale in mbq_scales:
        layer_idx = parse_layer_idx(prev_op)
        group_name = classify_group(prev_op, layer_names)
        records_by_layer[layer_idx][group_name]["mbq"] = mbq_scale

    for layer_idx in sorted(records_by_layer):
        for group_name in GROUP_ORDER:
            if group_name not in records_by_layer[layer_idx]:
                continue
            item = records_by_layer[layer_idx][group_name]
            metrics = compare_metrics(item["head_aware"], item["mbq"])
            item["metrics"] = metrics
            summary_rows.append(
                {
                    "layer_idx": layer_idx,
                    "group_name": group_name,
                    "group_label": GROUP_LABELS[group_name],
                    "prev_op": item["prev_op"],
                    "target_layers": ",".join(item["layer_names"]),
                    **metrics,
                }
            )

    return records_by_layer, summary_rows


def write_summary_csv(summary_rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "layer_idx",
                "group_name",
                "group_label",
                "prev_op",
                "target_layers",
                "channels",
                "mean_head_aware",
                "mean_mbq",
                "std_head_aware",
                "std_mbq",
                "mean_abs_diff",
                "max_abs_diff",
                "mean_rel_diff",
                "max_rel_diff",
                "cosine_similarity",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def plot_overview(summary_rows):
    layer_indices = sorted({row["layer_idx"] for row in summary_rows})
    grouped = {group_name: {} for group_name in GROUP_ORDER}
    for row in summary_rows:
        grouped[row["group_name"]][row["layer_idx"]] = row

    fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=True)
    metric_names = ["mean_abs_diff", "mean_rel_diff", "cosine_similarity"]
    metric_titles = ["Mean Absolute Difference", "Mean Relative Difference", "Cosine Similarity"]

    for axis, metric_name, metric_title in zip(axes, metric_names, metric_titles):
        for group_name in GROUP_ORDER:
            values = [grouped[group_name][layer_idx][metric_name] for layer_idx in layer_indices]
            axis.plot(
                layer_indices,
                values,
                color=GROUP_COLORS[group_name],
                linewidth=2.2,
                marker="o",
                markersize=4,
                label=GROUP_LABELS[group_name],
            )
        axis.set_title(metric_title)
        axis.set_ylabel(metric_title)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.legend(loc="best")

    axes[-1].set_xlabel("Layer Index")
    fig.suptitle("Head-Aware vs MBQ Scale Differences", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OVERVIEW_PNG, dpi=220)
    plt.close(fig)


def add_metric_box(axis, metrics):
    text = "\n".join(
        [
            f"MAE: {metrics['mean_abs_diff']:.4f}",
            f"Max |diff|: {metrics['max_abs_diff']:.4f}",
            f"Mean rel diff: {metrics['mean_rel_diff']:.4f}",
            f"Cosine: {metrics['cosine_similarity']:.4f}",
        ]
    )
    axis.text(
        0.995,
        0.96,
        text,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#ffffff", "edgecolor": "#dee2e6", "alpha": 0.95},
    )


def plot_layer_figure(layer_idx, layer_records):
    fig, axes = plt.subplots(len(GROUP_ORDER), 1, figsize=(17, 13))
    if len(GROUP_ORDER) == 1:
        axes = [axes]

    for axis, group_name in zip(axes, GROUP_ORDER):
        if group_name not in layer_records:
            axis.set_axis_off()
            continue

        record = layer_records[group_name]
        head_aware = record["head_aware"]
        mbq = record["mbq"]
        diff = (head_aware - mbq).abs()
        channels = torch.arange(head_aware.numel())

        axis.plot(
            channels.numpy(),
            head_aware.numpy(),
            color=HEAD_AWARE_COLOR,
            linewidth=1.15,
            alpha=0.95,
            label="Head-Aware",
        )
        axis.plot(
            channels.numpy(),
            mbq.numpy(),
            color=MBQ_COLOR,
            linewidth=1.15,
            alpha=0.85,
            label="MBQ",
        )
        axis.fill_between(
            channels.numpy(),
            head_aware.numpy(),
            mbq.numpy(),
            color=GAP_COLOR,
            alpha=0.22,
            linewidth=0,
        )

        axis.set_title(f"{GROUP_LABELS[group_name]} | channels={head_aware.numel()}")
        axis.set_ylabel("Scale")
        axis.set_xlim(0, head_aware.numel() - 1)
        axis.legend(loc="upper left")
        add_metric_box(axis, record["metrics"])

        diff_axis = axis.twinx()
        diff_axis.plot(
            channels.numpy(),
            diff.numpy(),
            color="#c92a2a",
            linewidth=0.9,
            alpha=0.35,
            label="|diff|",
        )
        diff_axis.set_ylabel("|diff|", color="#c92a2a")
        diff_axis.tick_params(axis="y", colors="#c92a2a")
        diff_axis.grid(False)

    axes[-1].set_xlabel("Channel Index")
    fig.suptitle(f"Layer {layer_idx}: Head-Aware vs MBQ Scale Curves", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(LAYER_DIR / f"layer_{layer_idx:02d}.png", dpi=220)
    plt.close(fig)


def plot_all_layers(records_by_layer):
    LAYER_DIR.mkdir(parents=True, exist_ok=True)
    for layer_idx in sorted(records_by_layer):
        plot_layer_figure(layer_idx, records_by_layer[layer_idx])


def main():
    configure_plot_style()
    head_aware_scales = load_scale_list(HEAD_AWARE_PATH)
    mbq_scales = load_scale_list(MBQ_PATH)
    ensure_same_structure(head_aware_scales, mbq_scales)

    records_by_layer, summary_rows = build_records(head_aware_scales, mbq_scales)
    write_summary_csv(summary_rows)
    plot_overview(summary_rows)
    plot_all_layers(records_by_layer)

    print(f"Saved summary csv to: {SUMMARY_CSV}")
    print(f"Saved overview figure to: {OVERVIEW_PNG}")
    print(f"Saved per-layer figures to: {LAYER_DIR}")


if __name__ == "__main__":
    main()
