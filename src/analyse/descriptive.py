from __future__ import annotations

"""Produce descriptive tables and figures for the acoustic and neural spaces."""

from itertools import combinations
from pathlib import Path
from typing import Any
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.stats import chi2, spearmanr
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity


def load_params(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def resolve_value(value: Any, root: dict[str, Any]) -> Any:
    """Resolve ${...} references inside params.yaml values."""
    if isinstance(value, str):
        pattern = re.compile(r"\$\{([^}]+)\}")

        def replace(match: re.Match[str]) -> str:
            current: Any = root
            for part in match.group(1).split("."):
                current = current[part]
            return str(resolve_value(current, root))

        return pattern.sub(replace, value)

    if isinstance(value, dict):
        return {key: resolve_value(item, root) for key, item in value.items()}

    if isinstance(value, list):
        return [resolve_value(item, root) for item in value]

    return value


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def output_exists(path: Path) -> bool:
    if path.exists():
        print(f"exists: {path}")
        return True
    return False


def save_csv(path: Path, frame: pd.DataFrame, written: list[tuple[Path, Any]]) -> None:
    if output_exists(path):
        return
    ensure_parent(path)
    frame.to_csv(path, index=False)
    print(f"wrote: {path}")
    written.append((path, frame.shape))


def save_figure(path: Path, fig: plt.Figure, written: list[tuple[Path, Any]]) -> None:
    if output_exists(path):
        plt.close(fig)
        return
    ensure_parent(path)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote: {path}")
    written.append((path, "figure"))


def add_group_labels(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    l1_map = {"fr": "L1", "ru": "L2"}
    gender_map = {"f": "F", "m": "M"}
    output["group"] = (
        output["l1_status"].map(l1_map).fillna(output["l1_status"].astype(str).str.upper())
        + "/"
        + output["gender"].map(gender_map).fillna(output["gender"].astype(str).str.upper())
    )
    return output


def load_token_table(params: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(Path(params["paths"]["phoneme_tokens_csv"]))
    frame["token_idx"] = np.arange(len(frame))
    return frame


def load_acoustic_frame(params: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(Path(params["paths"]["acoustic_norm_csv"]))
    tokens = load_token_table(params)
    if len(frame) != len(tokens):
        raise ValueError("features_acoustic_norm.csv and phoneme_tokens.csv are not row-aligned")
    for column in ["speaker_id", "sentence_id", "l1_status", "gender", "phoneme"]:
        if column not in frame.columns and column in tokens.columns:
            frame[column] = tokens[column]
    frame["token_idx"] = np.arange(len(frame))
    frame = add_group_labels(frame)
    vowels = set(config["canonical_vowels"])
    return frame[frame["phoneme"].isin(vowels) & ~frame["formant_qc_fail"]].copy()


def build_acoustic_descriptive(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["phoneme", "group"])
    summary = grouped.agg(
        F1_norm_mean=("F1_norm", "mean"),
        F1_norm_median=("F1_norm", "median"),
        F1_norm_std=("F1_norm", "std"),
        F2_norm_mean=("F2_norm", "mean"),
        F2_norm_median=("F2_norm", "median"),
        F2_norm_std=("F2_norm", "std"),
    )
    quantiles = grouped[["F1_norm", "F2_norm"]].quantile([0.25, 0.75]).unstack()
    summary["F1_norm_iqr"] = quantiles[("F1_norm", 0.75)] - quantiles[("F1_norm", 0.25)]
    summary["F2_norm_iqr"] = quantiles[("F2_norm", 0.75)] - quantiles[("F2_norm", 0.25)]
    summary["F1_norm_cv"] = summary["F1_norm_std"] / summary["F1_norm_mean"].replace(0, np.nan)
    summary["F2_norm_cv"] = summary["F2_norm_std"] / summary["F2_norm_mean"].replace(0, np.nan)
    return summary.reset_index()


def build_variance_decomposition(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for phoneme, sub in frame.groupby("phoneme"):
        print(f"variance decomposition: /{phoneme}/")
        values = sub["F1_norm"].to_numpy()
        grand_mean = float(values.mean())
        total_ss = float(((values - grand_mean) ** 2).sum())
        inter_ss = 0.0
        intra_ss = 0.0
        residual_ss = 0.0
        for _, speaker_sub in sub.groupby("speaker_id"):
            speaker_values = speaker_sub["F1_norm"].to_numpy()
            speaker_mean = float(speaker_values.mean())
            inter_ss += len(speaker_values) * (speaker_mean - grand_mean) ** 2
            for _, sentence_sub in speaker_sub.groupby("sentence_id"):
                sentence_values = sentence_sub["F1_norm"].to_numpy()
                sentence_mean = float(sentence_values.mean())
                intra_ss += len(sentence_values) * (sentence_mean - speaker_mean) ** 2
                residual_ss += float(((sentence_values - sentence_mean) ** 2).sum())
        rows.append(
            {
                "phoneme": phoneme,
                "n_tokens": len(sub),
                "n_speakers": sub["speaker_id"].nunique(),
                "total_ss": total_ss,
                "inter_speaker_ss": inter_ss,
                "intra_speaker_ss": intra_ss,
                "residual_ss": residual_ss,
                "inter_speaker_prop": inter_ss / total_ss if total_ss else np.nan,
                "intra_speaker_prop": intra_ss / total_ss if total_ss else np.nan,
                "residual_prop": residual_ss / total_ss if total_ss else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_confidence_ellipse(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: Any) -> None:
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    if not np.isfinite(cov).all():
        return
    vals, vecs = np.linalg.eigh(cov)
    if np.any(vals <= 0):
        return
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    scale = np.sqrt(chi2.ppf(0.95, df=2))
    width, height = 2 * scale * np.sqrt(vals)
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    ellipse = Ellipse(
        xy=(float(np.mean(x)), float(np.mean(y))),
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor=color,
        linewidth=1.0,
        alpha=0.7,
    )
    ax.add_patch(ellipse)


def plot_vowel_chart(frame: pd.DataFrame, path: Path, written: list[tuple[Path, Any]]) -> None:
    if output_exists(path):
        return
    phonemes = sorted(frame["phoneme"].unique())
    groups = ["L1/F", "L1/M", "L2/F", "L2/M"]
    colors = {phoneme: plt.cm.tab10(index % 10) for index, phoneme in enumerate(phonemes)}
    markers = {"L1/F": "o", "L1/M": "s", "L2/F": "^", "L2/M": "D"}
    centroids = frame.groupby(["phoneme", "group"])[["F1_norm", "F2_norm"]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 7))
    for (phoneme, group), sub in frame.groupby(["phoneme", "group"]):
        add_confidence_ellipse(ax, sub["F2_norm"].to_numpy(), sub["F1_norm"].to_numpy(), colors[phoneme])
        centroid = centroids[(centroids["phoneme"] == phoneme) & (centroids["group"] == group)]
        if centroid.empty:
            continue
        ax.scatter(
            centroid["F2_norm"],
            centroid["F1_norm"],
            color=colors[phoneme],
            marker=markers[group],
            s=55,
            edgecolor="black",
            linewidth=0.4,
        )
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("F2_norm")
    ax.set_ylabel("F1_norm")
    ax.set_title("French oral vowel centroids with 95% ellipses")

    phoneme_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[p], markeredgecolor="black", label=p)
        for p in phonemes
    ]
    group_handles = [
        Line2D([0], [0], marker=markers[g], color="black", linestyle="", label=g)
        for g in groups
    ]
    legend1 = ax.legend(handles=phoneme_handles, title="Phoneme", loc="upper right", bbox_to_anchor=(1.22, 1.0))
    ax.add_artist(legend1)
    ax.legend(handles=group_handles, title="Group", loc="lower right", bbox_to_anchor=(1.2, 0.0))
    save_figure(path, fig, written)


def plot_boxplots(frame: pd.DataFrame, path: Path, written: list[tuple[Path, Any]]) -> None:
    if output_exists(path):
        return
    phonemes = sorted(frame["phoneme"].unique())
    l1_groups = ["fr", "ru"]
    labels = {"fr": "L1", "ru": "L2"}
    colors = {"fr": "#4C78A8", "ru": "#F58518"}
    offsets = {"fr": -0.17, "ru": 0.17}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    for ax, column in zip(axes, ["F1_norm", "F2_norm"]):
        for l1 in l1_groups:
            data = [
                frame[(frame["phoneme"] == phoneme) & (frame["l1_status"] == l1)][column].to_numpy()
                for phoneme in phonemes
            ]
            positions = np.arange(len(phonemes)) + offsets[l1]
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=0.28,
                patch_artist=True,
                showfliers=False,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[l1])
                patch.set_alpha(0.65)
        ax.set_title(column)
        ax.set_xticks(np.arange(len(phonemes)))
        ax.set_xticklabels(phonemes)
        ax.set_ylabel(column)
    handles = [Line2D([0], [0], color=colors[key], linewidth=8, label=labels[key]) for key in l1_groups]
    axes[1].legend(handles=handles, title="L1 status", loc="upper right")
    fig.suptitle("Normalised formants by phoneme and L1 status")
    save_figure(path, fig, written)


def plot_violin(frame: pd.DataFrame, path: Path, written: list[tuple[Path, Any]]) -> None:
    if output_exists(path):
        return
    top_vowels = frame["phoneme"].value_counts().head(4).index.tolist()
    colors = {"fr": "#4C78A8", "ru": "#F58518"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    for ax, phoneme in zip(axes.flat, top_vowels):
        sub = frame[frame["phoneme"] == phoneme]
        speakers = sorted(sub["speaker_id"].unique())
        positions = np.arange(1, len(speakers) + 1)
        datasets = [sub[sub["speaker_id"] == speaker]["F1_norm"].to_numpy() for speaker in speakers]
        violins = ax.violinplot(datasets, positions=positions, showmeans=False, showmedians=True, widths=0.8)
        for body, speaker in zip(violins["bodies"], speakers):
            l1 = sub[sub["speaker_id"] == speaker]["l1_status"].iloc[0]
            body.set_facecolor(colors[l1])
            body.set_edgecolor("black")
            body.set_alpha(0.6)
        ax.set_title(f"/{phoneme}/")
        ax.set_xticks(positions)
        ax.set_xticklabels(speakers, rotation=90)
        ax.set_ylabel("F1_norm")
    handles = [Line2D([0], [0], color=colors[key], linewidth=8, label=("L1" if key == "fr" else "L2")) for key in colors]
    axes.flat[1].legend(handles=handles, title="L1 status", loc="upper right")
    fig.suptitle("Intra-speaker F1 variability for the four most frequent vowels")
    save_figure(path, fig, written)


def between_class_variance_ratio(points: np.ndarray, labels: pd.Series) -> float:
    total_var = float(points.var(axis=0, ddof=1).sum())
    class_means = np.vstack([points[labels == label].mean(axis=0) for label in sorted(labels.unique())])
    between_var = float(class_means.var(axis=0, ddof=1).sum()) if len(class_means) > 1 else 0.0
    return between_var / total_var if total_var else np.nan


def average_cosine_stats(vectors: np.ndarray, labels: pd.Series) -> tuple[float, float, float]:
    normed = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
    total_sum = normed.sum(axis=0)
    total_pairs = len(normed) * (len(normed) - 1) / 2
    total_dot_sum = (float(np.dot(total_sum, total_sum)) - len(normed)) / 2

    within_sum = 0.0
    within_pairs = 0.0
    for label in sorted(labels.unique()):
        sub = normed[labels == label]
        count = len(sub)
        if count < 2:
            continue
        sum_vec = sub.sum(axis=0)
        within_sum += (float(np.dot(sum_vec, sum_vec)) - count) / 2
        within_pairs += count * (count - 1) / 2

    between_sum = total_dot_sum - within_sum
    between_pairs = total_pairs - within_pairs
    within_mean = within_sum / within_pairs if within_pairs else np.nan
    between_mean = between_sum / between_pairs if between_pairs else np.nan
    ratio = within_mean / between_mean if np.isfinite(between_mean) and between_mean != 0 else np.nan
    return within_mean, between_mean, ratio


def plot_neural_scatter(points: np.ndarray, meta: pd.DataFrame, color_by: str, title: str, path: Path, written: list[tuple[Path, Any]]) -> None:
    if output_exists(path):
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = meta[color_by].astype(str)
    unique_labels = sorted(labels.unique())
    colors = {label: plt.cm.tab20(index % 20) for index, label in enumerate(unique_labels)}
    for label in unique_labels:
        idx = labels == label
        ax.scatter(points[idx, 0], points[idx, 1], s=8, alpha=0.6, color=colors[label], label=label)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=8, loc="best")
    save_figure(path, fig, written)


def load_embedding_set(embeddings_dir: Path, model: str, layer: int) -> dict[str, Any]:
    prefix = embeddings_dir / f"{model}_layer{layer}"
    return {
        "pca2": np.load(prefix.with_name(f"{prefix.name}_pca2.npy")),
        "umap2": np.load(prefix.with_name(f"{prefix.name}_umap2.npy")),
        "pca50": np.load(prefix.with_name(f"{prefix.name}_pca50.npy")),
        "meta": pd.read_csv(prefix.with_name(f"{prefix.name}_meta.csv")),
    }


def build_neural_tables_and_figures(params: dict[str, Any], config: dict[str, Any], figure_dir: Path, table_dir: Path, written: list[tuple[Path, Any]]) -> None:
    embeddings_dir = Path(params["paths"]["embeddings_dir"])
    layers = config["representative_layers"]
    methods = config["reduction_methods"]

    variance_rows: list[dict[str, Any]] = []
    cosine_rows: list[dict[str, Any]] = []
    for model, layer in layers.items():
        print(f"neural descriptive: loading {model} layer {layer}")
        data = load_embedding_set(embeddings_dir, model, int(layer))
        meta = data["meta"]
        for method in methods:
            print(f"neural descriptive: {model} layer {layer} {method}")
            points = data[method]
            method_label = {"pca2": "pca", "umap2": "umap"}[method]
            for color_by in ["phoneme", "l1_status", "gender"]:
                print(f"plotting: {method_label}_{model}_layer{layer}_{color_by}.png")
                fig_path = figure_dir / f"{method_label}_{model}_layer{layer}_{color_by}.png"
                plot_neural_scatter(
                    points=points,
                    meta=meta,
                    color_by=color_by,
                    title=f"{model} layer {layer} {method_label} by {color_by}",
                    path=fig_path,
                    written=written,
                )
            variance_rows.append(
                {
                    "model": model,
                    "layer": layer,
                    "reduction": method,
                    "between_class_variance_ratio": between_class_variance_ratio(points, meta["phoneme"]),
                }
            )

        print(f"cosine summary: {model} layer {layer} pca50")
        within_mean, between_mean, ratio = average_cosine_stats(data["pca50"], meta["phoneme"])
        cosine_rows.append(
            {
                "model": model,
                "layer": layer,
                "within_phoneme_cosine": within_mean,
                "between_phoneme_cosine": between_mean,
                "ratio": ratio,
            }
        )

    save_csv(table_dir / "between_class_variance.csv", pd.DataFrame(variance_rows), written)
    save_csv(table_dir / "cosine_similarity.csv", pd.DataFrame(cosine_rows), written)


def sample_rsm_indices(frame: pd.DataFrame, phonemes: list[str], per_phoneme: int, random_state: int) -> np.ndarray:
    sampled: list[np.ndarray] = []
    for phoneme in phonemes:
        idx = frame.index[frame["phoneme"] == phoneme].to_numpy()
        if len(idx) == 0:
            continue
        size = min(len(idx), per_phoneme)
        rng = np.random.default_rng(random_state + sum(ord(ch) for ch in phoneme))
        sampled.append(np.sort(rng.choice(idx, size=size, replace=False)))
    if not sampled:
        raise ValueError("No tokens available for RSM sampling")
    return np.concatenate(sampled)


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    tri = np.triu_indices_from(matrix, k=1)
    return matrix[tri]


def mantel_test(a: np.ndarray, b: np.ndarray, permutations: int, random_state: int) -> tuple[float, float]:
    observed = spearmanr(upper_triangle_values(a), upper_triangle_values(b)).statistic
    print(f"mantel: observed spearman = {observed:.4f}")
    rng = np.random.default_rng(random_state)
    exceed = 0
    step = max(1, permutations // 10)
    for index in range(permutations):
        perm = rng.permutation(a.shape[0])
        permuted_b = b[np.ix_(perm, perm)]
        value = spearmanr(upper_triangle_values(a), upper_triangle_values(permuted_b)).statistic
        if abs(value) >= abs(observed):
            exceed += 1
        if (index + 1) % step == 0 or index + 1 == permutations:
            print(f"mantel: {index + 1}/{permutations} permutations")
    p_value = (exceed + 1) / (permutations + 1)
    return float(observed), float(p_value)


def build_rsms(params: dict[str, Any], config: dict[str, Any], table_dir: Path, written: list[tuple[Path, Any]]) -> None:
    print("RSM: loading acoustic data")
    acoustic = pd.read_csv(Path(params["paths"]["acoustic_norm_csv"]))
    acoustic = acoustic[~acoustic["formant_qc_fail"]].copy()
    phonemes = config["rsm_phonemes"]
    acoustic = acoustic[acoustic["phoneme"].isin(phonemes)].copy()
    sampled_idx = sample_rsm_indices(
        acoustic,
        phonemes=phonemes,
        per_phoneme=int(config["rsm_subsample_per_phoneme"]),
        random_state=int(config["random_state"]),
    )
    print(f"RSM: sampled {len(sampled_idx)} tokens across {len(phonemes)} phonemes")
    sampled = acoustic.loc[sampled_idx].copy()

    embeddings_dir = Path(params["paths"]["embeddings_dir"])
    whisper_layer = int(config["representative_layers"]["whisper"])
    xlsr_layer = int(config["representative_layers"]["xlsr"])
    print(f"RSM: loading whisper layer {whisper_layer} pca50")
    whisper = np.load(embeddings_dir / f"whisper_layer{whisper_layer}_pca50.npy")[sampled_idx]
    print(f"RSM: loading xlsr layer {xlsr_layer} pca50")
    xlsr = np.load(embeddings_dir / f"xlsr_layer{xlsr_layer}_pca50.npy")[sampled_idx]

    print("RSM: building similarity matrices")
    acoustic_rsm = -pairwise_distances(sampled[["F1_norm", "F2_norm"]].to_numpy(), metric="euclidean")
    whisper_rsm = cosine_similarity(whisper)
    xlsr_rsm = cosine_similarity(xlsr)

    rows = []
    for left_name, right_name, left_rsm, right_rsm in [
        ("acoustic", "whisper", acoustic_rsm, whisper_rsm),
        ("acoustic", "xlsr", acoustic_rsm, xlsr_rsm),
        ("whisper", "xlsr", whisper_rsm, xlsr_rsm),
    ]:
        print(f"Mantel test: {left_name} vs {right_name}")
        correlation, p_value = mantel_test(
            left_rsm,
            right_rsm,
            permutations=999,
            random_state=int(config["random_state"]),
        )
        rows.append(
            {
                "left": left_name,
                "right": right_name,
                "spearman_r": correlation,
                "p_value": p_value,
                "n_tokens": int(len(sampled)),
            }
        )

    save_csv(table_dir / "mantel_test.csv", pd.DataFrame(rows), written)


def main() -> None:
    raw_params = load_params(Path("params.yaml"))
    params = resolve_value(raw_params, raw_params)
    config = params["descriptive"]

    figure_dir = Path(params["paths"]["figures_dir"])
    table_dir = Path(params["paths"]["tables_dir"])
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[Path, Any]] = []

    print("Stage 1/3: acoustic descriptive")
    acoustic = load_acoustic_frame(params, config)
    if not output_exists(table_dir / "acoustic_descriptive.csv"):
        save_csv(table_dir / "acoustic_descriptive.csv", build_acoustic_descriptive(acoustic), written)
    if not output_exists(table_dir / "variance_decomposition.csv"):
        save_csv(table_dir / "variance_decomposition.csv", build_variance_decomposition(acoustic), written)
    plot_vowel_chart(acoustic, figure_dir / "vowel_chart.png", written)
    plot_boxplots(acoustic, figure_dir / "boxplot_F1_F2.png", written)
    plot_violin(acoustic, figure_dir / "violin_intra_speaker.png", written)
    print("Stage 2/3: neural descriptive")
    build_neural_tables_and_figures(params, config, figure_dir, table_dir, written)
    print("Stage 3/3: RSM and Mantel tests")
    build_rsms(params, config, table_dir, written)

    print("Summary:")
    for path, shape in written:
        print(f"{path}: {shape}")


if __name__ == "__main__":
    main()
