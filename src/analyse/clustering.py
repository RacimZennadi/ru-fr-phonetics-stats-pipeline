from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


SECTION9_VOWELS = ["i", "e", "ɛ", "a", "y", "ø", "u", "o"]
SECTION9_CONSONANTS = ["p", "t", "k", "s", "f", "n", "m", "l"]


def load_params(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def resolve_value(value: Any, root: dict[str, Any]) -> Any:
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


def all_outputs_exist(paths: list[Path]) -> bool:
    if all(path.exists() for path in paths):
        for path in paths:
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


def label_variants(label: str) -> list[str]:
    variants = {
        label,
        unicodedata.normalize("NFC", label),
        unicodedata.normalize("NFD", label),
    }
    for encoding in ["latin1", "cp1252"]:
        try:
            repaired = label.encode(encoding).decode("utf-8")
        except Exception:
            continue
        variants.add(repaired)
        variants.add(unicodedata.normalize("NFC", repaired))
    return list(variants)


def resolve_label(label: str, observed: list[str]) -> str:
    observed_map: dict[str, str] = {}
    for value in observed:
        for variant in label_variants(value):
            observed_map[variant] = value
    for variant in label_variants(label):
        if variant in observed_map:
            return observed_map[variant]
    raise ValueError(f"Could not resolve phoneme label {label!r}")


def section9_vowels(observed: list[str]) -> list[str]:
    return [resolve_label(label, observed) for label in SECTION9_VOWELS]


def clustering_random_state(params: dict[str, Any]) -> int:
    return int(params.get("bootstrap", {}).get("random_state", params.get("project", {}).get("seed", 42)))


def load_tables(params: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    acoustic_norm = pd.read_csv(Path(params["paths"]["acoustic_norm_csv"]))
    acoustic_raw = pd.read_csv(Path(params["paths"]["acoustic_csv"]))
    tokens = pd.read_csv(Path(params["paths"]["phoneme_tokens_csv"]))
    return acoustic_norm, acoustic_raw, tokens


def representative_layer_matrix(params: dict[str, Any], model: str) -> tuple[np.ndarray, pd.DataFrame]:
    layer = int(params["tests"]["representative_layers"][model])
    prefix = Path(params["paths"]["embeddings_dir"]) / f"{model}_layer{layer}"
    matrix = np.load(prefix.with_name(f"{prefix.name}_pca50.npy"))
    meta = pd.read_csv(prefix.with_name(f"{prefix.name}_meta.csv")).reset_index(drop=True)
    return matrix, meta


def vowel_truth(vowels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    front_back = {vowels[i]: 0 for i in [0, 1, 2, 4, 5]}
    front_back.update({vowels[i]: 1 for i in [3, 6, 7]})
    high_mid_low = {vowels[i]: 2 for i in [0, 4, 6]}
    high_mid_low.update({vowels[i]: 1 for i in [1, 2, 5, 7]})
    high_mid_low[vowels[3]] = 0
    return (
        np.array([front_back[vowel] for vowel in vowels]),
        np.array([high_mid_low[vowel] for vowel in vowels]),
    )


def linkage_and_distance(matrix: np.ndarray, metric: str, method: str) -> tuple[np.ndarray, np.ndarray]:
    if metric == "euclidean":
        condensed = pdist(matrix, metric="euclidean")
        return linkage(condensed, method=method), squareform(condensed)
    distance = 1.0 - cosine_similarity(matrix)
    distance = np.clip(distance, 0.0, None)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    return linkage(condensed, method=method), distance


def safe_silhouette(matrix: np.ndarray, distance_matrix: np.ndarray, labels: np.ndarray, metric: str) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    if metric == "euclidean":
        return float(silhouette_score(matrix, labels, metric="euclidean"))
    return float(silhouette_score(distance_matrix, labels, metric="precomputed"))


def color_tick_labels(ax: plt.Axes, color_map: dict[str, str]) -> None:
    for tick in ax.get_xmajorticklabels():
        text = tick.get_text()
        if text in color_map:
            tick.set_color(color_map[text])


def plot_dendrogram(
    path: Path,
    linkage_matrix: np.ndarray,
    labels: list[str],
    title: str,
    written: list[tuple[Path, Any]],
    color_map: dict[str, str] | None = None,
) -> None:
    if output_exists(path):
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    dendrogram(linkage_matrix, labels=labels, ax=ax, leaf_rotation=0, leaf_font_size=10)
    ax.set_title(title)
    ax.set_ylabel("Distance")
    if color_map is not None:
        color_tick_labels(ax, color_map)
    save_figure(path, fig, written)


def mean_vectors_by_phoneme(frame: pd.DataFrame, values: list[str], phonemes: list[str]) -> pd.DataFrame:
    return frame.groupby("phoneme")[values].mean().reindex(phonemes)


def neural_frame(matrix: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    frame = meta.copy().reset_index(drop=True)
    for index in range(matrix.shape[1]):
        frame[f"dim_{index}"] = matrix[:, index]
    return frame


def vowel_clustering(
    params: dict[str, Any],
    acoustic_norm: pd.DataFrame,
    written: list[tuple[Path, Any]],
    table_dir: Path,
    figure_dir: Path,
) -> pd.DataFrame:
    outputs = [
        table_dir / "vowel_clustering_ari.csv",
        table_dir / "silhouette_by_k.csv",
        figure_dir / "dendrogram_acoustic.png",
        figure_dir / "dendrogram_whisper.png",
        figure_dir / "dendrogram_xlsr.png",
    ]
    if all_outputs_exist(outputs):
        return pd.read_csv(table_dir / "silhouette_by_k.csv")

    observed = sorted(acoustic_norm["phoneme"].dropna().unique().tolist())
    vowels = section9_vowels(observed)
    acoustic = acoustic_norm.loc[acoustic_norm["phoneme"].isin(vowels) & ~acoustic_norm["formant_qc_fail"]].copy()
    acoustic_centroids = mean_vectors_by_phoneme(acoustic, ["F1_norm", "F2_norm"], vowels)

    whisper_matrix, whisper_meta = representative_layer_matrix(params, "whisper")
    whisper_centroids = mean_vectors_by_phoneme(neural_frame(whisper_matrix, whisper_meta), [f"dim_{i}" for i in range(50)], vowels)

    xlsr_matrix, xlsr_meta = representative_layer_matrix(params, "xlsr")
    xlsr_centroids = mean_vectors_by_phoneme(neural_frame(xlsr_matrix, xlsr_meta), [f"dim_{i}" for i in range(50)], vowels)

    representations = {
        "acoustic": (acoustic_centroids.to_numpy(), "euclidean", "ward"),
        # Average linkage is used for cosine distances because Ward assumes Euclidean geometry.
        "whisper": (whisper_centroids.to_numpy(), "cosine", "average"),
        "xlsr": (xlsr_centroids.to_numpy(), "cosine", "average"),
    }
    truth_front_back, truth_height = vowel_truth(vowels)
    ari_rows: list[dict[str, Any]] = []
    silhouette_rows: list[dict[str, Any]] = []

    for name, (matrix, metric, method) in representations.items():
        print(f"vowel clustering: {name}")
        linkage_matrix, distance_matrix = linkage_and_distance(matrix, metric, method)
        labels_k2 = fcluster(linkage_matrix, t=2, criterion="maxclust")
        labels_k3 = fcluster(linkage_matrix, t=3, criterion="maxclust")
        ari_rows.append(
            {
                "representation": name,
                "partition": "front_back",
                "k": 2,
                "ARI": float(adjusted_rand_score(truth_front_back, labels_k2)),
                "silhouette": safe_silhouette(matrix, distance_matrix, labels_k2, metric),
            }
        )
        ari_rows.append(
            {
                "representation": name,
                "partition": "high_mid_low",
                "k": 3,
                "ARI": float(adjusted_rand_score(truth_height, labels_k3)),
                "silhouette": safe_silhouette(matrix, distance_matrix, labels_k3, metric),
            }
        )
        for k in [2, 3, 4, 5]:
            labels = fcluster(linkage_matrix, t=k, criterion="maxclust")
            silhouette_rows.append(
                {
                    "representation": name,
                    "k": k,
                    "silhouette": safe_silhouette(matrix, distance_matrix, labels, metric),
                }
            )
        plot_dendrogram(figure_dir / f"dendrogram_{name}.png", linkage_matrix, vowels, f"{name} vowel dendrogram", written)

    save_csv(table_dir / "vowel_clustering_ari.csv", pd.DataFrame(ari_rows), written)
    silhouette_frame = pd.DataFrame(silhouette_rows)
    save_csv(table_dir / "silhouette_by_k.csv", silhouette_frame, written)
    return silhouette_frame


def consonant_vowel_clustering(
    params: dict[str, Any],
    acoustic_norm: pd.DataFrame,
    acoustic_raw: pd.DataFrame,
    tokens: pd.DataFrame,
    written: list[tuple[Path, Any]],
    table_dir: Path,
    figure_dir: Path,
) -> None:
    outputs = [
        table_dir / "consonant_vowel_clustering_ari.csv",
        figure_dir / "dendrogram_cv_acoustic.png",
        figure_dir / "dendrogram_cv_whisper.png",
        figure_dir / "dendrogram_cv_xlsr.png",
    ]
    if all_outputs_exist(outputs):
        return

    observed = sorted(tokens["phoneme"].dropna().unique().tolist())
    vowels = section9_vowels(observed)
    consonants: list[str] = []
    for label in SECTION9_CONSONANTS:
        try:
            consonants.append(resolve_label(label, observed))
        except ValueError:
            print(f"warning: consonant {label} missing from corpus, skipping")
    phonemes = vowels + [label for label in consonants if label not in vowels]

    acoustic = acoustic_norm.merge(
        tokens[["speaker_id", "sentence_id", "repetition", "phoneme", "onset", "offset", "duration_ms"]].rename(columns={"duration_ms": "duration_ms_tokens"}),
        on=["speaker_id", "sentence_id", "repetition", "phoneme", "onset", "offset"],
        how="left",
    ).merge(
        acoustic_raw[["speaker_id", "sentence_id", "repetition", "phoneme", "onset", "offset", "SCG"]].rename(columns={"SCG": "scg"}),
        on=["speaker_id", "sentence_id", "repetition", "phoneme", "onset", "offset"],
        how="left",
    )
    acoustic["duration_ms_tokens"] = acoustic["duration_ms_tokens"].fillna(acoustic["duration_ms"])
    acoustic_subset = acoustic.loc[acoustic["phoneme"].isin(phonemes)].copy()
    acoustic_centroids = mean_vectors_by_phoneme(acoustic_subset, ["F1_norm", "F2_norm", "duration_ms_tokens", "scg"], phonemes).fillna(0.0)

    whisper_matrix, whisper_meta = representative_layer_matrix(params, "whisper")
    whisper_centroids = mean_vectors_by_phoneme(neural_frame(whisper_matrix, whisper_meta), [f"dim_{i}" for i in range(50)], phonemes).fillna(0.0)

    xlsr_matrix, xlsr_meta = representative_layer_matrix(params, "xlsr")
    xlsr_centroids = mean_vectors_by_phoneme(neural_frame(xlsr_matrix, xlsr_meta), [f"dim_{i}" for i in range(50)], phonemes).fillna(0.0)

    truth = np.array([0 if phoneme in vowels else 1 for phoneme in phonemes])
    color_map = {phoneme: ("blue" if phoneme in vowels else "red") for phoneme in phonemes}
    rows: list[dict[str, Any]] = []
    representations = {
        "acoustic": (acoustic_centroids.to_numpy(), "euclidean", "ward"),
        "whisper": (whisper_centroids.to_numpy(), "cosine", "average"),
        "xlsr": (xlsr_centroids.to_numpy(), "cosine", "average"),
    }

    for name, (matrix, metric, method) in representations.items():
        print(f"consonant-vowel clustering: {name}")
        linkage_matrix, distance_matrix = linkage_and_distance(matrix, metric, method)
        labels = fcluster(linkage_matrix, t=2, criterion="maxclust")
        rows.append(
            {
                "representation": name,
                "ARI": float(adjusted_rand_score(truth, labels)),
                "silhouette": safe_silhouette(matrix, distance_matrix, labels, metric),
            }
        )
        plot_dendrogram(
            figure_dir / f"dendrogram_cv_{name}.png",
            linkage_matrix,
            phonemes,
            f"{name} consonant-vowel dendrogram",
            written,
            color_map=color_map,
        )

    save_csv(table_dir / "consonant_vowel_clustering_ari.csv", pd.DataFrame(rows), written)


def build_speaker_matrix(frame: pd.DataFrame, value_columns: list[str], vowels: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    speakers = (
        frame[["speaker_id", "l1_status", "gender"]]
        .drop_duplicates("speaker_id")
        .sort_values("speaker_id")
        .reset_index(drop=True)
    )
    global_means = frame.groupby("phoneme")[value_columns].mean()
    vectors: list[np.ndarray] = []
    for speaker in speakers["speaker_id"]:
        speaker_frame = frame.loc[frame["speaker_id"] == speaker]
        parts: list[np.ndarray] = []
        for vowel in vowels:
            vowel_frame = speaker_frame.loc[speaker_frame["phoneme"] == vowel, value_columns]
            if vowel_frame.empty:
                parts.append(global_means.loc[vowel].to_numpy())
            else:
                parts.append(vowel_frame.mean().to_numpy())
        vectors.append(np.concatenate(parts))
    return np.vstack(vectors), speakers


def speaker_clustering(
    params: dict[str, Any],
    acoustic_norm: pd.DataFrame,
    written: list[tuple[Path, Any]],
    table_dir: Path,
    figure_dir: Path,
) -> None:
    outputs = [
        table_dir / "speaker_clustering_ari.csv",
        figure_dir / "dendrogram_speakers_acoustic.png",
        figure_dir / "dendrogram_speakers_whisper.png",
        figure_dir / "dendrogram_speakers_xlsr.png",
    ]
    if all_outputs_exist(outputs):
        return

    observed = sorted(acoustic_norm["phoneme"].dropna().unique().tolist())
    vowels = section9_vowels(observed)
    acoustic = acoustic_norm.loc[acoustic_norm["phoneme"].isin(vowels) & ~acoustic_norm["formant_qc_fail"]].copy()
    acoustic_matrix, speakers = build_speaker_matrix(acoustic, ["F1_norm", "F2_norm"], vowels)

    whisper_matrix, whisper_meta = representative_layer_matrix(params, "whisper")
    whisper = neural_frame(whisper_matrix, whisper_meta)
    whisper = whisper.loc[whisper["phoneme"].isin(vowels)].copy()
    whisper_matrix_speakers, _ = build_speaker_matrix(whisper, [f"dim_{i}" for i in range(50)], vowels)
    whisper_matrix_speakers = PCA(
        n_components=min(10, whisper_matrix_speakers.shape[0], whisper_matrix_speakers.shape[1]),
        random_state=clustering_random_state(params),
    ).fit_transform(whisper_matrix_speakers)

    xlsr_matrix, xlsr_meta = representative_layer_matrix(params, "xlsr")
    xlsr = neural_frame(xlsr_matrix, xlsr_meta)
    xlsr = xlsr.loc[xlsr["phoneme"].isin(vowels)].copy()
    xlsr_matrix_speakers, _ = build_speaker_matrix(xlsr, [f"dim_{i}" for i in range(50)], vowels)
    xlsr_matrix_speakers = PCA(
        n_components=min(10, xlsr_matrix_speakers.shape[0], xlsr_matrix_speakers.shape[1]),
        random_state=clustering_random_state(params),
    ).fit_transform(xlsr_matrix_speakers)

    truth_l1 = (speakers["l1_status"] == "ru").astype(int).to_numpy()
    truth_gender = (speakers["gender"] == "m").astype(int).to_numpy()
    color_map = {speaker: ("blue" if l1 == "fr" else "red") for speaker, l1 in zip(speakers["speaker_id"], speakers["l1_status"])}
    rows: list[dict[str, Any]] = []
    representations = {
        "acoustic": acoustic_matrix,
        "whisper": whisper_matrix_speakers,
        "xlsr": xlsr_matrix_speakers,
    }

    for name, matrix in representations.items():
        print(f"speaker clustering: {name}")
        linkage_matrix, distance_matrix = linkage_and_distance(matrix, "euclidean", "ward")
        labels = fcluster(linkage_matrix, t=2, criterion="maxclust")
        sil = safe_silhouette(matrix, distance_matrix, labels, "euclidean")
        rows.append({"representation": name, "ground_truth": "l1_status", "ARI": float(adjusted_rand_score(truth_l1, labels)), "silhouette": sil})
        rows.append({"representation": name, "ground_truth": "gender", "ARI": float(adjusted_rand_score(truth_gender, labels)), "silhouette": sil})
        plot_dendrogram(
            figure_dir / f"dendrogram_speakers_{name}.png",
            linkage_matrix,
            speakers["speaker_id"].tolist(),
            f"{name} speaker dendrogram",
            written,
            color_map=color_map,
        )

    save_csv(table_dir / "speaker_clustering_ari.csv", pd.DataFrame(rows), written)


def plot_silhouette_by_k(frame: pd.DataFrame, path: Path, written: list[tuple[Path, Any]]) -> None:
    if output_exists(path):
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for representation, sub in frame.groupby("representation"):
        sub = sub.sort_values("k")
        ax.plot(sub["k"], sub["silhouette"], marker="o", label=representation)
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette")
    ax.set_title("Silhouette by k for vowel clustering")
    ax.legend()
    save_figure(path, fig, written)


def main() -> None:
    raw_params = load_params(Path("params.yaml"))
    params = resolve_value(raw_params, raw_params)
    figure_dir = Path(params["paths"]["figures_dir"])
    table_dir = Path(params["paths"]["tables_dir"])
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[Path, Any]] = []

    acoustic_norm, acoustic_raw, tokens = load_tables(params)

    print("Section 9.1: vowel clustering")
    silhouette_frame = vowel_clustering(params, acoustic_norm, written, table_dir, figure_dir)

    print("Section 9.2: consonant-vowel clustering")
    consonant_vowel_clustering(params, acoustic_norm, acoustic_raw, tokens, written, table_dir, figure_dir)

    print("Section 9.3: speaker clustering")
    speaker_clustering(params, acoustic_norm, written, table_dir, figure_dir)

    print("Section 9.4: number of clusters")
    plot_silhouette_by_k(silhouette_frame, figure_dir / "silhouette_by_k.png", written)

    print("Summary:")
    for path, shape in written:
        print(f"{path}: {shape}")


if __name__ == "__main__":
    main()
