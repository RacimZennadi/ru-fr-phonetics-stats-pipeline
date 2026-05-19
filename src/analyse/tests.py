from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2, levene, mannwhitneyu, probplot, shapiro, spearmanr, ttest_ind
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics.pairwise import cosine_distances
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


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


def save_csv(path: Path, frame: pd.DataFrame, written: list[tuple[Path, Any]]) -> None:
    if output_exists(path):
        return
    ensure_parent(path)
    frame.to_csv(path, index=True if frame.index.name else False)
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


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


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


def resolve_labels(labels: list[str], observed: list[str]) -> list[str]:
    resolved = []
    for label in labels:
        value = resolve_label(label, observed)
        if value not in resolved:
            resolved.append(value)
    return resolved


def resolve_pairs(pairs: list[tuple[str, str]], observed: list[str]) -> list[tuple[str, str]]:
    return [(resolve_label(left, observed), resolve_label(right, observed)) for left, right in pairs]


def load_tables(params: dict[str, Any], vowels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    acoustic_path = Path(params["paths"]["acoustic_norm_csv"])
    tokens_path = Path(params["paths"]["phoneme_tokens_csv"])
    require_file(acoustic_path)
    require_file(tokens_path)

    acoustic = pd.read_csv(acoustic_path)
    tokens = pd.read_csv(tokens_path)
    if len(acoustic) != len(tokens):
        raise ValueError("features_acoustic_norm.csv and phoneme_tokens.csv are not row-aligned")
    for column in ["speaker_id", "sentence_id", "l1_status", "gender", "phoneme"]:
        if column not in acoustic.columns and column in tokens.columns:
            acoustic[column] = tokens[column]
    acoustic["token_idx"] = np.arange(len(acoustic))
    tokens["token_idx"] = np.arange(len(tokens))

    counts = tokens["phoneme"].value_counts()
    eligible = [phoneme for phoneme in vowels if counts.get(phoneme, 0) >= 10]
    acoustic = acoustic[acoustic["phoneme"].isin(eligible) & ~acoustic["formant_qc_fail"]].copy()
    tokens = tokens[tokens["phoneme"].isin(eligible)].copy()
    return acoustic, tokens


def l1_masks(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    return frame["l1_status"] == "fr", frame["l1_status"] == "ru"


def acoustic_group_tests(acoustic: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for phoneme in sorted(acoustic["phoneme"].unique()):
        sub = acoustic[acoustic["phoneme"] == phoneme]
        l1_mask, l2_mask = l1_masks(sub)
        for formant in ["F1_norm", "F2_norm"]:
            left = sub.loc[l1_mask, formant].dropna().to_numpy()
            right = sub.loc[l2_mask, formant].dropna().to_numpy()
            shapiro_left = shapiro(left).pvalue if len(left) >= 3 else np.nan
            shapiro_right = shapiro(right).pvalue if len(right) >= 3 else np.nan
            levene_p = levene(left, right).pvalue if len(left) >= 2 and len(right) >= 2 else np.nan
            if np.isfinite(shapiro_left) and np.isfinite(shapiro_right) and shapiro_left > alpha and shapiro_right > alpha:
                equal_var = bool(np.isfinite(levene_p) and levene_p > alpha)
                result = ttest_ind(left, right, equal_var=equal_var)
                test_used = "t_test"
                statistic = float(result.statistic)
                p_raw = float(result.pvalue)
            else:
                result = mannwhitneyu(left, right, alternative="two-sided")
                test_used = "mannwhitney_u"
                statistic = float(result.statistic)
                p_raw = float(result.pvalue)
            rows.append(
                {
                    "phoneme": phoneme,
                    "formant": formant,
                    "test_used": test_used,
                    "statistic": statistic,
                    "p_raw": p_raw,
                    "shapiro_l1_p": shapiro_left,
                    "shapiro_l2_p": shapiro_right,
                    "levene_p": levene_p,
                }
            )

    frame = pd.DataFrame(rows)
    reject, p_fdr, _, _ = multipletests(frame["p_raw"], method="fdr_bh", alpha=alpha)
    frame["p_fdr"] = p_fdr
    frame["significant_fdr"] = reject
    return frame[["phoneme", "formant", "test_used", "statistic", "p_raw", "p_fdr", "significant_fdr"]]


def qq_plot_grid(acoustic: pd.DataFrame, path: Path, written: list[tuple[Path, Any]]) -> None:
    if output_exists(path):
        return
    phonemes = sorted(acoustic["phoneme"].unique())
    fig, axes = plt.subplots(len(phonemes), 2, figsize=(10, 3 * len(phonemes)))
    colors = {"fr": "#4C78A8", "ru": "#F58518"}
    labels = {"fr": "L1", "ru": "L2"}
    if len(phonemes) == 1:
        axes = np.array([axes])
    for row, phoneme in enumerate(phonemes):
        sub = acoustic[acoustic["phoneme"] == phoneme]
        for col, formant in enumerate(["F1_norm", "F2_norm"]):
            ax = axes[row, col]
            for code in ["fr", "ru"]:
                values = sub.loc[sub["l1_status"] == code, formant].dropna().to_numpy()
                if len(values) < 2:
                    continue
                (theoretical, ordered), (slope, intercept, _) = probplot(values, dist="norm", fit=True)
                ax.scatter(theoretical, ordered, s=10, alpha=0.7, color=colors[code], label=labels[code])
                x_line = np.array([theoretical.min(), theoretical.max()])
                ax.plot(x_line, slope * x_line + intercept, color=colors[code], linewidth=1.0)
            ax.set_title(f"/{phoneme}/ {formant}")
            ax.set_xlabel("Theoretical quantiles")
            ax.set_ylabel("Sample quantiles")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=colors[key], label=labels[key]) for key in colors]
    axes[0, 1].legend(handles=handles, loc="best")
    fig.suptitle("Q-Q plots for normalised formants by vowel")
    save_figure(path, fig, written)


def gender_test(acoustic: pd.DataFrame) -> pd.DataFrame:
    speaker_means = (
        acoustic.groupby(["speaker_id", "gender"])[["F1_norm", "F2_norm"]]
        .mean()
        .reset_index()
    )
    rows = []
    for formant in ["F1_norm", "F2_norm"]:
        female = speaker_means.loc[speaker_means["gender"] == "f", formant].to_numpy()
        male = speaker_means.loc[speaker_means["gender"] == "m", formant].to_numpy()
        result = mannwhitneyu(female, male, alternative="two-sided")
        rows.append(
            {
                "formant": formant,
                "statistic": float(result.statistic),
                "p_raw": float(result.pvalue),
                "n_female_speakers": len(female),
                "n_male_speakers": len(male),
            }
        )
    return pd.DataFrame(rows)


def load_embedding_views(params: dict[str, Any], model: str, layer: int) -> tuple[np.ndarray, pd.DataFrame]:
    prefix = Path(params["paths"]["embeddings_dir"]) / f"{model}_layer{layer}"
    matrix = np.load(prefix.with_name(f"{prefix.name}_pca50.npy"))
    meta = pd.read_csv(prefix.with_name(f"{prefix.name}_meta.csv"))
    return matrix, meta.reset_index(drop=True)


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    value = cosine_distances(left.reshape(1, -1), right.reshape(1, -1))[0, 0]
    return float(value)


def neural_group_tests(params: dict[str, Any], vowels: list[str], permutations: int, alpha: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, layer in params["tests"]["representative_layers"].items():
        print(f"neural group tests: {model} layer {layer}")
        matrix, meta = load_embedding_views(params, model, int(layer))
        mask = meta["phoneme"].isin(vowels).to_numpy()
        matrix = matrix[mask]
        meta = meta.loc[mask].reset_index(drop=True)
        model_rows: list[dict[str, Any]] = []
        for phoneme in sorted(vowels):
            sub = meta[meta["phoneme"] == phoneme].copy()
            sub_matrix = matrix[sub.index.to_numpy()]
            labels = sub["l1_status"].to_numpy()
            l1_mask = labels == "fr"
            l2_mask = labels == "ru"
            observed = cosine_distance(sub_matrix[l1_mask].mean(axis=0), sub_matrix[l2_mask].mean(axis=0))
            rng = np.random.default_rng(params["bootstrap"]["random_state"] + int(layer) + sum(ord(ch) for ch in phoneme))
            exceed = 0
            for _ in range(permutations):
                permuted = rng.permutation(labels)
                perm_l1 = permuted == "fr"
                perm_l2 = permuted == "ru"
                value = cosine_distance(sub_matrix[perm_l1].mean(axis=0), sub_matrix[perm_l2].mean(axis=0))
                if value >= observed:
                    exceed += 1
            p_raw = (exceed + 1) / (permutations + 1)
            model_rows.append(
                {
                    "model": model,
                    "phoneme": phoneme,
                    "observed_distance": observed,
                    "p_raw": p_raw,
                }
            )
        model_frame = pd.DataFrame(model_rows)
        reject, p_fdr, _, _ = multipletests(model_frame["p_raw"], method="fdr_bh", alpha=alpha)
        model_frame["p_fdr"] = p_fdr
        model_frame["significant_fdr"] = reject
        rows.extend(model_frame.to_dict("records"))
    return pd.DataFrame(rows)[["model", "phoneme", "observed_distance", "p_raw", "p_fdr", "significant_fdr"]]


def centroid_matrix(centroids: dict[str, np.ndarray], phonemes: list[str], metric: str) -> pd.DataFrame:
    data = []
    for left in phonemes:
        row = []
        for right in phonemes:
            if metric == "euclidean":
                row.append(float(np.linalg.norm(centroids[left] - centroids[right])))
            elif metric == "cosine":
                row.append(cosine_distance(centroids[left], centroids[right]))
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        data.append(row)
    return pd.DataFrame(data, index=phonemes, columns=phonemes)


def acoustic_distance_matrices(acoustic: pd.DataFrame, vowels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    centroids = {
        phoneme: sub[["F1_norm", "F2_norm"]].mean().to_numpy()
        for phoneme, sub in acoustic.groupby("phoneme")
        if phoneme in vowels
    }
    euclidean = centroid_matrix(centroids, vowels, "euclidean")

    residuals = []
    for phoneme, sub in acoustic.groupby("phoneme"):
        if phoneme not in vowels:
            continue
        centered = sub[["F1_norm", "F2_norm"]].to_numpy() - centroids[phoneme]
        residuals.append(centered)
    pooled = np.vstack(residuals)
    covariance = np.cov(pooled, rowvar=False)
    inverse = np.linalg.pinv(covariance)
    maha = []
    for left in vowels:
        row = []
        for right in vowels:
            diff = centroids[left] - centroids[right]
            row.append(float(np.sqrt(diff.T @ inverse @ diff)))
        maha.append(row)
    mahalanobis = pd.DataFrame(maha, index=vowels, columns=vowels)
    return euclidean, mahalanobis


def neural_distance_matrix(params: dict[str, Any], model: str, layer: int, vowels: list[str]) -> pd.DataFrame:
    matrix, meta = load_embedding_views(params, model, layer)
    mask = meta["phoneme"].isin(vowels).to_numpy()
    matrix = matrix[mask]
    meta = meta.loc[mask].reset_index(drop=True)
    centroids = {
        phoneme: matrix[meta["phoneme"] == phoneme].mean(axis=0)
        for phoneme in vowels
    }
    return centroid_matrix(centroids, vowels, "cosine")


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    tri = np.triu_indices_from(matrix, k=1)
    return matrix[tri]


def mantel_test(left: np.ndarray, right: np.ndarray, permutations: int, random_state: int) -> tuple[float, float]:
    observed = spearmanr(upper_triangle_values(left), upper_triangle_values(right)).statistic
    rng = np.random.default_rng(random_state)
    exceed = 0
    for _ in range(permutations):
        perm = rng.permutation(left.shape[0])
        permuted = right[np.ix_(perm, perm)]
        value = spearmanr(upper_triangle_values(left), upper_triangle_values(permuted)).statistic
        if abs(value) >= abs(observed):
            exceed += 1
    p_value = (exceed + 1) / (permutations + 1)
    return float(observed), float(p_value)


def centroid_mantel_tables(
    acoustic_euclidean: pd.DataFrame,
    whisper: pd.DataFrame,
    xlsr: pd.DataFrame,
    permutations: int,
    random_state: int,
) -> pd.DataFrame:
    rows = []
    for left_name, right_name, left, right in [
        ("acoustic_euclidean", "whisper", acoustic_euclidean.to_numpy(), whisper.to_numpy()),
        ("acoustic_euclidean", "xlsr", acoustic_euclidean.to_numpy(), xlsr.to_numpy()),
        ("whisper", "xlsr", whisper.to_numpy(), xlsr.to_numpy()),
    ]:
        correlation, p_value = mantel_test(left, right, permutations, random_state)
        rows.append({"left": left_name, "right": right_name, "spearman_r": correlation, "p_value": p_value})
    return pd.DataFrame(rows)


def bootstrap_sample_frame(frame: pd.DataFrame, speakers: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    draws = rng.choice(speakers, size=len(speakers), replace=True)
    pieces = []
    for draw_id, speaker in enumerate(draws):
        piece = frame[frame["speaker_id"] == speaker].copy()
        piece["boot_speaker"] = f"{speaker}_{draw_id}"
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def bootstrap_sample_vectors(meta: pd.DataFrame, vectors: np.ndarray, speakers: np.ndarray, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    draws = rng.choice(speakers, size=len(speakers), replace=True)
    meta_parts = []
    vec_parts = []
    for draw_id, speaker in enumerate(draws):
        idx = meta.index[meta["speaker_id"] == speaker].to_numpy()
        part = meta.loc[idx].copy()
        part["boot_speaker"] = f"{speaker}_{draw_id}"
        meta_parts.append(part)
        vec_parts.append(vectors[idx])
    return pd.concat(meta_parts, ignore_index=True), np.vstack(vec_parts)


def centroid_distance_for_pair(frame: pd.DataFrame, pair: tuple[str, str], columns: list[str]) -> float:
    left_values = frame.loc[frame["phoneme"] == pair[0], columns].to_numpy()
    right_values = frame.loc[frame["phoneme"] == pair[1], columns].to_numpy()
    if len(left_values) == 0 or len(right_values) == 0:
        raise ValueError(f"Missing phoneme in bootstrap sample for pair {pair}")
    left = left_values.mean(axis=0)
    right = right_values.mean(axis=0)
    return float(np.linalg.norm(left - right))


def neural_centroid_distance_for_pair(meta: pd.DataFrame, vectors: np.ndarray, pair: tuple[str, str]) -> float:
    left_values = vectors[meta["phoneme"] == pair[0]]
    right_values = vectors[meta["phoneme"] == pair[1]]
    if len(left_values) == 0 or len(right_values) == 0:
        raise ValueError(f"Missing phoneme in bootstrap sample for pair {pair}")
    left = left_values.mean(axis=0)
    right = right_values.mean(axis=0)
    return cosine_distance(left, right)


def bootstrap_distance_cis(
    acoustic: pd.DataFrame,
    params: dict[str, Any],
    vowels: list[str],
    pairs: list[tuple[str, str]],
    n_resamples: int,
    random_state: int,
) -> pd.DataFrame:
    speakers = np.sort(acoustic["speaker_id"].unique())
    whisper_layer = int(params["tests"]["representative_layers"]["whisper"])
    xlsr_layer = int(params["tests"]["representative_layers"]["xlsr"])
    whisper_vectors, whisper_meta = load_embedding_views(params, "whisper", whisper_layer)
    xlsr_vectors, xlsr_meta = load_embedding_views(params, "xlsr", xlsr_layer)
    whisper_mask = whisper_meta["phoneme"].isin(vowels).to_numpy()
    xlsr_mask = xlsr_meta["phoneme"].isin(vowels).to_numpy()
    whisper_vectors = whisper_vectors[whisper_mask]
    xlsr_vectors = xlsr_vectors[xlsr_mask]
    whisper_meta = whisper_meta.loc[whisper_mask].reset_index(drop=True)
    xlsr_meta = xlsr_meta.loc[xlsr_mask].reset_index(drop=True)

    rows = []
    for pair in pairs:
        pair_name = f"{pair[0]}-{pair[1]}"
        for representation in ["acoustic_euclidean", "whisper", "xlsr"]:
            print(f"bootstrap CI: {pair_name} {representation}")
            values = []
            rng = np.random.default_rng(random_state + sum(ord(ch) for ch in pair_name + representation))
            step = max(1, n_resamples // 10)
            while len(values) < n_resamples:
                if representation == "acoustic_euclidean":
                    sample = bootstrap_sample_frame(acoustic, speakers, rng)
                    if not (sample["phoneme"] == pair[0]).any() or not (sample["phoneme"] == pair[1]).any():
                        continue
                    values.append(centroid_distance_for_pair(sample, pair, ["F1_norm", "F2_norm"]))
                elif representation == "whisper":
                    sample_meta, sample_vectors = bootstrap_sample_vectors(whisper_meta, whisper_vectors, speakers, rng)
                    if not (sample_meta["phoneme"] == pair[0]).any() or not (sample_meta["phoneme"] == pair[1]).any():
                        continue
                    values.append(neural_centroid_distance_for_pair(sample_meta, sample_vectors, pair))
                else:
                    sample_meta, sample_vectors = bootstrap_sample_vectors(xlsr_meta, xlsr_vectors, speakers, rng)
                    if not (sample_meta["phoneme"] == pair[0]).any() or not (sample_meta["phoneme"] == pair[1]).any():
                        continue
                    values.append(neural_centroid_distance_for_pair(sample_meta, sample_vectors, pair))
                if len(values) % step == 0 or len(values) == n_resamples:
                    print(f"bootstrap CI: {pair_name} {representation} {len(values)}/{n_resamples}")
            lower, upper = np.percentile(values, [2.5, 97.5])
            rows.append(
                {
                    "pair": pair_name,
                    "representation": representation,
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                }
            )
    return pd.DataFrame(rows)


def nearest_centroid_predict(train_vectors: np.ndarray, train_labels: np.ndarray, test_vectors: np.ndarray, labels: list[str], metric: str) -> np.ndarray:
    centroids = {label: train_vectors[train_labels == label].mean(axis=0) for label in labels}
    predictions = []
    for vector in test_vectors:
        distances = {}
        for label, centroid in centroids.items():
            if metric == "euclidean":
                distances[label] = float(np.linalg.norm(vector - centroid))
            else:
                distances[label] = cosine_distance(vector, centroid)
        predictions.append(min(distances, key=distances.get))
    return np.array(predictions)


def plot_confusion(matrix: np.ndarray, labels: list[str], title: str, path: Path, written: list[tuple[Path, Any]]) -> None:
    if output_exists(path):
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(path, fig, written)


def acoustic_classifier(acoustic: pd.DataFrame, vowels: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = acoustic.sort_values("token_idx").copy()
    speakers = np.sort(frame["speaker_id"].unique())
    truth = []
    predictions = []
    for speaker in speakers:
        train = frame[frame["speaker_id"] != speaker]
        test = frame[frame["speaker_id"] == speaker]
        pred = nearest_centroid_predict(
            train_vectors=train[["F1_norm", "F2_norm"]].to_numpy(),
            train_labels=train["phoneme"].to_numpy(),
            test_vectors=test[["F1_norm", "F2_norm"]].to_numpy(),
            labels=vowels,
            metric="euclidean",
        )
        truth.append(test["phoneme"].to_numpy())
        predictions.append(pred)
    y_true = np.concatenate(truth)
    y_pred = np.concatenate(predictions)
    f1_values = f1_score(y_true, y_pred, labels=vowels, average=None, zero_division=0)
    accuracy = float((y_true == y_pred).mean())
    rows = pd.DataFrame({"representation": "acoustic", "phoneme": vowels, "f1": f1_values, "overall_accuracy": accuracy})
    return rows, y_true, y_pred


def neural_classifier(
    acoustic: pd.DataFrame,
    params: dict[str, Any],
    model: str,
    layer: int,
    vowels: list[str],
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    raw = np.load(Path(params["paths"][f"{model}_npz"]))[f"layer_{layer}"]
    meta = acoustic[["token_idx", "speaker_id", "phoneme"]].sort_values("token_idx").copy()
    vectors = raw[meta["token_idx"].to_numpy()]
    speakers = np.sort(meta["speaker_id"].unique())
    truth = []
    predictions = []
    for fold, speaker in enumerate(speakers):
        train_mask = meta["speaker_id"] != speaker
        test_mask = ~train_mask
        pca = PCA(n_components=50, random_state=random_state + fold)
        train_proj = pca.fit_transform(vectors[train_mask.to_numpy()])
        test_proj = pca.transform(vectors[test_mask.to_numpy()])
        pred = nearest_centroid_predict(
            train_vectors=train_proj,
            train_labels=meta.loc[train_mask, "phoneme"].to_numpy(),
            test_vectors=test_proj,
            labels=vowels,
            metric="cosine",
        )
        truth.append(meta.loc[test_mask, "phoneme"].to_numpy())
        predictions.append(pred)
    y_true = np.concatenate(truth)
    y_pred = np.concatenate(predictions)
    f1_values = f1_score(y_true, y_pred, labels=vowels, average=None, zero_division=0)
    accuracy = float((y_true == y_pred).mean())
    rows = pd.DataFrame({"representation": model, "phoneme": vowels, "f1": f1_values, "overall_accuracy": accuracy})
    return rows, y_true, y_pred


def mcnemar_rows(y_true: np.ndarray, left_pred: np.ndarray, right_pred: np.ndarray, left_name: str, right_name: str) -> dict[str, Any]:
    left_correct = left_pred == y_true
    right_correct = right_pred == y_true
    both_correct = int(np.sum(left_correct & right_correct))
    left_only = int(np.sum(left_correct & ~right_correct))
    right_only = int(np.sum(~left_correct & right_correct))
    both_wrong = int(np.sum(~left_correct & ~right_correct))
    result = mcnemar([[both_correct, left_only], [right_only, both_wrong]], exact=False, correction=True)
    return {
        "left": left_name,
        "right": right_name,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "left_only_correct": left_only,
        "right_only_correct": right_only,
    }


def main() -> None:
    raw_params = load_params(Path("params.yaml"))
    params = resolve_value(raw_params, raw_params)
    requested_vowels = params["tests"]["vowels"]
    alpha = float(params["analyse"]["bh_alpha"])
    n_resamples = int(params["bootstrap"]["n_resamples"])
    permutation_n = int(params["bootstrap"]["permutation_n"])
    mantel_permutations = int(params["bootstrap"]["mantel_permutations"])
    random_state = int(params["bootstrap"]["random_state"])
    requested_pairs = [tuple(pair) for pair in params["tests"]["selected_pairs"]]

    figure_dir = Path(params["paths"]["figures_dir"])
    table_dir = Path(params["paths"]["tables_dir"])
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[Path, Any]] = []

    raw_acoustic = pd.read_csv(Path(params["paths"]["acoustic_norm_csv"]))
    raw_acoustic = raw_acoustic[~raw_acoustic["formant_qc_fail"]].copy()
    observed = sorted(raw_acoustic["phoneme"].dropna().unique().tolist())
    vowels = resolve_labels(requested_vowels, observed)
    pairs = resolve_pairs(requested_pairs, observed)
    acoustic, _ = load_tables(params, vowels)
    vowels = sorted(acoustic["phoneme"].unique())

    print("Section 6.1a: acoustic group tests")
    save_csv(table_dir / "acoustic_group_tests.csv", acoustic_group_tests(acoustic, alpha), written)
    qq_plot_grid(acoustic, figure_dir / "qq_plots.png", written)

    print("Section 6.1b: residual gender test")
    save_csv(table_dir / "gender_test.csv", gender_test(acoustic), written)

    print("Section 6.1c: neural group tests")
    save_csv(table_dir / "neural_group_tests.csv", neural_group_tests(params, vowels, permutation_n, alpha), written)

    print("Section 6.2a: distance matrices")
    acoustic_euclidean, acoustic_mahalanobis = acoustic_distance_matrices(acoustic, vowels)
    acoustic_euclidean.index.name = "phoneme"
    acoustic_mahalanobis.index.name = "phoneme"
    save_csv(table_dir / "acoustic_distance_euclidean.csv", acoustic_euclidean, written)
    save_csv(table_dir / "acoustic_distance_mahalanobis.csv", acoustic_mahalanobis, written)

    whisper_layer = int(params["tests"]["representative_layers"]["whisper"])
    xlsr_layer = int(params["tests"]["representative_layers"]["xlsr"])
    whisper_distance = neural_distance_matrix(params, "whisper", whisper_layer, vowels)
    xlsr_distance = neural_distance_matrix(params, "xlsr", xlsr_layer, vowels)
    whisper_distance.index.name = "phoneme"
    xlsr_distance.index.name = "phoneme"
    save_csv(table_dir / "whisper_distance.csv", whisper_distance, written)
    save_csv(table_dir / "xlsr_distance.csv", xlsr_distance, written)
    save_csv(
        table_dir / "centroid_mantel.csv",
        centroid_mantel_tables(acoustic_euclidean, whisper_distance, xlsr_distance, mantel_permutations, random_state),
        written,
    )

    print("Section 6.2b: bootstrap distance CIs")
    save_csv(
        table_dir / "distance_bootstrap_ci.csv",
        bootstrap_distance_cis(acoustic, params, vowels, pairs, n_resamples, random_state),
        written,
    )

    print("Section 6.2c: LOSO classifier")
    acoustic_rows, y_true, acoustic_pred = acoustic_classifier(acoustic, vowels)
    whisper_rows, y_true_whisper, whisper_pred = neural_classifier(acoustic, params, "whisper", whisper_layer, vowels, random_state)
    xlsr_rows, y_true_xlsr, xlsr_pred = neural_classifier(acoustic, params, "xlsr", xlsr_layer, vowels, random_state)
    if not (np.array_equal(y_true, y_true_whisper) and np.array_equal(y_true, y_true_xlsr)):
        raise ValueError("Classifier outputs are not aligned on the same token order")

    classifier_f1 = pd.concat([acoustic_rows, whisper_rows, xlsr_rows], ignore_index=True)
    save_csv(table_dir / "classifier_f1.csv", classifier_f1, written)
    save_csv(
        table_dir / "mcnemar_test.csv",
        pd.DataFrame(
            [
                mcnemar_rows(y_true, acoustic_pred, whisper_pred, "acoustic", "whisper"),
                mcnemar_rows(y_true, acoustic_pred, xlsr_pred, "acoustic", "xlsr"),
            ]
        ),
        written,
    )

    plot_confusion(
        confusion_matrix(y_true, acoustic_pred, labels=vowels),
        vowels,
        "Acoustic confusion matrix",
        figure_dir / "confusion_acoustic.png",
        written,
    )
    plot_confusion(
        confusion_matrix(y_true, whisper_pred, labels=vowels),
        vowels,
        "Whisper confusion matrix",
        figure_dir / "confusion_whisper.png",
        written,
    )
    plot_confusion(
        confusion_matrix(y_true, xlsr_pred, labels=vowels),
        vowels,
        "XLS-R confusion matrix",
        figure_dir / "confusion_xlsr.png",
        written,
    )

    print("Summary:")
    for path, shape in written:
        print(f"{path}: {shape}")


if __name__ == "__main__":
    main()
