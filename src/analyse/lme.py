from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2
from sklearn.metrics.pairwise import cosine_distances
from statsmodels.regression.mixed_linear_model import MixedLM


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


def build_acoustic_frame(params: dict[str, Any], requested_vowels: list[str]) -> tuple[pd.DataFrame, list[str]]:
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

    observed = sorted(acoustic.loc[~acoustic["formant_qc_fail"], "phoneme"].dropna().unique().tolist())
    vowels = resolve_labels(requested_vowels, observed)
    counts = acoustic.loc[~acoustic["formant_qc_fail"], "phoneme"].value_counts()
    vowels = [phoneme for phoneme in vowels if counts.get(phoneme, 0) >= 10]

    frame = acoustic[acoustic["phoneme"].isin(vowels) & ~acoustic["formant_qc_fail"]].copy()
    frame["l1_num"] = (frame["l1_status"] == "ru").astype(int)
    frame["gender_num"] = (frame["gender"] == "m").astype(int)
    return frame, vowels


def height_map(vowels: list[str]) -> dict[str, int]:
    observed = set(vowels)
    mapping = {}
    groups = {
        2: {"i", "y", "u"},
        1: {"e", "ɛ", "ø", "o"},
        0: {"a"},
    }
    for height, labels in groups.items():
        for label in labels:
            if label in observed:
                mapping[label] = height
    return mapping


def load_embedding_view(params: dict[str, Any], model: str, layer: int) -> tuple[np.ndarray, pd.DataFrame]:
    prefix = Path(params["paths"]["embeddings_dir"]) / f"{model}_layer{layer}"
    matrix = np.load(prefix.with_name(f"{prefix.name}_pca50.npy"))
    meta = pd.read_csv(prefix.with_name(f"{prefix.name}_meta.csv")).reset_index(drop=True)
    return matrix, meta


def load_neural_frame(params: dict[str, Any], acoustic: pd.DataFrame, vowels: list[str], model: str, layer: int) -> pd.DataFrame:
    matrix, meta = load_embedding_view(params, model, layer)
    mask = meta["phoneme"].isin(vowels).to_numpy()
    matrix = matrix[mask]
    meta = meta.loc[mask].reset_index(drop=True)
    frame = meta.copy()
    for index in range(5):
        frame[f"PC{index + 1}"] = matrix[:, index]
    frame["l1_num"] = (frame["l1_status"] == "ru").astype(int)
    frame["gender_num"] = (frame["gender"] == "m").astype(int)
    acoustic_lookup = acoustic[["token_idx", "phoneme"]].drop_duplicates("token_idx")
    frame = frame.merge(acoustic_lookup, on="token_idx", how="left", suffixes=("", "_acoustic"))
    if "phoneme_acoustic" in frame.columns:
        frame["phoneme"] = frame["phoneme_acoustic"].fillna(frame["phoneme"])
        frame = frame.drop(columns=["phoneme_acoustic"])
    return frame


def fit_mixedlm(formula: str, frame: pd.DataFrame, groups: str, re_formula: str | None, reml: bool):
    model = MixedLM.from_formula(formula, groups=frame[groups], re_formula=re_formula, data=frame)
    last_error: Exception | None = None
    for method in ["lbfgs", "powell", "cg"]:
        try:
            return model.fit(reml=reml, method=method, maxiter=300, disp=False)
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise RuntimeError(f"MixedLM failed for formula {formula}: {last_error}") from last_error
    raise RuntimeError(f"MixedLM failed for formula {formula}")


def model_specs_acoustic() -> list[tuple[str, str, str | None]]:
    return [
        ("null", "{response} ~ 1", "1"),
        ("main_effects", "{response} ~ l1_num + gender_num", "1"),
        ("full", "{response} ~ l1_num * gender_num", "1"),
        ("extended", "{response} ~ l1_num * gender_num + vowel_height", "1"),
        ("random_slope", "{response} ~ l1_num * gender_num + vowel_height", "1 + l1_num"),
    ]


def model_specs_neural() -> list[tuple[str, str, str | None]]:
    return [
        ("null", "{response} ~ 1", "1"),
        ("main_effects", "{response} ~ l1_num + gender_num", "1"),
        ("full", "{response} ~ l1_num * gender_num", "1"),
    ]


def lrt(simple, complex_result) -> tuple[float, float, int]:
    if not np.isfinite(simple.llf) or not np.isfinite(complex_result.llf):
        return np.nan, np.nan, 0
    stat = 2 * (complex_result.llf - simple.llf)
    df = len(complex_result.params) - len(simple.params)
    p_value = chi2.sf(stat, df)
    return float(stat), float(p_value), int(df)


def random_variance(result) -> float:
    cov_re = np.asarray(result.cov_re)
    if cov_re.ndim == 0:
        return float(cov_re)
    exog_re = result.model.exog_re
    if cov_re.shape == (1, 1):
        return float(cov_re[0, 0])
    row_vars = np.einsum("ij,jk,ik->i", exog_re, cov_re, exog_re)
    return float(np.mean(row_vars))


def icc_row(result, response: str) -> dict[str, Any]:
    var_random = random_variance(result)
    var_residual = float(result.scale)
    icc = var_random / (var_random + var_residual)
    return {
        "response": response,
        "var_random": var_random,
        "var_residual": var_residual,
        "ICC": icc,
    }


def fixed_effect_rows(result, response: str, model_name: str, extra: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    conf = result.conf_int()
    rows = []
    extra = extra or {}
    for term in result.fe_params.index:
        rows.append(
            {
                **extra,
                "response": response,
                "model_name": model_name,
                "term": term,
                "coef": float(result.fe_params[term]),
                "se": float(result.bse_fe[term]),
                "z": float(result.fe_params[term] / result.bse_fe[term]),
                "p": float(result.pvalues[term]),
                "ci_lower": float(conf.loc[term, 0]),
                "ci_upper": float(conf.loc[term, 1]),
            }
        )
    return rows


def compare_models(frame: pd.DataFrame, response: str, specs: list[tuple[str, str, str | None]], extra: dict[str, Any] | None = None):
    results = []
    comparison_rows = []
    extra = extra or {}
    previous_success = None
    for name, formula_template, re_formula in specs:
        formula = formula_template.format(response=response)
        print(f"fit: {extra.get('neural_model', 'acoustic')} {response} {name}")
        try:
            result = fit_mixedlm(formula, frame, "speaker_id", re_formula, reml=False)
        except Exception as exc:
            print(f"fit failed: {extra.get('neural_model', 'acoustic')} {response} {name}: {exc}")
            row = {
                **extra,
                "response": response,
                "model_name": name,
                "loglik": np.nan,
                "AIC": np.nan,
                "BIC": np.nan,
                "df": np.nan,
                "LRT_stat": np.nan,
                "LRT_p": np.nan,
            }
            comparison_rows.append(row)
            continue

        results.append((name, formula, re_formula, result))
        row = {
            **extra,
            "response": response,
            "model_name": name,
            "loglik": float(result.llf),
            "AIC": float(result.aic),
            "BIC": float(result.bic),
            "df": int(len(result.params)),
            "LRT_stat": np.nan,
            "LRT_p": np.nan,
        }
        if previous_success is not None:
            stat, p_value, _ = lrt(previous_success[3], result)
            row["LRT_stat"] = stat
            row["LRT_p"] = p_value
        comparison_rows.append(row)
        previous_success = (name, formula, re_formula, result)
    return results, pd.DataFrame(comparison_rows)


def refit_selected(frame: pd.DataFrame, response: str, selected: tuple[str, str, str | None, Any]):
    name, formula, re_formula, _ = selected
    print(f"refit REML: {response} {name}")
    result = fit_mixedlm(formula, frame, "speaker_id", re_formula, reml=True)
    return name, result


def nakagawa_r2(result) -> tuple[float, float]:
    exog = np.asarray(result.model.exog)
    beta = np.asarray(result.fe_params)
    fixed_part = exog @ beta
    var_fixed = float(np.var(fixed_part, ddof=1))
    var_random = random_variance(result)
    var_residual = float(result.scale)
    denom = var_fixed + var_random + var_residual
    return var_fixed / denom, (var_fixed + var_random) / denom


def run_acoustic_lme(acoustic: pd.DataFrame, written: list[tuple[Path, Any]], table_dir: Path):
    compare_path = table_dir / "lme_acoustic_model_comparison.csv"
    fixed_path = table_dir / "lme_acoustic_fixed_effects.csv"
    icc_path = table_dir / "lme_acoustic_icc.csv"
    r2_path = table_dir / "lme_r2.csv"

    if all(path.exists() for path in [compare_path, fixed_path, icc_path]):
        for path in [compare_path, fixed_path, icc_path]:
            print(f"exists: {path}")
        return None, None

    acoustic = acoustic.copy()
    acoustic["vowel_height"] = acoustic["phoneme"].map(height_map(sorted(acoustic["phoneme"].unique()))).astype(int)

    comparison_frames = []
    fixed_rows = []
    icc_rows = []
    selected_results = {}
    r2_rows = []

    for response in ["F1_norm", "F2_norm"]:
        results, comparison = compare_models(acoustic, response, model_specs_acoustic())
        comparison_frames.append(comparison)
        if not results:
            print(f"no successful acoustic fits for {response}")
            continue
        icc_rows.append(icc_row(results[0][3], response))
        finite_results = [item for item in results if np.isfinite(item[3].bic)]
        selected = min(finite_results or results, key=lambda item: item[3].bic if np.isfinite(item[3].bic) else np.inf)
        selected_results[response] = selected
        model_name, reml_result = refit_selected(acoustic, response, selected)
        fixed_rows.extend(fixed_effect_rows(reml_result, response, model_name))
        marginal, conditional = nakagawa_r2(reml_result)
        r2_rows.append(
            {
                "representation": "acoustic",
                "response": response,
                "marginal_r2": marginal,
                "conditional_r2": conditional,
            }
        )

    save_csv(compare_path, pd.concat(comparison_frames, ignore_index=True), written)
    save_csv(fixed_path, pd.DataFrame(fixed_rows), written)
    save_csv(icc_path, pd.DataFrame(icc_rows), written)
    return selected_results, r2_rows


def run_neural_lme(params: dict[str, Any], acoustic: pd.DataFrame, vowels: list[str], written: list[tuple[Path, Any]], table_dir: Path):
    compare_path = table_dir / "lme_neural_model_comparison.csv"
    fixed_path = table_dir / "lme_neural_fixed_effects.csv"
    icc_path = table_dir / "lme_neural_icc.csv"

    if all(path.exists() for path in [compare_path, fixed_path, icc_path]):
        for path in [compare_path, fixed_path, icc_path]:
            print(f"exists: {path}")
        return []

    comparison_rows = []
    fixed_rows = []
    icc_rows = []
    r2_rows = []

    for neural_model, layer in params["tests"]["representative_layers"].items():
        frame = load_neural_frame(params, acoustic, vowels, neural_model, int(layer))
        for pc_index in range(1, 6):
            response = f"PC{pc_index}"
            extra = {"neural_model": neural_model, "pc": pc_index}
            results, comparison = compare_models(frame, response, model_specs_neural(), extra=extra)
            comparison_rows.extend(comparison.to_dict("records"))
            if not results:
                print(f"no successful neural fits for {neural_model} {response}")
                continue
            icc = icc_row(results[0][3], response)
            icc_rows.append({"neural_model": neural_model, "pc": pc_index, **{k: v for k, v in icc.items() if k != "response"}})
            finite_results = [item for item in results if np.isfinite(item[3].bic)]
            selected = min(finite_results or results, key=lambda item: item[3].bic if np.isfinite(item[3].bic) else np.inf)
            model_name, reml_result = refit_selected(frame, response, selected)
            fixed_rows.extend(
                fixed_effect_rows(
                    reml_result,
                    response,
                    model_name,
                    extra={"neural_model": neural_model, "pc": pc_index},
                )
            )
            main_effects_result = next(item[3] for item in results if item[0] == "main_effects")
            marginal, conditional = nakagawa_r2(main_effects_result)
            r2_rows.append(
                {
                    "representation": neural_model,
                    "response": response,
                    "marginal_r2": marginal,
                    "conditional_r2": conditional,
                }
            )

    save_csv(compare_path, pd.DataFrame(comparison_rows), written)
    save_csv(fixed_path, pd.DataFrame(fixed_rows), written)
    save_csv(icc_path, pd.DataFrame(icc_rows), written)
    return r2_rows


def acoustic_per_vowel_lme(acoustic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phoneme in sorted(acoustic["phoneme"].unique()):
        sub = acoustic[acoustic["phoneme"] == phoneme].copy()
        for response in ["F1_norm", "F2_norm"]:
            print(f"per-vowel acoustic LME: /{phoneme}/ {response}")
            result = fit_mixedlm(f"{response} ~ l1_num + gender_num", sub, "speaker_id", "1", reml=False)
            conf = result.conf_int()
            rows.append(
                {
                    "phoneme": phoneme,
                    "response": response,
                    "term": "l1_num",
                    "coef": float(result.fe_params["l1_num"]),
                    "se": float(result.bse_fe["l1_num"]),
                    "z": float(result.fe_params["l1_num"] / result.bse_fe["l1_num"]),
                    "p": float(result.pvalues["l1_num"]),
                    "ci_lower": float(conf.loc["l1_num", 0]),
                    "ci_upper": float(conf.loc["l1_num", 1]),
                }
            )
    return pd.DataFrame(rows)


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(cosine_distances(left.reshape(1, -1), right.reshape(1, -1))[0, 0])


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


def bootstrap_neural_l1l2_ci(params: dict[str, Any], vowels: list[str], n_resamples: int, random_state: int) -> pd.DataFrame:
    rows = []
    for model, layer in params["tests"]["representative_layers"].items():
        vectors, meta = load_embedding_view(params, model, int(layer))
        mask = meta["phoneme"].isin(vowels).to_numpy()
        vectors = vectors[mask]
        meta = meta.loc[mask].reset_index(drop=True)
        speakers = np.sort(meta["speaker_id"].unique())
        for phoneme in vowels:
            print(f"neural bootstrap CI: {model} /{phoneme}/")
            sub = meta[meta["phoneme"] == phoneme].reset_index(drop=True)
            sub_vectors = vectors[meta["phoneme"] == phoneme]
            observed = cosine_distance(
                sub_vectors[sub["l1_status"] == "fr"].mean(axis=0),
                sub_vectors[sub["l1_status"] == "ru"].mean(axis=0),
            )
            values = []
            rng = np.random.default_rng(random_state + int(layer) + sum(ord(ch) for ch in phoneme))
            step = max(1, n_resamples // 10)
            while len(values) < n_resamples:
                sample_meta, sample_vectors = bootstrap_sample_vectors(sub, sub_vectors, speakers, rng)
                if not (sample_meta["l1_status"] == "fr").any() or not (sample_meta["l1_status"] == "ru").any():
                    continue
                left = sample_vectors[sample_meta["l1_status"] == "fr"].mean(axis=0)
                right = sample_vectors[sample_meta["l1_status"] == "ru"].mean(axis=0)
                values.append(cosine_distance(left, right))
                if len(values) % step == 0 or len(values) == n_resamples:
                    print(f"neural bootstrap CI: {model} /{phoneme}/ {len(values)}/{n_resamples}")
            lower, upper = np.percentile(values, [2.5, 97.5])
            rows.append(
                {
                    "model": model,
                    "phoneme": phoneme,
                    "observed": observed,
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                }
            )
    return pd.DataFrame(rows)


def mean_intra_speaker_cosine(params: dict[str, Any], vowels: list[str]) -> float:
    values = []
    for model, layer in params["tests"]["representative_layers"].items():
        vectors, meta = load_embedding_view(params, model, int(layer))
        mask = meta["phoneme"].isin(vowels).to_numpy()
        vectors = vectors[mask]
        meta = meta.loc[mask].reset_index(drop=True)
        for (speaker, phoneme), sub in meta.groupby(["speaker_id", "phoneme"]):
            idx = sub.index.to_numpy()
            if len(idx) < 2:
                continue
            block = vectors[idx]
            sims = cosine_distances(block, block)
            tri = np.triu_indices_from(sims, k=1)
            values.extend(sims[tri].tolist())
    return float(np.mean(values))


def plot_forest_acoustic(frame: pd.DataFrame, path: Path, written: list[tuple[Path, Any]]) -> None:
    if output_exists(path):
        return
    responses = ["F1_norm", "F2_norm"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, response in zip(axes, responses):
        sub = frame[frame["response"] == response].sort_values("phoneme")
        y = np.arange(len(sub))
        ax.axvspan(-0.3, 0.3, color="grey", alpha=0.2)
        ax.errorbar(
            sub["coef"],
            y,
            xerr=[sub["coef"] - sub["ci_lower"], sub["ci_upper"] - sub["coef"]],
            fmt="o",
            color="#4C78A8",
        )
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(response)
        ax.set_xlabel("L1/L2 contrast")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["phoneme"])
    fig.suptitle("Acoustic L1/L2 contrasts with 95% CIs")
    save_figure(path, fig, written)


def plot_forest_neural(frame: pd.DataFrame, delta0: float, path: Path, written: list[tuple[Path, Any]]) -> None:
    if output_exists(path):
        return
    phonemes = sorted(frame["phoneme"].unique())
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvspan(0, delta0, color="grey", alpha=0.2)
    offsets = {"whisper": -0.12, "xlsr": 0.12}
    colors = {"whisper": "#4C78A8", "xlsr": "#F58518"}
    for model in ["whisper", "xlsr"]:
        sub = frame[frame["model"] == model].sort_values("phoneme")
        y = np.arange(len(sub)) + offsets[model]
        ax.errorbar(
            sub["observed"],
            y,
            xerr=[sub["observed"] - sub["ci_lower"], sub["ci_upper"] - sub["observed"]],
            fmt="o",
            color=colors[model],
            label=model,
        )
    ax.set_yticks(np.arange(len(phonemes)))
    ax.set_yticklabels(phonemes)
    ax.set_xlabel("Cosine distance")
    ax.set_title("Neural L1/L2 contrasts with 95% CIs")
    ax.legend()
    save_figure(path, fig, written)


def rope_classification(acoustic_per_vowel: pd.DataFrame, neural_ci: pd.DataFrame, delta0: float) -> pd.DataFrame:
    rows = []
    for _, row in acoustic_per_vowel.iterrows():
        lower = -0.3
        upper = 0.3
        if row["ci_lower"] >= lower and row["ci_upper"] <= upper:
            cls = "equivalent"
        elif row["ci_upper"] < lower or row["ci_lower"] > upper:
            cls = "non_equivalent"
        else:
            cls = "indeterminate"
        rows.append(
            {
                "phoneme": row["phoneme"],
                "representation": row["response"],
                "point_estimate": row["coef"],
                "ci_lower": row["ci_lower"],
                "ci_upper": row["ci_upper"],
                "rope_lower": lower,
                "rope_upper": upper,
                "classification": cls,
            }
        )
    for _, row in neural_ci.iterrows():
        lower = 0.0
        upper = delta0
        if row["ci_lower"] >= lower and row["ci_upper"] <= upper:
            cls = "equivalent"
        elif row["ci_lower"] > upper or row["ci_upper"] < lower:
            cls = "non_equivalent"
        else:
            cls = "indeterminate"
        rows.append(
            {
                "phoneme": row["phoneme"],
                "representation": row["model"],
                "point_estimate": row["observed"],
                "ci_lower": row["ci_lower"],
                "ci_upper": row["ci_upper"],
                "rope_lower": lower,
                "rope_upper": upper,
                "classification": cls,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    raw_params = load_params(Path("params.yaml"))
    params = resolve_value(raw_params, raw_params)
    requested_vowels = params["tests"]["vowels"]
    n_resamples = int(params["bootstrap"]["n_resamples"])
    random_state = int(params["bootstrap"]["random_state"])

    figure_dir = Path(params["paths"]["figures_dir"])
    table_dir = Path(params["paths"]["tables_dir"])
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[Path, Any]] = []

    acoustic, vowels = build_acoustic_frame(params, requested_vowels)

    print("Section 7.1: acoustic LME")
    selected_results, acoustic_r2_rows = run_acoustic_lme(acoustic, written, table_dir)

    print("Section 7.2: neural LME")
    neural_r2_rows = run_neural_lme(params, acoustic, vowels, written, table_dir)

    r2_path = table_dir / "lme_r2.csv"
    if not output_exists(r2_path):
        save_csv(r2_path, pd.DataFrame((acoustic_r2_rows or []) + neural_r2_rows), written)

    print("Section 8.1: acoustic per-vowel LMEs")
    acoustic_vowel_path = table_dir / "lme_acoustic_per_vowel.csv"
    if acoustic_vowel_path.exists():
        print(f"exists: {acoustic_vowel_path}")
        acoustic_per_vowel = pd.read_csv(acoustic_vowel_path)
    else:
        acoustic_per_vowel = acoustic_per_vowel_lme(acoustic)
        save_csv(acoustic_vowel_path, acoustic_per_vowel, written)

    print("Section 8.2: neural bootstrap CIs")
    neural_ci_path = table_dir / "neural_l1l2_bootstrap_ci.csv"
    if neural_ci_path.exists():
        print(f"exists: {neural_ci_path}")
        neural_ci = pd.read_csv(neural_ci_path)
    else:
        neural_ci = bootstrap_neural_l1l2_ci(params, vowels, n_resamples, random_state)
        save_csv(neural_ci_path, neural_ci, written)

    print("Section 8.3: ROPE thresholds")
    delta0 = mean_intra_speaker_cosine(params, vowels)

    print("Section 8.4: forest plots and ROPE classification")
    plot_forest_acoustic(acoustic_per_vowel, figure_dir / "forest_acoustic.png", written)
    plot_forest_neural(neural_ci, delta0, figure_dir / "forest_neural.png", written)

    rope_path = table_dir / "rope_classification.csv"
    if not output_exists(rope_path):
        save_csv(rope_path, rope_classification(acoustic_per_vowel, neural_ci, delta0), written)

    print("Summary:")
    for path, shape in written:
        print(f"{path}: {shape}")


if __name__ == "__main__":
    main()
