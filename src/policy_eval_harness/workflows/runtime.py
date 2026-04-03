from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

from policy_eval_harness.replay import canonical_json
from policy_eval_harness.replay.runtime import normalize_json_value
from policy_eval_harness.workflows.types import (
    AblationArtifacts,
    AblationManifest,
    AblationMetricConfig,
    LabelCompareArtifacts,
    LabelCompareManifest,
    LabelDatasetConfig,
    LabelEvaluationConfig,
)

LABEL_COMPARE_AUC_FILENAME = "oos_auc_by_label.csv"
LABEL_COMPARE_SUMMARY_FILENAME = "model_label_summary.json"
LABEL_COMPARE_FEATURE_SNAPSHOT_FILENAME = "feature_manifest_snapshot.json"

ABLATION_FACTOR_EFFECTS_FILENAME = "factor_effects.csv"
ABLATION_INTERACTION_SUMMARY_FILENAME = "interaction_summary.csv"
ABLATION_REPORT_FILENAME = "ablation_report.md"

ABLATION_KEYS = ("A0_B0", "A1_B0", "A0_B1", "A1_B1")
SUPPORTED_MODELS = ("random_forest", "gradient_boosting")


def run_label_compare_from_manifest(manifest_path: Path, out_dir: Path) -> LabelCompareArtifacts:
    manifest = load_label_compare_manifest(Path(manifest_path))
    data = _read_table(manifest.dataset.path)
    _require_columns(
        data,
        {
            manifest.dataset.id_column,
            manifest.dataset.split_column,
            *manifest.dataset.feature_columns,
            *manifest.dataset.label_variants,
        },
        "label_compare.dataset",
    )

    data = data.copy()
    data[manifest.dataset.id_column] = data[manifest.dataset.id_column].map(str)
    data[manifest.dataset.split_column] = data[manifest.dataset.split_column].map(str)

    train_mask = data[manifest.dataset.split_column].isin(manifest.evaluation.train_split_values)
    oos_mask = data[manifest.dataset.split_column] == manifest.evaluation.oos_split_value
    if not train_mask.any() or not oos_mask.any():
        raise ValueError("Label comparison requires non-empty train and out-of-sample partitions.")

    x_train = data.loc[train_mask, list(manifest.dataset.feature_columns)].astype(float)
    x_oos = data.loc[oos_mask, list(manifest.dataset.feature_columns)].astype(float)
    train_ids = tuple(data.loc[train_mask, manifest.dataset.id_column].tolist())
    oos_ids = tuple(data.loc[oos_mask, manifest.dataset.id_column].tolist())
    train_id_hash = _stable_id_hash(train_ids)
    oos_id_hash = _stable_id_hash(oos_ids)

    rows: List[Dict[str, Any]] = []
    summary_models: Dict[str, Any] = {}

    for model_name in manifest.models:
        model_rows = []
        for label_name in manifest.dataset.label_variants:
            y_train = data.loc[train_mask, label_name].map(_to_binary_label)
            y_oos = data.loc[oos_mask, label_name].map(_to_binary_label)
            _validate_binary_partition(y_train, f"{label_name} train")
            _validate_binary_partition(y_oos, f"{label_name} holdout")

            model = _build_model(model_name, manifest.evaluation.random_seed)
            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_oos)[:, 1]
            auc = float(roc_auc_score(y_oos, probabilities))
            row = {
                "model_family": model_name,
                "label_variant": label_name,
                "oos_auc": auc,
                "n_train": int(y_train.shape[0]),
                "n_oos": int(y_oos.shape[0]),
                "train_positive_rate": float(y_train.mean()),
                "oos_positive_rate": float(y_oos.mean()),
                "train_id_hash": train_id_hash,
                "oos_id_hash": oos_id_hash,
            }
            rows.append(row)
            model_rows.append(row)

        ordered_model_rows = sorted(model_rows, key=lambda item: (-item["oos_auc"], item["label_variant"]))
        summary_models[model_name] = {
            "best_label_variant": ordered_model_rows[0]["label_variant"],
            "best_oos_auc": ordered_model_rows[0]["oos_auc"],
            "ranking": [
                {
                    "label_variant": item["label_variant"],
                    "oos_auc": item["oos_auc"],
                }
                for item in ordered_model_rows
            ],
        }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    oos_auc_path = out_dir / LABEL_COMPARE_AUC_FILENAME
    summary_path = out_dir / LABEL_COMPARE_SUMMARY_FILENAME
    feature_snapshot_path = out_dir / LABEL_COMPARE_FEATURE_SNAPSHOT_FILENAME

    result_frame = pd.DataFrame(rows).sort_values(["model_family", "label_variant"]).reset_index(drop=True)
    result_frame.to_csv(oos_auc_path, index=False)

    summary_payload = {
        "manifest_file": manifest.manifest_path.name,
        "models": summary_models,
        "train_split_values": list(manifest.evaluation.train_split_values),
        "oos_split_value": manifest.evaluation.oos_split_value,
        "random_seed": manifest.evaluation.random_seed,
    }
    summary_path.write_text(canonical_json(summary_payload) + "\n", encoding="utf-8")

    feature_snapshot = {
        "dataset_file": manifest.dataset.path.name,
        "id_column": manifest.dataset.id_column,
        "split_column": manifest.dataset.split_column,
        "feature_columns": list(manifest.dataset.feature_columns),
        "label_variants": list(manifest.dataset.label_variants),
        "models": list(manifest.models),
        "train_id_hash": train_id_hash,
        "oos_id_hash": oos_id_hash,
    }
    feature_snapshot_path.write_text(canonical_json(feature_snapshot) + "\n", encoding="utf-8")

    return LabelCompareArtifacts(
        oos_auc_path=oos_auc_path,
        summary_path=summary_path,
        feature_manifest_snapshot_path=feature_snapshot_path,
    )


def run_ablation_2x2_from_manifest(manifest_path: Path, out_dir: Path) -> AblationArtifacts:
    manifest = load_ablation_manifest(Path(manifest_path))
    frame = _read_table(manifest.input_path)
    metric_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    if {"case_id", "variant_id", "split"}.issubset(frame.columns):
        source_kind = "panel"
    else:
        source_kind = "scorecard"

    for metric in manifest.metrics:
        values = {
            key: _metric_value_from_input(
                frame=frame,
                source_kind=source_kind,
                split=manifest.split,
                variant_id=variant_id,
                metric_name=metric.name,
            )
            for key, variant_id in manifest.variant_map.items()
        }
        baseline_value = values["A0_B0"]
        a_effect = values["A1_B0"] - baseline_value
        b_effect = values["A0_B1"] - baseline_value
        combined_effect = values["A1_B1"] - baseline_value
        interaction_term = combined_effect - a_effect - b_effect
        oriented_interaction = interaction_term * _goal_sign(metric.goal)
        qualitative_read = _qualitative_read(oriented_interaction, metric.interaction_epsilon)

        metric_rows.append(
            {
                "split": manifest.split,
                "metric": metric.name,
                "metric_group": metric.group,
                "goal": metric.goal,
                "baseline_value": baseline_value,
                "a1_b0_value": values["A1_B0"],
                "a0_b1_value": values["A0_B1"],
                "a1_b1_value": values["A1_B1"],
                "a_effect": a_effect,
                "b_effect": b_effect,
                "combined_effect": combined_effect,
                "interaction_term": interaction_term,
            }
        )
        summary_rows.append(
            {
                "split": manifest.split,
                "metric": metric.name,
                "metric_group": metric.group,
                "goal": metric.goal,
                "interaction_term": interaction_term,
                "oriented_interaction": oriented_interaction,
                "qualitative_read": qualitative_read,
                "interaction_epsilon": metric.interaction_epsilon,
            }
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    factor_effects_path = out_dir / ABLATION_FACTOR_EFFECTS_FILENAME
    interaction_summary_path = out_dir / ABLATION_INTERACTION_SUMMARY_FILENAME
    report_path = out_dir / ABLATION_REPORT_FILENAME

    metric_frame = pd.DataFrame(metric_rows).sort_values(["metric_group", "metric"]).reset_index(drop=True)
    summary_frame = pd.DataFrame(summary_rows).sort_values(["metric_group", "metric"]).reset_index(drop=True)
    metric_frame.to_csv(factor_effects_path, index=False)
    summary_frame.to_csv(interaction_summary_path, index=False)
    report_path.write_text(_build_ablation_report(manifest, metric_frame, summary_frame) + "\n", encoding="utf-8")

    return AblationArtifacts(
        factor_effects_path=factor_effects_path,
        interaction_summary_path=interaction_summary_path,
        report_path=report_path,
    )


def load_label_compare_manifest(manifest_path: Path) -> LabelCompareManifest:
    manifest_path = manifest_path.resolve()
    raw = _load_mapping(manifest_path)
    base_dir = manifest_path.parent
    dataset_raw = _require_mapping(raw, "dataset")
    evaluation_raw = raw.get("evaluation", {})
    if evaluation_raw is None:
        evaluation_raw = {}
    if not isinstance(evaluation_raw, Mapping):
        raise ValueError("Manifest field 'evaluation' must be a mapping.")
    models = tuple(_require_string_list(raw, "models"))
    if not models:
        raise ValueError("Manifest field 'models' must not be empty.")
    unsupported = [model for model in models if model not in SUPPORTED_MODELS]
    if unsupported:
        raise ValueError(f"Unsupported model family: {', '.join(sorted(unsupported))}")

    dataset = LabelDatasetConfig(
        path=_resolve_path(base_dir, _require_string(dataset_raw, "path")),
        id_column=_require_string(dataset_raw, "id_column"),
        split_column=_require_string(dataset_raw, "split_column"),
        feature_columns=tuple(_require_string_list(dataset_raw, "feature_columns")),
        label_variants=tuple(_require_string_list(dataset_raw, "label_variants")),
    )
    if not dataset.feature_columns:
        raise ValueError("Manifest field 'dataset.feature_columns' must not be empty.")
    if not dataset.label_variants:
        raise ValueError("Manifest field 'dataset.label_variants' must not be empty.")

    train_split_values = tuple(_require_string_list(evaluation_raw, "train_split_values")) or ("train",)
    evaluation = LabelEvaluationConfig(
        train_split_values=train_split_values,
        oos_split_value=_optional_string(evaluation_raw, "oos_split_value") or "holdout",
        random_seed=int(evaluation_raw.get("random_seed", 17) or 17),
    )
    return LabelCompareManifest(
        manifest_path=manifest_path,
        dataset=dataset,
        models=models,
        evaluation=evaluation,
    )


def load_ablation_manifest(manifest_path: Path) -> AblationManifest:
    manifest_path = manifest_path.resolve()
    raw = _load_mapping(manifest_path)
    base_dir = manifest_path.parent
    raw_variant_map = _require_mapping(raw, "variant_map")
    variant_map = {key: _require_string(raw_variant_map, key) for key in ABLATION_KEYS}
    metrics_raw = raw.get("metrics", [])
    if not isinstance(metrics_raw, Sequence) or isinstance(metrics_raw, (str, bytes)):
        raise ValueError("Manifest field 'metrics' must be a sequence.")
    metrics: List[AblationMetricConfig] = []
    for index, item in enumerate(metrics_raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"metrics[{index}] must be a mapping.")
        goal = _require_string(item, "goal")
        if goal not in {"maximize", "minimize"}:
            raise ValueError("Ablation metric goal must be 'maximize' or 'minimize'.")
        metrics.append(
            AblationMetricConfig(
                name=_require_string(item, "name"),
                goal=goal,
                group=_require_string(item, "group"),
                interaction_epsilon=float(item.get("interaction_epsilon", 0.01) or 0.01),
            )
        )
    if not metrics:
        raise ValueError("Manifest field 'metrics' must not be empty.")

    bootstrap = raw.get("bootstrap", {})
    if bootstrap is None:
        bootstrap = {}
    if not isinstance(bootstrap, Mapping):
        raise ValueError("Manifest field 'bootstrap' must be a mapping when provided.")

    return AblationManifest(
        manifest_path=manifest_path,
        input_path=_resolve_path(base_dir, _require_string(raw, "input_scorecard_or_panel_path")),
        split=_optional_string(raw, "split") or "holdout",
        variant_map=variant_map,
        metrics=tuple(metrics),
        bootstrap={str(key): normalize_json_value(value) for key, value in bootstrap.items()},
    )


def _metric_value_from_input(
    *,
    frame: pd.DataFrame,
    source_kind: str,
    split: str,
    variant_id: str,
    metric_name: str,
) -> float:
    if source_kind == "scorecard":
        required = {"split", "candidate_variant_id", "metric", "candidate_value"}
        _require_columns(frame, required, "ablation.scorecard")
        rows = frame[
            (frame["split"] == split)
            & (frame["candidate_variant_id"] == variant_id)
            & (frame["metric"] == metric_name)
        ]
        if rows.empty:
            raise ValueError(f"Missing scorecard row for split={split!r}, variant={variant_id!r}, metric={metric_name!r}")
        return float(rows.iloc[0]["candidate_value"])

    variant_frame = frame[(frame["split"] == split) & (frame["variant_id"] == variant_id)]
    if variant_frame.empty:
        raise ValueError(f"Missing panel rows for split={split!r}, variant={variant_id!r}")

    if metric_name == "mean_utility":
        return float(variant_frame["utility"].astype(float).mean())
    if metric_name == "pathology_rate":
        if "pathology" not in variant_frame.columns:
            raise ValueError("Panel input missing 'pathology' for pathology_rate.")
        return float(variant_frame["pathology"].map(_to_bool).mean())
    if metric_name == "error_case_rate":
        if "error_count" in variant_frame.columns:
            return float((variant_frame["error_count"].fillna(0).astype(float) > 0).mean())
        if "error_decision" in variant_frame.columns:
            return float(variant_frame["error_decision"].map(_to_bool).mean())
        raise ValueError("Panel input missing error field for error_case_rate.")
    if metric_name == "mean_decision_count":
        if "decision_count" not in variant_frame.columns:
            raise ValueError("Panel input missing 'decision_count' for mean_decision_count.")
        return float(variant_frame["decision_count"].astype(float).mean())
    if metric_name == "accept_rate":
        if "selected" not in variant_frame.columns:
            raise ValueError("Panel input missing 'selected' for accept_rate.")
        return float(variant_frame["selected"].map(_to_bool).mean())
    if metric_name == "balanced_accuracy":
        if not {"selected", "label"}.issubset(variant_frame.columns):
            raise ValueError("Panel input missing 'selected'/'label' for balanced_accuracy.")
        selected = variant_frame["selected"].map(_to_bool)
        labels = variant_frame["label"].map(_to_bool)
        return float(_balanced_accuracy(selected, labels))
    raise ValueError(f"Unsupported ablation metric: {metric_name!r}")


def _build_model(model_name: str, random_seed: int) -> Any:
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=80,
            max_depth=4,
            min_samples_leaf=1,
            random_state=random_seed,
        )
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_seed)
    raise ValueError(f"Unsupported model family: {model_name!r}")


def _balanced_accuracy(selected: pd.Series, labels: pd.Series) -> float:
    positive_mask = labels.astype(bool)
    negative_mask = ~positive_mask
    tpr = float(selected[positive_mask].mean()) if positive_mask.any() else float("nan")
    tnr = float((~selected[negative_mask]).mean()) if negative_mask.any() else float("nan")
    if np.isnan(tpr) or np.isnan(tnr):
        return float("nan")
    return (tpr + tnr) / 2.0


def _build_ablation_report(
    manifest: AblationManifest,
    metric_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
) -> str:
    lines = [
        "# 2x2 Ablation Report",
        "",
        f"- Split: `{manifest.split}`",
        f"- Input: `{manifest.input_path.name}`",
        "",
        "## Variant Map",
        "",
    ]
    for key in ABLATION_KEYS:
        lines.append(f"- `{key}`: `{manifest.variant_map[key]}`")
    lines.extend(["", "## Interaction Summary", ""])
    for row in summary_frame.to_dict(orient="records"):
        lines.append(
            "- `{metric}` ({group}, {goal}): {read} "
            "(interaction={interaction:.4f}, oriented={oriented:.4f})".format(
                metric=row["metric"],
                group=row["metric_group"],
                goal=row["goal"],
                read=row["qualitative_read"],
                interaction=row["interaction_term"],
                oriented=row["oriented_interaction"],
            )
        )
    lines.extend(["", "## Factor Effects", ""])
    for row in metric_frame.to_dict(orient="records"):
        lines.append(
            "- `{metric}`: A={a:.4f}, B={b:.4f}, combined={combined:.4f}, interaction={interaction:.4f}".format(
                metric=row["metric"],
                a=row["a_effect"],
                b=row["b_effect"],
                combined=row["combined_effect"],
                interaction=row["interaction_term"],
            )
        )
    return "\n".join(lines)


def _qualitative_read(oriented_interaction: float, epsilon: float) -> str:
    if oriented_interaction > epsilon:
        return "synergy"
    if oriented_interaction < -epsilon:
        return "interference"
    return "near_additive"


def _goal_sign(goal: str) -> float:
    return 1.0 if goal == "maximize" else -1.0


def _stable_id_hash(values: Iterable[str]) -> str:
    payload = canonical_json(sorted(str(value) for value in values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_binary_partition(values: pd.Series, label_name: str) -> None:
    observed = sorted({int(value) for value in values.tolist()})
    if observed != [0, 1]:
        raise ValueError(f"{label_name} partition must contain both classes, found {observed!r}.")


def _to_binary_label(value: Any) -> int:
    normalized = _to_bool(value)
    return 1 if normalized else 0


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"Unable to coerce value to bool: {value!r}")


def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def _load_mapping(path: Path) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected mapping in {path}, found {type(data).__name__}.")
    return data


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _require_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Manifest field {key!r} must be a mapping.")
    return value


def _require_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest field {key!r} must be a non-empty string.")
    return value


def _optional_string(mapping: Mapping[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest field {key!r} must be a non-empty string when provided.")
    return value


def _require_string_list(mapping: Mapping[str, Any], key: str) -> List[str]:
    value = mapping.get(key, [])
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"Manifest field {key!r} must be a sequence of strings.")
    result = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Manifest field {key!r} must contain non-empty strings.")
        result.append(item)
    return result


def _require_columns(frame: pd.DataFrame, required_columns: Iterable[str], label: str) -> None:
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")
