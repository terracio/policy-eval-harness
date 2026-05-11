from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

from policy_eval_harness._utils.json import canonical_json
from policy_eval_harness._utils.paths import resolve_path
from policy_eval_harness.io.tables import read_table
from policy_eval_harness.workflows.common import (
    load_mapping,
    require_columns,
    require_mapping,
    require_string,
    require_string_list,
    optional_string,
    to_bool,
)
from policy_eval_harness.workflows.constants import (
    LABEL_COMPARE_AUC_FILENAME,
    LABEL_COMPARE_FEATURE_SNAPSHOT_FILENAME,
    LABEL_COMPARE_SUMMARY_FILENAME,
    SUPPORTED_MODELS,
)
from policy_eval_harness.workflows.types import (
    LabelCompareArtifacts,
    LabelCompareManifest,
    LabelDatasetConfig,
    LabelEvaluationConfig,
)


def run_label_compare_from_manifest(manifest_path: Path, out_dir: Path) -> LabelCompareArtifacts:
    manifest = load_label_compare_manifest(Path(manifest_path))
    data = read_table(manifest.dataset.path)
    require_columns(
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

    rows: list[Dict[str, Any]] = []
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

    pd.DataFrame(rows).sort_values(["model_family", "label_variant"]).reset_index(drop=True).to_csv(oos_auc_path, index=False)
    summary_path.write_text(canonical_json(_summary_payload(manifest, summary_models)) + "\n", encoding="utf-8")
    feature_snapshot_path.write_text(
        canonical_json(_feature_snapshot(manifest, train_id_hash, oos_id_hash)) + "\n",
        encoding="utf-8",
    )

    return LabelCompareArtifacts(
        oos_auc_path=oos_auc_path,
        summary_path=summary_path,
        feature_manifest_snapshot_path=feature_snapshot_path,
    )


def load_label_compare_manifest(manifest_path: Path) -> LabelCompareManifest:
    manifest_path = manifest_path.resolve()
    raw = load_mapping(manifest_path)
    base_dir = manifest_path.parent
    dataset_raw = require_mapping(raw, "dataset")
    evaluation_raw = raw.get("evaluation", {})
    if evaluation_raw is None:
        evaluation_raw = {}
    if not isinstance(evaluation_raw, Mapping):
        raise ValueError("Manifest field 'evaluation' must be a mapping.")
    models = tuple(require_string_list(raw, "models"))
    if not models:
        raise ValueError("Manifest field 'models' must not be empty.")
    unsupported = [model for model in models if model not in SUPPORTED_MODELS]
    if unsupported:
        raise ValueError(f"Unsupported model family: {', '.join(sorted(unsupported))}")

    dataset = LabelDatasetConfig(
        path=resolve_path(base_dir, require_string(dataset_raw, "path")),
        id_column=require_string(dataset_raw, "id_column"),
        split_column=require_string(dataset_raw, "split_column"),
        feature_columns=tuple(require_string_list(dataset_raw, "feature_columns")),
        label_variants=tuple(require_string_list(dataset_raw, "label_variants")),
    )
    if not dataset.feature_columns:
        raise ValueError("Manifest field 'dataset.feature_columns' must not be empty.")
    if not dataset.label_variants:
        raise ValueError("Manifest field 'dataset.label_variants' must not be empty.")

    train_split_values = tuple(require_string_list(evaluation_raw, "train_split_values")) or ("train",)
    evaluation = LabelEvaluationConfig(
        train_split_values=train_split_values,
        oos_split_value=optional_string(evaluation_raw, "oos_split_value") or "holdout",
        random_seed=int(evaluation_raw.get("random_seed", 17) or 17),
    )
    return LabelCompareManifest(
        manifest_path=manifest_path,
        dataset=dataset,
        models=models,
        evaluation=evaluation,
    )


def _summary_payload(manifest: LabelCompareManifest, summary_models: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "manifest_file": manifest.manifest_path.name,
        "models": summary_models,
        "train_split_values": list(manifest.evaluation.train_split_values),
        "oos_split_value": manifest.evaluation.oos_split_value,
        "random_seed": manifest.evaluation.random_seed,
    }


def _feature_snapshot(manifest: LabelCompareManifest, train_id_hash: str, oos_id_hash: str) -> Dict[str, Any]:
    return {
        "dataset_file": manifest.dataset.path.name,
        "id_column": manifest.dataset.id_column,
        "split_column": manifest.dataset.split_column,
        "feature_columns": list(manifest.dataset.feature_columns),
        "label_variants": list(manifest.dataset.label_variants),
        "models": list(manifest.models),
        "train_id_hash": train_id_hash,
        "oos_id_hash": oos_id_hash,
    }


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


def _stable_id_hash(values: Iterable[str]) -> str:
    payload = canonical_json(sorted(str(value) for value in values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_binary_partition(values: pd.Series, label_name: str) -> None:
    observed = sorted({int(value) for value in values.tolist()})
    if observed != [0, 1]:
        raise ValueError(f"{label_name} partition must contain both classes, found {observed!r}.")


def _to_binary_label(value: Any) -> int:
    normalized = to_bool(value)
    return 1 if normalized else 0
