from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class LabelDatasetConfig:
    path: Path
    id_column: str
    split_column: str
    feature_columns: Tuple[str, ...]
    label_variants: Tuple[str, ...]


@dataclass(frozen=True)
class LabelEvaluationConfig:
    train_split_values: Tuple[str, ...]
    oos_split_value: str
    random_seed: int = 0


@dataclass(frozen=True)
class LabelCompareManifest:
    manifest_path: Path
    dataset: LabelDatasetConfig
    models: Tuple[str, ...]
    evaluation: LabelEvaluationConfig


@dataclass(frozen=True)
class LabelCompareArtifacts:
    oos_auc_path: Path
    summary_path: Path
    feature_manifest_snapshot_path: Path


@dataclass(frozen=True)
class AblationMetricConfig:
    name: str
    goal: str
    group: str
    interaction_epsilon: float = 0.01


@dataclass(frozen=True)
class AblationManifest:
    manifest_path: Path
    input_path: Path
    split: str
    variant_map: Dict[str, str]
    metrics: Tuple[AblationMetricConfig, ...]
    bootstrap: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AblationArtifacts:
    factor_effects_path: Path
    interaction_summary_path: Path
    report_path: Path
