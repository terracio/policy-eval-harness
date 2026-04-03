from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class SplitConfig:
    mode: str = "existing"
    field: Optional[str] = None
    holdout_cutoff_utc: Optional[str] = None
    holdout_pct: Optional[float] = None


@dataclass(frozen=True)
class GateConfig:
    profile_id: str
    thresholds: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BootstrapConfig:
    enabled: bool = False
    n_samples: int = 0
    seed: int = 0


@dataclass(frozen=True)
class EvaluationManifest:
    manifest_path: Path
    manifest_hash: str
    profile_id: str
    inputs: Dict[str, Any]
    baseline_variant_id: str
    candidate_variant_ids: Tuple[str, ...]
    splits: SplitConfig
    gates: GateConfig
    bootstrap: BootstrapConfig


@dataclass(frozen=True)
class EvaluationArtifacts:
    comparison_panel_path: Path
    scorecard_path: Path
    promotion_decisions_path: Path
