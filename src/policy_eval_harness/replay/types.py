from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Protocol, Tuple, Union

JSONScalar = Union[None, bool, int, float, str]
JSONValue = Union[JSONScalar, List["JSONValue"], Dict[str, "JSONValue"]]


@dataclass(frozen=True)
class ReplayCase:
    case_id: str
    start_ts_utc: Optional[str]
    metadata: Dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayStep:
    case_id: str
    step_index: int
    ts_utc: str
    observation: Dict[str, JSONValue] = field(default_factory=dict)
    metadata: Dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayStepInput:
    variant_id: str
    case: ReplayCase
    step: ReplayStep
    state: JSONValue


@dataclass(frozen=True)
class ReplayTransition:
    next_state: JSONValue = None
    terminated: bool = False
    termination_reason: Optional[str] = None
    final_status: Optional[str] = None
    metrics: Dict[str, JSONValue] = field(default_factory=dict)
    trace: Dict[str, JSONValue] = field(default_factory=dict)


class PolicyCallable(Protocol):
    def __call__(
        self,
        step_input: ReplayStepInput,
        params: Optional[Mapping[str, JSONValue]] = None,
    ) -> Optional[Mapping[str, JSONValue]]:
        ...


class ExecutorCallable(Protocol):
    def __call__(
        self,
        step_input: ReplayStepInput,
        action: Optional[Mapping[str, JSONValue]],
        params: Optional[Mapping[str, JSONValue]] = None,
    ) -> ReplayTransition:
        ...


@dataclass(frozen=True)
class ReplayUniverse:
    cases: Tuple[ReplayCase, ...]
    steps_by_case: Mapping[str, Tuple[ReplayStep, ...]]

    @property
    def case_count(self) -> int:
        return len(self.cases)

    @property
    def step_count(self) -> int:
        return sum(len(steps) for steps in self.steps_by_case.values())


@dataclass(frozen=True)
class UniverseConfig:
    cases_path: Path
    steps_path: Path


@dataclass(frozen=True)
class ExecutorConfig:
    import_path: str
    params: Dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class VariantConfig:
    variant_id: str
    policy_import_path: str
    policy_params: Dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputConfig:
    episode_summary: str = "episode_summary.csv"
    step_trace: str = "step_trace.parquet"
    run_metadata: str = "run_metadata.json"


@dataclass(frozen=True)
class ReplayManifest:
    manifest_path: Path
    manifest_hash: str
    universe: UniverseConfig
    executor: ExecutorConfig
    variants: Tuple[VariantConfig, ...]
    run_variant_ids: Tuple[str, ...]
    output: OutputConfig = OutputConfig()


@dataclass(frozen=True)
class ReplayArtifacts:
    episode_summary_path: Path
    step_trace_path: Path
    run_metadata_path: Path
