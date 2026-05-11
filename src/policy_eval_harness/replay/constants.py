from __future__ import annotations

SUMMARY_COLUMNS = [
    "case_id",
    "variant_id",
    "start_ts_utc",
    "end_ts_utc",
    "n_steps",
    "terminated",
    "termination_reason",
    "final_status",
    "decision_count",
    "error_count",
    "summary_metrics_json",
]

TRACE_COLUMNS = [
    "case_id",
    "variant_id",
    "step_index",
    "ts_utc",
    "observation_json",
    "action_json",
    "state_before_json",
    "state_after_json",
    "transition_metrics_json",
    "trace_json",
    "error",
]

CASE_RESERVED_COLUMNS = {"case_id", "start_ts_utc"}
STEP_RESERVED_COLUMNS = {"case_id", "step_index", "ts_utc"}

