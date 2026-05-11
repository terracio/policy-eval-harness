from __future__ import annotations

COMPARISON_PANEL_FILENAME = "comparison_panel.parquet"
SCORECARD_FILENAME = "scorecard.csv"
PROMOTION_DECISIONS_FILENAME = "promotion_decisions.json"

PROFILE_REPLAY = "replay_outcomes_v1"
PROFILE_SELECTION = "selection_panel_v1"
SEQUENTIAL_GATES = "sequential_promotion_v1"
SELECTION_GATES = "selection_promotion_v1"
SPLIT_HOLDOUT = "holdout"
SPLIT_DEV = "dev"

REPLAY_METRICS = (
    "mean_utility",
    "median_utility",
    "paired_coverage_rate",
    "terminal_rate",
    "pathology_rate",
    "error_case_rate",
    "mean_decision_count",
)

SELECTION_METRICS = (
    "balanced_accuracy",
    "tpr",
    "tnr",
    "mean_utility",
    "accept_rate",
    "mean_utility_selected",
    "parse_fail_rate",
    "invalid_output_rate",
    "error_decision_rate",
)

DEFAULT_SEQUENTIAL_THRESHOLDS = {
    "min_delta_mean_utility": 0.0,
    "min_paired_coverage_rate": 1.0,
    "max_candidate_pathology_rate": 1.0,
    "max_candidate_error_case_rate": 1.0,
}

DEFAULT_SELECTION_THRESHOLDS = {
    "min_delta_balanced_accuracy_vs_chance": 0.0,
    "min_delta_mean_utility_vs_accept_all": 0.0,
    "min_delta_mean_utility_vs_random_rate_matched": 0.0,
    "min_paired_coverage_rate": 1.0,
    "min_accept_rate": 0.0,
    "max_accept_rate": 1.0,
    "max_parse_fail_rate": 1.0,
    "max_invalid_output_rate": 1.0,
    "max_error_decision_rate": 1.0,
}

