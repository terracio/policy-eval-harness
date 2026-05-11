from __future__ import annotations

LABEL_COMPARE_AUC_FILENAME = "oos_auc_by_label.csv"
LABEL_COMPARE_SUMMARY_FILENAME = "model_label_summary.json"
LABEL_COMPARE_FEATURE_SNAPSHOT_FILENAME = "feature_manifest_snapshot.json"

ABLATION_FACTOR_EFFECTS_FILENAME = "factor_effects.csv"
ABLATION_INTERACTION_SUMMARY_FILENAME = "interaction_summary.csv"
ABLATION_REPORT_FILENAME = "ablation_report.md"

ABLATION_KEYS = ("A0_B0", "A1_B0", "A0_B1", "A1_B1")
SUPPORTED_MODELS = ("random_forest", "gradient_boosting")
