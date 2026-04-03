# 2x2 Ablation Report

- Split: `holdout`
- Input: `comparison_panel.csv`

## Variant Map

- `A0_B0`: `approval_base`
- `A1_B0`: `approval_guarded`
- `A0_B1`: `approval_prioritized`
- `A1_B1`: `approval_guarded_prioritized`

## Interaction Summary

- `error_case_rate` (reliability, minimize): interference (interaction=0.2500, oriented=-0.2500)
- `pathology_rate` (stability, minimize): interference (interaction=0.5000, oriented=-0.5000)
- `mean_utility` (utility, maximize): interference (interaction=-0.1475, oriented=-0.1475)

## Factor Effects

- `error_case_rate`: A=0.0000, B=0.0000, combined=0.2500, interaction=0.2500
- `pathology_rate`: A=-0.2500, B=-0.2500, combined=0.0000, interaction=0.5000
- `mean_utility`: A=0.1550, B=0.0875, combined=0.0950, interaction=-0.1475
