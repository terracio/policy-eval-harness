# 2x2 Ablation Example

This example uses a small comparison panel with four variants:

- `approval_base`: `A0_B0`
- `approval_guarded`: `A1_B0`
- `approval_prioritized`: `A0_B1`
- `approval_guarded_prioritized`: `A1_B1`

Run it with:

```bash
policy-eval workflow ablation-2x2 run \
  --manifest examples/ablation_2x2/ablation_2x2.yaml \
  --out-dir ./artifacts/ablation_2x2
```

The expected pattern is a negative interaction on holdout utility: the combined variant helps, but less than the sum of the two standalone factors.
