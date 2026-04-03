# Label Comparison Example

This example compares two label schemes on the same fixed tabular dataset:

- `baseline_label`: a noisier target with hidden flips that the features cannot explain cleanly
- `clean_label`: a more coherent target built from the same underlying cases

Run it with:

```bash
policy-eval workflow label-compare run \
  --manifest examples/label_compare/label_compare.yaml \
  --out-dir ./artifacts/label_compare
```

The expected pattern is that `clean_label` ranks above `baseline_label` for both bundled model families on out-of-sample AUC.
