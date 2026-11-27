# Changes

## 27 Nov 2026 - decision thresholds chosen on training data, not on the test set

**Problem.** The cost-minimising threshold was searched on the test set and the resulting recall and approval rates were reported on that same test set, so the threshold table was a best case picked with hindsight.

**Fix.** Thresholds are now chosen on 5-fold out-of-fold predictions from the training set and then applied once to the untouched test set. The final model and the train/test split are unchanged, so ROC-AUC, PR-AUC, KS, calibration and SHAP results are identical. The hindsight-optimal test threshold is printed for comparison only. Headline metrics are now written to `results.json`.

| Cost ratio (missed default : bad decline) | Before: threshold / caught / approved | After: threshold / caught / approved |
|---|---|---|
| 2:1 | 0.65 / 51% / 81% | 0.62 / 53% / 79% |
| 5:1 | 0.41 / 72% / 59% | 0.38 / 75% / 53% |
| 10:1 | 0.22 / 92% / 26% | 0.20 / 94% / 22% |

Test-set cost at the new thresholds is within about 1% of the hindsight optimum (2,273 vs 2,260; 4,289 vs 4,255; 5,362 vs 5,311).

**Unchanged:** XGBoost ROC-AUC 0.775, PR-AUC 0.558, KS 0.43; logistic ROC-AUC 0.748; default rate among approved at the safest 50% / 70% / 90%: 9.7% / 11.8% / 16.9%.

**Wording:** "cuts the default rate from 22% to under 10%" now states the approval rate it holds at (the safest 50%).
