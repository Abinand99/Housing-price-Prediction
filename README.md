# Predicting Housing Affordability in the U.S. (Kaggle)

Predict `AffordabilityPercentageTrue`, the share of households spending ≤30% of income on
housing, for each ZIP in July 2026. Training data covers May and June.

- `housing_affordability.ipynb`: Kaggle notebook. Attach the competition data and run all cells.
- `housing_affordability.py`: the same code as a script (`DATA_DIR=<folder> python housing_affordability.py`).

The competition data is **not** in this repo: the rules forbid redistributing it.

## What we found in the data

| Fact | Why it matters |
|---|---|
| All 6,830 training ZIPs appear in both May and June. In test, 6,830 ZIPs are seen and 1,704 are new. | Two separate problems. |
| The May→June target correlation is 0.999, and "June = May" has RMSE 1.25. | Seen ZIPs are nearly solved by copying the last value. |
| The target drifts down about 0.33 per month, and the amount differs by state (NY −0.63, FL +0.25). | Add a state drift. Tested May→June: RMSE falls from 1.25 to 1.17. |
| The target is strongly geographic: CA ≈ 9%, WA ≈ 20%, TX ≈ 62%, OH ≈ 69%. | Nearby ZIPs' known values are the best predictors for new ZIPs. |
| Target changes are **not** related to rent changes (corr ≈ 0). | The target isn't a simple rent formula, so feature-only models plateau. |
| Income never changes between months. | It's a fixed ZIP attribute. |
| July features jumped for the same ZIPs: senior% +0.37 std, adults +0.29 std, young adults +0.32 std. | We also train on July features so the model matches the new ZIPs' July rows. |
| `RentIndex` is missing for 8.7% of train rows but only 9 test rows. | Fill it from the same ZIP's July value. |

## Approach

1. **Seen ZIPs:** June value plus the state's average monthly drift.
2. **New ZIPs:** LightGBM and CatBoost trained on:
   - rent vs income: `log(income / (40 × rent))`
   - household-mix ratios
   - each ZIP compared with its area's average
   - neighbour labels: smoothed averages over state, metro, zip3 and city, plus the closest
     1, 3 and 8 ZIP numbers. These are always built without the ZIP's own label.
3. **Validation:** 5-fold over ZIPs. Hidden ZIPs are scored on their July-feature rows,
   which mimics a genuinely new ZIP.

## Experiment log (new-ZIP CV RMSE; lower is better)

| Experiment | RMSE |
|---|---|
| Features only, no geography | 13.61 |
| + raw State / Metro / City / zip3 categories | 12.32 |
| + engineered + "relative to area" features + neighbour labels, no raw high-cardinality categories | 11.99 |
| + `colsample_bytree=0.3` | 11.95 |
| Extra ideas: neighbour feature differences, residual kriging | no gain |
| LightGBM alone / CatBoost alone | 11.95 / 12.02 |
| LightGBM + CatBoost blend (0.6 / 0.4) | 11.87 |
| + zip4 average + look-alike neighbours (blend) | 11.85 |
| **+ extra_trees, train on June+July rows only (blend 0.75/0.25)** | **11.77** (67% within ±10) |

## Ideas not tried yet

- More seeds and folds, and Optuna tuning of LightGBM and CatBoost.
- A level-aware drift for seen ZIPs (state × target level). It gave a small May→June gain: 1.166 vs 1.169.
- Use the public leaderboard sparingly, e.g. one submission with and one without the seen-ZIP drift.
  Only 30% of test is public, so don't chase it.
