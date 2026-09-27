# Predicting Housing Affordability in the U.S. (Kaggle)

Predict `AffordabilityPercentageTrue` (share of households spending ≤30% of income on
housing) per ZIP code for July 2026. Training data covers May and June.

- `housing_affordability.ipynb`: Kaggle notebook. Upload it and attach the competition data.
- `housing_affordability.py`: the same code as a script (`DATA_DIR=... python housing_affordability.py`).

## How the score works

- **Seen ZIPs** are July rows for ZIPs that appear in train. The model can use each ZIP's own history.
- **New ZIPs** are July rows for ZIPs held out of train entirely. They get **more weight** in
  the final score.
- The public leaderboard uses only 30% of test, so trust local CV over leaderboard moves.

## Strategy

1. **Validate like the test.**
   - New ZIPs: `GroupKFold` by ZIP. A random row split leaks the same ZIP's other month
     into validation.
   - Seen ZIPs: predict one month from the other.
   - Combine the two RMSEs with the competition weighting.
2. **Physics-style features.** The target is roughly the CDF of income vs annual rent:
   `Φ(log(0.3·income / (12·rent)) / σ)`. Add rent-to-income ratios, owner/renter
   interactions, and income per person or adult.
3. **Missing RentIndex.** Impute from the same ZIP's other month, then from
   City → zip3 → Metro → State medians. Add a missing flag.
4. **Geographic target encoding.** Encode zip3, zip2, Metro, City and State strictly
   out-of-fold. This is the main lever for New ZIPs.
5. **Two models.**
   - Model A uses features only and handles New ZIPs.
   - Model B handles Seen ZIPs. It starts from the last known target and learns a
     correction from month-over-month feature deltas and Model A.
6. **Ensemble.** Blend LightGBM and CatBoost, try several seeds, and clip predictions to [0, 100].

## Ideas to try next

- Logit-transform the target, and weight samples by `SampleSize`. Compare both on CV.
- Nearest-neighbour target features using nearby ZIP numbers within the same zip3.
- Tune σ in the Φ feature, or fit it per state.
- Optuna tuning. Add XGBoost or a ridge model on the engineered features to the blend.
- Pseudo-relationships from the test *features* are fine to use; test *labels* do not exist.
  External data is **not allowed**.
