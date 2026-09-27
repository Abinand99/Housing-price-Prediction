# %% [markdown]
# # Predicting Housing Affordability in the U.S. (starter pipeline)
#
# Two scores matter: **Seen ZIPs** (in May/June train, predict July) and **New ZIPs**
# (never seen). New ZIPs carry more weight, so we build two models:
#
# * **Model A (geo)** – uses features only, no ZIP history. Validated with
#   `GroupKFold` on ZIP so both months of a ZIP land in the same fold. Used for New ZIPs.
# * **Model B (seen)** – Model A's features + the ZIP's last known target and the
#   month-over-month change in its features. Validated by predicting one month
#   from the other. Used for Seen ZIPs.
#
# Only competition files are used (external data is banned by the rules).

# %%
import os
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import norm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

DATA_DIR = os.environ.get(
    "DATA_DIR", "/kaggle/input/predicting-housing-affordability-in-the-u-s"
)
TARGET = "AffordabilityPercentageTrue"
SEED = 42
N_FOLDS = 5
W_NEW = 0.6  # weight of the New-ZIP RMSE in the final score; check the Evaluation page
USE_CATBOOST = True

try:
    from catboost import CatBoostRegressor
except ImportError:
    USE_CATBOOST = False


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


# %% [markdown]
# ## Load & normalise
# Column names differ slightly between the data page (`Zip`, `YearMonth`/`Month`) and the
# submission (`ZipCode`), so normalise them. ZIPs must stay 5-digit strings.

# %%
def load(name):
    df = pd.read_csv(os.path.join(DATA_DIR, name), dtype={"Zip": str, "ZipCode": str})
    df = df.rename(columns={"ZipCode": "Zip", "YearMonth": "Month"})
    df["Zip"] = df["Zip"].astype(str).str.split(".").str[0].str.zfill(5)
    if "Month" in df.columns:
        df["Month"] = pd.to_datetime(df["Month"]).dt.month
    return df


train = load("train.csv")
test = load("test.csv")
sample = load("sample_submission.csv")
print(train.shape, test.shape, sample.shape)
print(train["Month"].value_counts())

seen_zips = set(train["Zip"])
test["is_seen"] = test["Zip"].isin(seen_zips)
print("Test seen ZIPs:", test["is_seen"].sum(), " new ZIPs:", (~test["is_seen"]).sum())

# %% [markdown]
# ## Quick EDA: how stable is the target month to month?
# If May→June correlation is ~1, the last known value is the backbone for Seen ZIPs.

# %%
wide = train.pivot_table(index="Zip", columns="Month", values=TARGET)
if wide.shape[1] >= 2:
    both = wide.dropna()
    m0, m1 = both.columns[:2]
    print(f"ZIPs in both months: {len(both)}")
    print(f"corr(M{m0}, M{m1}) = {both[m0].corr(both[m1]):.4f}")
    print(f"RMSE of 'June = May' baseline: {rmse(both[m1], both[m0]):.4f}")
print(train.isna().mean().sort_values(ascending=False).head(10))

# %% [markdown]
# ## Feature engineering
# The target is "share of households whose housing cost ≤ 30% of income", so the key
# signal is **rent vs income**. If incomes are roughly log-normal around the median,
# share affordable ≈ Φ(log(0.3·income / annual_rent) / σ) — we add that as a feature.
#
# Features are built on train+test *predictors* together (no labels), which is allowed.

# %%
NUM_COLS = [
    "SampleSize", "RentIndex", "AnnualMedianHouseholdIncome", "OwnerPercent",
    "RenterPercent", "MedianHomeLengthofResidence", "AvgNumberofChildren",
    "AvgNumberofAdults", "AvgHouseholdSize", "AvgGenerationsinHousehold",
    "YoungAdultinHouseholdPercent", "SeniorAdultinHouseholdPercent",
]
CAT_COLS = ["State", "City", "Metro", "zip3", "zip2"]


def build_features(df):
    df = df.copy()
    df["zip3"] = df["Zip"].str[:3]
    df["zip2"] = df["Zip"].str[:2]
    df["zip_num"] = df["Zip"].astype(int)
    for c in ["State", "City", "Metro"]:
        df[c] = df[c].fillna("NA").astype(str)
    df["City"] = df["State"] + "_" + df["City"]

    # RentIndex is missing for many ZIPs: impute hierarchically from the same ZIP in another
    # month, then City -> zip3 -> Metro -> State medians (all within the same month).
    df["rent_missing"] = df["RentIndex"].isna().astype(int)
    df["rent_imp"] = df["RentIndex"]
    df["rent_imp"] = df["rent_imp"].fillna(df.groupby("Zip")["RentIndex"].transform("median"))
    for keys in (["Month", "City"], ["Month", "zip3"], ["Month", "Metro"], ["Month", "State"],
                 ["Month"]):
        df["rent_imp"] = df["rent_imp"].fillna(df.groupby(keys)["RentIndex"].transform("median"))

    inc = df["AnnualMedianHouseholdIncome"]
    rent_y = df["rent_imp"] * 12
    df["log_income"] = np.log1p(inc)
    df["log_rent"] = np.log1p(df["rent_imp"])
    df["rent_to_income"] = rent_y / inc
    df["log_income_over_rent"] = np.log(inc / rent_y)
    df["income_needed_ratio"] = inc / (rent_y / 0.30)  # >1 means median household affords rent
    for sigma in (0.6, 0.9, 1.2):
        df[f"phi_{sigma}"] = norm.cdf(np.log(0.30 * inc / rent_y) / sigma) * 100
    df["renter_x_burden"] = df["RenterPercent"] * df["rent_to_income"]
    df["owner_x_burden"] = df["OwnerPercent"] * df["rent_to_income"]
    df["income_per_person"] = inc / df["AvgHouseholdSize"]
    df["income_per_adult"] = inc / df["AvgNumberofAdults"]
    df["rent_per_person"] = df["rent_imp"] / df["AvgHouseholdSize"]
    df["log_sample"] = np.log1p(df["SampleSize"])

    # Relative position within the area (is this ZIP richer/cheaper than its neighbours?)
    for grp in ["zip3", "Metro", "State"]:
        for c in ["log_income", "log_rent", "rent_to_income", "RenterPercent"]:
            df[f"{c}_rel_{grp}"] = df[c] - df.groupby(["Month", grp])[c].transform("median")
        df[f"cnt_{grp}"] = df.groupby(["Month", grp])["Zip"].transform("count")
    return df


full = pd.concat([train.assign(_set="train"), test.assign(_set="test")], ignore_index=True)
full = build_features(full)
for c in CAT_COLS:
    full[c] = full[c].astype("category")
train_f = full[full["_set"] == "train"].reset_index(drop=True)
test_f = full[full["_set"] == "test"].reset_index(drop=True)

BASE_FEATS = [c for c in full.columns
              if c not in {"Zip", TARGET, "_set", "is_seen", "City"}]
print(len(BASE_FEATS), "base features")

# %% [markdown]
# ## Out-of-fold target encoding (geographic neighbours)
# For New ZIPs the best extra signal is "how affordable are nearby ZIPs we *do* know".
# Encodings are computed strictly out-of-fold, grouped by ZIP, to avoid leakage.

# %%
TE_COLS = ["zip3", "zip2", "Metro", "City", "State"]


def te_map(src, col, prior, alpha=10):
    g = src.groupby(col, observed=True)[TARGET].agg(["sum", "count"])
    return (g["sum"] + prior * alpha) / (g["count"] + alpha)


def add_te(fit_df, apply_df, n_inner=5):
    """TE for apply_df from all of fit_df; TE for fit_df from inner out-of-fold splits."""
    fit_df, apply_df = fit_df.copy(), apply_df.copy()
    prior = fit_df[TARGET].mean()
    gkf = GroupKFold(n_splits=n_inner)
    for col in TE_COLS:
        name = f"te_{col}"
        apply_df[name] = apply_df[col].astype(str).map(
            te_map(fit_df.assign(**{col: fit_df[col].astype(str)}), col, prior)
        ).fillna(prior).astype(float)
        fit_df[name] = np.nan
        for tr, va in gkf.split(fit_df, groups=fit_df["Zip"]):
            src = fit_df.iloc[tr].assign(**{col: fit_df[col].iloc[tr].astype(str)})
            fit_df.iloc[va, fit_df.columns.get_loc(name)] = (
                fit_df[col].iloc[va].astype(str).map(te_map(src, col, prior)).fillna(prior).values
            )
    # residual of the zip3 encoding vs the rent-to-income curve
    for d in (fit_df, apply_df):
        d["te_zip3_minus_phi"] = d["te_zip3"] - d["phi_0.9"]
    return fit_df, apply_df


TE_FEATS = [f"te_{c}" for c in TE_COLS] + ["te_zip3_minus_phi"]
FEATS_A = BASE_FEATS + TE_FEATS

# %% [markdown]
# ## Model A – geographic generalisation (GroupKFold by ZIP)

# %%
LGB_PARAMS = dict(
    objective="regression", learning_rate=0.03, num_leaves=31, min_child_samples=30,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
    n_estimators=5000, random_state=SEED, verbose=-1,
)


def fit_lgb(Xtr, ytr, Xva=None, yva=None, n_est=None):
    params = dict(LGB_PARAMS)
    if n_est:
        params["n_estimators"] = n_est
    m = lgb.LGBMRegressor(**params)
    if Xva is not None:
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(200, verbose=False)])
    else:
        m.fit(Xtr, ytr)
    return m


def fit_cat(Xtr, ytr, Xva=None, yva=None, n_est=None):
    m = CatBoostRegressor(
        iterations=n_est or 5000, learning_rate=0.05, depth=6, loss_function="RMSE",
        random_seed=SEED, verbose=0, cat_features=[c for c in CAT_COLS if c in Xtr.columns],
    )
    Xtr = Xtr.copy()
    for c in CAT_COLS:
        if c in Xtr:
            Xtr[c] = Xtr[c].astype(str)
    if Xva is not None:
        Xva = Xva.copy()
        for c in CAT_COLS:
            if c in Xva:
                Xva[c] = Xva[c].astype(str)
        m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=200)
    else:
        m.fit(Xtr, ytr)
    return m


def pred(m, X):
    if USE_CATBOOST and isinstance(m, CatBoostRegressor):
        X = X.copy()
        for c in CAT_COLS:
            if c in X:
                X[c] = X[c].astype(str)
    return m.predict(X)


y = train_f[TARGET].values
oof_a = {"lgb": np.zeros(len(train_f)), "cat": np.zeros(len(train_f))}
best_it = {"lgb": [], "cat": []}
gkf = GroupKFold(n_splits=N_FOLDS)
for fold, (tr, va) in enumerate(gkf.split(train_f, y, groups=train_f["Zip"])):
    ftr, fva = add_te(train_f.iloc[tr], train_f.iloc[va])
    m = fit_lgb(ftr[FEATS_A], y[tr], fva[FEATS_A], y[va])
    oof_a["lgb"][va] = m.predict(fva[FEATS_A])
    best_it["lgb"].append(m.best_iteration_)
    if USE_CATBOOST:
        m = fit_cat(ftr[FEATS_A], y[tr], fva[FEATS_A], y[va])
        oof_a["cat"][va] = pred(m, fva[FEATS_A])
        best_it["cat"].append(m.get_best_iteration())
    print(f"fold {fold}: lgb {rmse(y[va], oof_a['lgb'][va]):.4f}"
          + (f"  cat {rmse(y[va], oof_a['cat'][va]):.4f}" if USE_CATBOOST else ""))

oof_a_blend = (oof_a["lgb"] + oof_a["cat"]) / 2 if USE_CATBOOST else oof_a["lgb"]
print(f"Model A (new-ZIP proxy) CV RMSE  lgb={rmse(y, oof_a['lgb']):.4f}"
      + (f"  cat={rmse(y, oof_a['cat']):.4f}  blend={rmse(y, oof_a_blend):.4f}"
         if USE_CATBOOST else ""))
train_f["oof_a"] = oof_a_blend

# %% [markdown]
# ## Model B – Seen ZIPs (predict a month from the ZIP's other month)
# Training pairs are built in both directions (May→June and June→May) so we have enough
# rows. Features: Model A prediction, lag target, lag features, and month-over-month deltas.
# The model learns a *correction* on top of the lag target.

# %%
DELTA_COLS = ["RentIndex", "rent_imp", "AnnualMedianHouseholdIncome", "OwnerPercent",
              "RenterPercent", "SampleSize", "rent_to_income", "phi_0.9", "AvgHouseholdSize"]


def make_pairs(cur, prev):
    """cur: rows to predict; prev: the same ZIPs' earlier/other-month rows (with target)."""
    p = prev[["Zip", "Month", TARGET, "oof_a"] + DELTA_COLS].rename(
        columns={"Month": "lag_month", TARGET: "lag_target", "oof_a": "lag_oof_a",
                 **{c: f"lag_{c}" for c in DELTA_COLS}})
    d = cur.merge(p, on="Zip", how="inner")
    d["gap"] = d["Month"] - d["lag_month"]
    for c in DELTA_COLS:
        d[f"d_{c}"] = d[c] - d[f"lag_{c}"]
    # "what model A thinks changed" + the ZIP's own persistent residual
    d["lag_resid"] = d["lag_target"] - d["lag_oof_a"]
    d["a_shift"] = d["oof_a"] - d["lag_oof_a"]
    return d


may, jun = train_f[train_f["Month"] == 5], train_f[train_f["Month"] == 6]
pairs = pd.concat([make_pairs(jun, may), make_pairs(may, jun)], ignore_index=True)
FEATS_B = (["oof_a", "lag_target", "lag_resid", "a_shift", "gap"]
           + [f"d_{c}" for c in DELTA_COLS] + BASE_FEATS)
pairs["resid_target"] = pairs[TARGET] - pairs["lag_target"]
print("pair rows:", len(pairs))

oof_b = np.zeros(len(pairs))
best_it_b = []
if len(pairs) >= N_FOLDS * 10:
    for tr, va in GroupKFold(n_splits=N_FOLDS).split(pairs, groups=pairs["Zip"]):
        m = fit_lgb(pairs.iloc[tr][FEATS_B], pairs["resid_target"].iloc[tr],
                    pairs.iloc[va][FEATS_B], pairs["resid_target"].iloc[va])
        oof_b[va] = pairs["lag_target"].iloc[va] + m.predict(pairs.iloc[va][FEATS_B])
        best_it_b.append(m.best_iteration_)
    yb = pairs[TARGET].values
    print(f"Seen-ZIP baseline (lag target)  RMSE={rmse(yb, pairs['lag_target']):.4f}")
    print(f"Seen-ZIP Model A only           RMSE={rmse(yb, pairs['oof_a']):.4f}")
    print(f"Seen-ZIP Model B                RMSE={rmse(yb, oof_b):.4f}")
    seen_cv = rmse(yb, oof_b)
    new_cv = rmse(y, oof_a_blend)
    print(f"Estimated final score ≈ {(1 - W_NEW) * seen_cv + W_NEW * new_cv:.4f}")

# %% [markdown]
# ## Fit on all training data & predict July

# %%
train_te, test_te = add_te(train_f, test_f)
n_lgb = int(np.mean(best_it["lgb"]) * 1.1) or 1000
pa = fit_lgb(train_te[FEATS_A], y, n_est=n_lgb).predict(test_te[FEATS_A])
if USE_CATBOOST:
    n_cat = int(np.mean(best_it["cat"]) * 1.1) or 1000
    pa = (pa + pred(fit_cat(train_te[FEATS_A], y, n_est=n_cat), test_te[FEATS_A])) / 2
test_f["oof_a"] = pa  # Model A prediction plays the same role as oof_a in training
test_f["pred"] = pa

# Seen ZIPs: use the most recent labelled month (June if present, else May)
last = train_f.sort_values("Month").groupby("Zip").tail(1)
if best_it_b:
    mb = fit_lgb(pairs[FEATS_B], pairs["resid_target"], n_est=int(np.mean(best_it_b) * 1.1) or 500)
    tp = make_pairs(test_f, last)
    tp["pred_b"] = tp["lag_target"] + mb.predict(tp[FEATS_B])
    test_f = test_f.merge(tp[["Zip", "pred_b"]], on="Zip", how="left")
    test_f["pred"] = test_f["pred_b"].fillna(test_f["pred"])

test_f["pred"] = test_f["pred"].clip(0, 100)

# %% [markdown]
# ## Write submission in the sample's order / columns

# %%
sub = sample.rename(columns={"Zip": "ZipCode"}).copy()
key = test_f.assign(ZipCode=test_f["Zip"]).drop_duplicates(["ZipCode", "State"])
sub = sub.drop(columns=[TARGET]).merge(
    key[["ZipCode", "State", "pred"]].rename(columns={"pred": TARGET}),
    on=["ZipCode", "State"], how="left")
sub[TARGET] = sub[TARGET].fillna(float(np.mean(y))).clip(0, 100)
assert sub[TARGET].notna().all() and np.isfinite(sub[TARGET]).all()
assert len(sub) == len(sample)
sub.to_csv("submission.csv", index=False)
print(sub.head(), sub.shape)
