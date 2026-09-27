# %% [markdown]
# # Predicting Housing Affordability in the U.S.
#
# **Goal:** for every ZIP in July, predict `AffordabilityPercentageTrue`: the share of
# households that are *not* cost-burdened.
#
# ### What the data analysis showed (and what the plan is)
# 1. **Every training ZIP is in both May and June.** In test, 6,830 ZIPs are *seen* (80%)
#    and 1,704 are *new* (20%).
# 2. **Seen ZIPs are easy.** The target barely moves month to month. Predicting
#    "June = May" already gives RMSE ≈ 1.25, and the value tends to drift *down* a little
#    (about −0.33 per month, different by state). So for seen ZIPs we predict
#    **June value + state drift**.
# 3. **New ZIPs are hard, and they decide the ranking.** A model that only has features
#    gets RMSE ≈ 12. Geography matters most (CA ≈ 9%, TX ≈ 62%), so the strongest
#    signals are the known values of nearby ZIPs (same zip3, metro, city, and the
#    closest ZIP numbers).
# 4. **July features shifted.** Adults, senior %, young-adult % and household size jumped
#    by 0.2–0.4 standard deviations in July for the *same* ZIPs. New ZIPs only have July
#    features, so we also train on the seen ZIPs' July features, paired with their June
#    target, which puts training and test on the same footing.
# 5. **Missing rent is filled from the same ZIP's other months, including July.** July
#    rent is almost never missing.
#
# Validation copies the test: K-fold **over ZIPs**, so a validation ZIP is never seen in
# training, and we score its July-feature row against its June target.

# %%
import os
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/input/predicting-housing-affordability-in-the-u-s")
T = "AffordabilityPercentageTrue"
N_FOLDS = 5
SEEDS = [0, 1, 2]           # more seeds = steadier predictions (costs run time)
USE_CATBOOST = True
try:
    from catboost import CatBoostRegressor
except ImportError:
    USE_CATBOOST = False


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# %% [markdown]
# ## 1. Load data
# ZIPs are read as text so `01002` keeps its leading zero.

# %%
train = pd.read_csv(f"{DATA_DIR}/train.csv", dtype={"Zip": str})
test = pd.read_csv(f"{DATA_DIR}/test.csv", dtype={"Zip": str})
sample = pd.read_csv(f"{DATA_DIR}/sample_submission.csv", dtype={"Zip": str})
for d in (train, test):
    d["m"] = pd.to_datetime(d["Month"]).dt.month

seen = set(train["Zip"])
test["is_seen"] = test["Zip"].isin(seen)
print(train.shape, test.shape)
print("test seen ZIPs:", test.is_seen.sum(), "| new ZIPs:", (~test.is_seen).sum())

# %% [markdown]
# ## 2. Quick EDA: the facts the plan is built on

# %%
may = train[train.m == 5].set_index("Zip")
jun = train[train.m == 6].set_index("Zip").loc[may.index]
drift = jun[T] - may[T]
print(f"corr(May, June) = {may[T].corr(jun[T]):.4f}")
print(f"RMSE 'June = May' = {rmse(jun[T], may[T]):.3f}, mean change = {drift.mean():.3f}")
print("\nMean target by state (top 10 by count):")
print(train[train.m == 6].groupby("State")[T].agg(["mean", "count"])
      .sort_values("count", ascending=False).head(10).round(1))
shift_cols = ["AvgNumberofAdults", "AvgHouseholdSize", "YoungAdultinHouseholdPercent",
              "SeniorAdultinHouseholdPercent"]
t7 = test[test.is_seen].set_index("Zip")[shift_cols]
print("\nJuly minus June feature shift (same ZIPs), in std units:")
print(((t7 - jun.loc[t7.index, shift_cols]).mean() / jun[shift_cols].std()).round(2))

# %% [markdown]
# ## 3. Seen ZIPs: June value + state drift
# We learn how much each state's affordability moved from May to June, shrunk toward the
# national average for small states, and assume July moves the same way.
# Checked on May→June: plain "same as last month" = 1.25 RMSE; with state drift ≈ 1.17.

# %%
DRIFT_ALPHA = 20  # shrinkage: small states lean on the national drift
g = pd.DataFrame({"d": drift, "State": may["State"]}).groupby("State")["d"].agg(["sum", "count"])
state_drift = (g["sum"] + drift.mean() * DRIFT_ALPHA) / (g["count"] + DRIFT_ALPHA)


def get_drift(states):
    return states.map(state_drift).fillna(drift.mean()).values


# sanity check on May -> June with out-of-fold drift estimates
chk = np.zeros(len(may))
for a, b in KFold(5, shuffle=True, random_state=0).split(may):
    gg = pd.DataFrame({"d": drift.iloc[a], "S": may["State"].iloc[a]}).groupby("S")["d"].agg(["sum", "count"])
    sd = (gg["sum"] + drift.iloc[a].mean() * DRIFT_ALPHA) / (gg["count"] + DRIFT_ALPHA)
    chk[b] = may[T].iloc[b].values + may["State"].iloc[b].map(sd).fillna(drift.iloc[a].mean()).values
print(f"Seen-ZIP proxy RMSE (May->June): no drift {rmse(jun[T], may[T]):.3f} | "
      f"state drift {rmse(jun[T], chk):.3f}")

# %% [markdown]
# ## 4. Features for the new-ZIP model
# * **Rent vs income:** `log(income / (40 × rent))`. A household is affordable when
#   yearly rent ≤ 30% of income, i.e. income ≥ 40 × monthly rent.
# * **Household mix:** children per adult, income per person, owner/renter × rent burden.
# * **"Compared with neighbours":** each ZIP's value minus the average of its zip3, metro,
#   state and city in the same month. This separates "rich for this area" from "rich area".
# * Rent gaps are filled from the same ZIP's other months, then from area medians.

# %%
al = pd.concat([train, test], ignore_index=True)
al["zip3"] = al.Zip.str[:3]
al["zipn"] = al.Zip.astype(int)
al["City"] = al.State + "_" + al.City.fillna("NA")
al["Metro"] = al.Metro.fillna("NA")
al["rent"] = al.RentIndex.fillna(al.groupby("Zip").RentIndex.transform("median"))
for k in ["City", "zip3", "Metro", "State"]:
    al["rent"] = al.rent.fillna(al.groupby(["m", k]).RentIndex.transform("median"))

inc = al.AnnualMedianHouseholdIncome
al["lir"] = np.log(inc / (al.rent * 40))
al["rent_inc"] = al.rent * 12 / inc
al["log_inc"] = np.log(inc)
al["log_rent"] = np.log(al.rent)
al["kids_per_adult"] = al.AvgNumberofChildren / al.AvgNumberofAdults
al["inc_per_person"] = inc / al.AvgHouseholdSize
al["own_x_lir"] = al.OwnerPercent * al.lir
al["rent_x_lir"] = al.RenterPercent * al.lir

BASE = ["SampleSize", "rent", "AnnualMedianHouseholdIncome", "OwnerPercent",
        "MedianHomeLengthofResidence", "AvgNumberofChildren", "AvgNumberofAdults",
        "AvgHouseholdSize", "AvgGenerationsinHousehold", "YoungAdultinHouseholdPercent",
        "SeniorAdultinHouseholdPercent", "zipn"]
ENG = ["lir", "rent_inc", "log_inc", "log_rent", "kids_per_adult", "inc_per_person",
       "own_x_lir", "rent_x_lir"]
REL = []
for grp in ["zip3", "Metro", "State", "City"]:
    for c in ["log_inc", "log_rent", "lir", "OwnerPercent", "AvgNumberofChildren",
              "MedianHomeLengthofResidence", "SeniorAdultinHouseholdPercent"]:
        name = f"{c}_rel_{grp}"
        al[name] = al[c] - al.groupby(["m", grp])[c].transform("mean")
        REL.append(name)
for c in ["State", "Metro", "zip3", "City"]:
    al[c] = al[c].astype("category")
FEATS = BASE + ENG + REL + ["State"]
print(len(FEATS), "feature columns")

# %% [markdown]
# ## 5. Neighbour features (built only from labels the model is allowed to see)
# For each ZIP we look up **known** ZIPs near it:
# * average target of its state / metro / zip3 / city (smoothed toward the overall mean
#   when a group has few ZIPs);
# * the targets of the 1, 3 and 8 closest ZIP numbers (ZIP numbers that are close are
#   usually close on the map).
#
# To avoid cheating, a ZIP's own label is never used for its own features. Inside
# training this is done with an inner K-fold.

# %%
zips = np.array(sorted(seen))
y_jun = train[train.m == 6].set_index("Zip")[T]
y_may = train[train.m == 5].set_index("Zip")[T]
zinfo = al[al.m == 6].set_index("Zip").loc[zips, ["zipn", "State", "Metro", "zip3", "City"]]
zinfo["y"] = y_jun.loc[zips]


def target_feats(fit_z, rows, k=8):
    f = zinfo.loc[fit_z]
    prior = f.y.mean()
    out = pd.DataFrame(index=rows.index)
    for c, a in [("State", 20), ("Metro", 5), ("zip3", 5), ("City", 3)]:
        s = f.groupby(c, observed=True).y.agg(["sum", "count"])
        key = rows[c].astype(object)
        n = key.map(s["count"]).astype(float).fillna(0)
        out["te_" + c] = (key.map(s["sum"]).astype(float).fillna(0) + prior * a) / (n + a)
        out["n_" + c] = n
    fz, fy = f.zipn.values, f.y.values
    o = np.argsort(fz)
    fz, fy = fz[o], fy[o]
    q = rows.zipn.values
    idx = np.searchsorted(fz, q)
    cand = np.stack([np.clip(idx + d, 0, len(fz) - 1) for d in range(-k, k)], 1)
    dist = np.abs(fz[cand] - q[:, None]).astype(float)
    order = np.argsort(dist, 1)[:, :k]
    cand, dist = np.take_along_axis(cand, order, 1), np.take_along_axis(dist, order, 1)
    for kk in (1, 3, 8):
        w = 1 / (1 + dist[:, :kk])
        out[f"knn{kk}"] = (fy[cand[:, :kk]] * w).sum(1) / w.sum(1)
    out["knn_d1"] = dist[:, 0]
    out["knn_std"] = fy[cand].std(1)
    return out


def inner_target_feats(fit_z, rows):
    """target features for training rows, each computed without its own ZIP's fold"""
    fit_z = np.array(fit_z)
    parts = []
    for a, b in KFold(5, shuffle=True, random_state=11).split(fit_z):
        parts.append(target_feats(fit_z[a], rows[rows.Zip.isin(set(fit_z[b]))]))
    return pd.concat(parts).loc[rows.index]


def rows_for(zs, months=(5, 6, 7)):
    """May rows get May labels; June and July rows get the June label."""
    d = al[al.Zip.isin(set(zs)) & al.m.isin(months)].copy()
    d["y"] = np.where(d.m == 5, d.Zip.map(y_may), d.Zip.map(y_jun))
    return d


# %% [markdown]
# ## 6. Models: LightGBM + CatBoost
# Two different gradient-boosting libraries make different mistakes, so averaging them
# usually beats either one.

# %%
LGB_PARAMS = dict(n_estimators=6000, learning_rate=0.02, num_leaves=31, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=5,
                  verbose=-1)
CAT_PARAMS = dict(iterations=6000, learning_rate=0.04, depth=6, l2_leaf_reg=5,
                  loss_function="RMSE", verbose=0)


def to_cat(X):
    X = X.copy()
    for c in X.columns:
        if str(X[c].dtype) == "category":
            X[c] = X[c].astype(str)
    return X


def fit_model(kind, seed, XA, ya, XB=None, yb=None, n_iter=None):
    if kind == "lgb":
        p = dict(LGB_PARAMS, random_state=seed)
        if n_iter:
            p["n_estimators"] = n_iter
        m = lgb.LGBMRegressor(**p)
        if XB is not None:
            m.fit(XA, ya, eval_set=[(XB, yb)], callbacks=[lgb.early_stopping(300, verbose=False)])
            return m, m.best_iteration_
        return m.fit(XA, ya), n_iter
    p = dict(CAT_PARAMS, random_seed=seed)
    if n_iter:
        p["iterations"] = n_iter
    cats = [c for c in XA.columns if str(XA[c].dtype) == "category"]
    m = CatBoostRegressor(**p, cat_features=cats)
    if XB is not None:
        m.fit(to_cat(XA), ya, eval_set=(to_cat(XB), yb), early_stopping_rounds=300)
        return m, m.get_best_iteration()
    return m.fit(to_cat(XA), ya), n_iter


def predict(m, X):
    if USE_CATBOOST and isinstance(m, CatBoostRegressor):
        return m.predict(to_cat(X))
    return m.predict(X)


KINDS = ["lgb"] + (["cat"] if USE_CATBOOST else [])

# %% [markdown]
# ## 7. Cross-validation that copies the "new ZIP" test
# Each fold hides 20% of ZIPs completely: all of their months and their labels. We train
# on the rest and predict the hidden ZIPs from their **July** features.

# %%
oof = {k: pd.Series(0.0, index=zips) for k in KINDS}
best_iters = {k: [] for k in KINDS}
for a, b in KFold(N_FOLDS, shuffle=True, random_state=100).split(zips):
    za, zb = zips[a], zips[b]
    A, B = rows_for(za), rows_for(zb, (7,))
    XA = pd.concat([A[FEATS], inner_target_feats(za, A)], axis=1)
    XB = pd.concat([B[FEATS], target_feats(za, B)], axis=1)
    for k in KINDS:
        m, it = fit_model(k, 0, XA, A.y, XB, B.y)
        oof[k].loc[B.Zip.values] = predict(m, XB)
        best_iters[k].append(it)
    print("fold done:", {k: round(rmse(y_jun.loc[zb], oof[k].loc[zb]), 3) for k in KINDS})

for k in KINDS:
    print(f"{k}: new-ZIP CV RMSE = {rmse(y_jun.loc[zips], oof[k]):.4f}, "
          f"mean best iters = {int(np.mean(best_iters[k]))}")

# best blend weight (simple grid)
if USE_CATBOOST:
    ws = np.linspace(0, 1, 21)
    scores = [rmse(y_jun.loc[zips], w * oof["lgb"] + (1 - w) * oof["cat"]) for w in ws]
    W_LGB = float(ws[int(np.argmin(scores))])
    print(f"blend: w_lgb={W_LGB:.2f}  RMSE={min(scores):.4f}")
else:
    W_LGB = 1.0

# %% [markdown]
# ## 8. Train on all seen ZIPs and predict the new ZIPs

# %%
A = rows_for(zips)
XA = pd.concat([A[FEATS], inner_target_feats(zips, A)], axis=1)
new_rows = al[(al.m == 7) & (~al.Zip.isin(seen))].copy()
XN = pd.concat([new_rows[FEATS], target_feats(zips, new_rows)], axis=1)

pred_new = np.zeros(len(new_rows))
for k in KINDS:
    w = W_LGB if k == "lgb" else 1 - W_LGB
    n_iter = int(np.mean(best_iters[k]) * 1.15)
    for s in SEEDS:
        m, _ = fit_model(k, s, XA, A.y, n_iter=n_iter)
        pred_new += w * predict(m, XN) / len(SEEDS)
# model predicts a June-level value; move it to July with the state drift
pred_new += get_drift(new_rows.State.astype(str))
new_pred = pd.Series(pred_new, index=new_rows.Zip.values)

# %% [markdown]
# ## 9. Seen ZIPs + write the submission

# %%
seen_pred = pd.Series(y_jun.values + get_drift(jun.loc[y_jun.index, "State"]), index=y_jun.index)
pred = pd.concat([seen_pred, new_pred])

sub = sample.copy()
sub[T] = sub.Zip.map(pred).clip(0, 100)
assert sub[T].notna().all() and np.isfinite(sub[T]).all()
assert len(sub) == len(test) and sub.Zip.str.len().eq(5).all()
sub.to_csv("submission.csv", index=False)
print(sub.head())
print(sub[T].describe())
