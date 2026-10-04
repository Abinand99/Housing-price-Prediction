# STEP 1 — Understand the data with graphs (paste into ONE Kaggle cell, CPU is fine)
# Goal: find which features matter, whether the relationships are curved,
# and what is missing, so we know what features to build in Step 2.
import glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.environ.get("DATA_DIR") or os.path.dirname(
    (glob.glob("/kaggle/input/**/train.csv", recursive=True) or ["./train.csv"])[0])
T = "AffordabilityPercentageTrue"
train = pd.read_csv(f"{DATA_DIR}/train.csv", dtype={"Zip": str})
test = pd.read_csv(f"{DATA_DIR}/test.csv", dtype={"Zip": str})
train["m"] = pd.to_datetime(train["Month"]).dt.month
j = train[train.m == 6].copy()                     # one row per ZIP (June)
j["rent_to_income"] = j.RentIndex * 12 / j.AnnualMedianHouseholdIncome

NUM = ["SampleSize", "RentIndex", "AnnualMedianHouseholdIncome", "rent_to_income",
       "OwnerPercent", "MedianHomeLengthofResidence", "AvgNumberofChildren",
       "AvgNumberofAdults", "AvgHouseholdSize", "AvgGenerationsinHousehold",
       "YoungAdultinHouseholdPercent", "SeniorAdultinHouseholdPercent"]

# --- Graph 1: each feature vs target (dots + average line per 20 bins) -------------
fig, axes = plt.subplots(3, 4, figsize=(18, 11))
for ax, c in zip(axes.ravel(), NUM):
    x = j[c]
    lo, hi = x.quantile([0.01, 0.99])
    k = x.between(lo, hi)
    ax.scatter(x[k], j[T][k], s=2, alpha=0.15)
    b = pd.qcut(x[k], 20, duplicates="drop")
    line = j[k].groupby(b, observed=True).agg(x=(c, "median"), y=(T, "mean"))
    ax.plot(line.x, line.y, "r-", lw=2)
    ax.set_title(f"{c}\ncorr={x.corr(j[T]):+.2f}", fontsize=10)
plt.suptitle("Step 1a: each feature vs target (red = average)", fontsize=14)
plt.tight_layout(); plt.show()

# --- Graph 2: correlation heatmap ---------------------------------------------------
corr = j[NUM + [T]].corr()
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=8)
for i in range(len(corr)):
    for k2 in range(len(corr)):
        ax.text(k2, i, f"{corr.iloc[i, k2]:.1f}", ha="center", va="center", fontsize=6)
plt.colorbar(im); plt.title("Step 1b: correlations"); plt.tight_layout(); plt.show()

# --- Graph 3: geography — target along ZIP numbers, and spread inside areas ---------
fig, ax = plt.subplots(1, 2, figsize=(18, 4))
s = j.sort_values("Zip")
ax[0].scatter(s.Zip.astype(int), s[T], s=2, alpha=0.3)
ax[0].plot(s.Zip.astype(int), s[T].rolling(25, center=True).mean(), "r-", lw=1)
ax[0].set_title("Step 1c: target along ZIP numbers (red = rolling average of 25 neighbours)")
spread = {g: np.sqrt(((j[T] - j.groupby(key)[T].transform("mean")) ** 2).mean())
          for g, key in [("nothing", np.zeros(len(j))), ("State", j.State), ("Metro", j.Metro),
                         ("zip3", j.Zip.str[:3]), ("City", j.State + j.City.fillna(""))]}
pd.Series(spread).plot.bar(ax=ax[1], rot=0, color="#55A868")
ax[1].set_title("Spread left after knowing the area (lower = area explains more)")
plt.tight_layout(); plt.show()

# --- Graph 4: do relationships differ by state? (rent_to_income vs target) ----------
fig, ax = plt.subplots(figsize=(10, 5))
for st in ["CA", "TX", "FL", "NY", "OH"]:
    d = j[j.State == st].dropna(subset=["rent_to_income"])
    b = pd.qcut(d.rent_to_income, 8, duplicates="drop")
    line = d.groupby(b, observed=True).agg(x=("rent_to_income", "median"), y=(T, "mean"))
    ax.plot(line.x, line.y, "o-", label=st)
ax.set_xlabel("rent × 12 / income"); ax.set_ylabel("average target"); ax.legend()
ax.set_title("Step 1d: same feature, different effect per state?")
plt.tight_layout(); plt.show()

# --- Numbers ------------------------------------------------------------------------
print("Missing values (train | test):")
print(pd.concat([train.isna().sum(), test.isna().sum()], axis=1, keys=["train", "test"])
      .query("train > 0 or test > 0"))
print("\nCorrelation with target, strongest first:")
print(j[NUM].corrwith(j[T]).sort_values(key=abs, ascending=False).round(3))
