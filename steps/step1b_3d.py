# STEP 1b — 3D analysis: how do TWO features together shape the target?
# Run after Step 1 (it reuses `j`, `T`, `train`). Paste into ONE Kaggle cell.
# Each surface = average target for every (feature A bin, feature B bin) pair.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D)

j["zipn"] = j.Zip.astype(int)


def surface(ax, a, b, bins=10, title=""):
    d = j[[a, b, T]].dropna()
    for c in (a, b):  # drop the extreme 1% on each side so axes stay readable
        d = d[d[c].between(*d[c].quantile([0.01, 0.99]))]
    xa = pd.qcut(d[a], bins, duplicates="drop")
    xb = pd.qcut(d[b], bins, duplicates="drop")
    piv = d.groupby([xa, xb], observed=False)[T].mean().unstack()
    xs = d.groupby(xb, observed=False)[b].median().values   # bin centres = medians
    ys = d.groupby(xa, observed=False)[a].median().values
    X, Y = np.meshgrid(xs, ys)
    ax.plot_surface(X, Y, piv.values, cmap="viridis", edgecolor="k", linewidth=0.2)
    ax.view_init(elev=25, azim=-130)
    ax.set_xlabel(b, fontsize=8); ax.set_ylabel(a, fontsize=8); ax.set_zlabel("target")
    ax.set_title(title, fontsize=10)


fig = plt.figure(figsize=(20, 6))
surface(fig.add_subplot(131, projection="3d"), "RentIndex", "AvgNumberofChildren",
        title="Rent × Children → target")
surface(fig.add_subplot(132, projection="3d"), "rent_to_income", "OwnerPercent",
        title="Rent-to-income × Owner% → target")
surface(fig.add_subplot(133, projection="3d"), "SeniorAdultinHouseholdPercent",
        "AvgNumberofChildren", title="Seniors × Children → target")
plt.suptitle("Step 1b: 3D surfaces (flat = no effect, tilted both ways = both matter)")
plt.tight_layout(); plt.show()

# 3D scatter: geography (ZIP number) × rent × target, coloured by target
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
d = j.dropna(subset=["RentIndex"]).sample(3000, random_state=0)
sc = ax.scatter(d.zipn, np.log(d.RentIndex), d[T], c=d[T], cmap="RdYlGn", s=4)
ax.set_xlabel("ZIP number (≈ location)"); ax.set_ylabel("log rent"); ax.set_zlabel("target")
plt.colorbar(sc, shrink=0.6); ax.set_title("Location × rent × target")
plt.tight_layout(); plt.show()

# Same surfaces as numbers: does feature B still matter once feature A is fixed?
for a, b in [("RentIndex", "AvgNumberofChildren"), ("rent_to_income", "OwnerPercent")]:
    d = j[[a, b, T]].dropna()
    tab = d.groupby([pd.qcut(d[a], 4), pd.qcut(d[b], 4)], observed=False)[T].mean().unstack()
    print(f"\nAverage target: rows = {a} quartile, columns = {b} quartile")
    print(tab.round(1).to_string())

# OPTIONAL interactive version (rotate with the mouse on Kaggle):
try:
    import plotly.express as px
    px.scatter_3d(d.assign(children=j.loc[d.index, "AvgNumberofChildren"]),
                  x="rent_to_income", y="OwnerPercent", z=T, color="children",
                  opacity=0.5, height=600, title="Interactive: rent-to-income × owner% × target").show()
except Exception as e:
    print("plotly not available:", e)
