# STEP 2 — Test new feature ideas ONE AT A TIME.
# Run the main notebook cells 1–6 first (up to "## 6. Models"), with QUICK = True or False,
# then paste this into ONE new cell. Takes ~5–10 min on Kaggle CPU.
# Each idea is scored with the same honest "hide 20% of ZIPs" validation, on 2 different
# splits, against the current features. Negative "gain" = better.
import time

QUICK_PARAMS = dict(LGB_PARAMS, learning_rate=0.06)
_base_target_feats = target_feats


def state_curve_feats(fit_z, rows):
    """Idea A (from graph 1d): every state has its own level AND its own sensitivity to
    rent-to-income. Fit a straight line per state on training ZIPs only."""
    f = al[(al.m == 6) & al.Zip.isin(set(fit_z))][["State", "rent_inc"]].assign(
        y=zinfo.loc[al[(al.m == 6) & al.Zip.isin(set(fit_z))].Zip, "y"].values)
    f["x"] = np.log(f.rent_inc)
    gx, gy = f.x.mean(), f.y.mean()
    rows_out = pd.DataFrame(index=rows.index)
    stats = {}
    for st, d in f.groupby("State", observed=True):
        n = len(d)
        slope = np.cov(d.x, d.y)[0, 1] / max(d.x.var(), 1e-6) if n > 5 else 0.0
        w = n / (n + 30)  # small states lean toward the national line
        g_slope = np.cov(f.x, f.y)[0, 1] / f.x.var()
        stats[st] = (d.x.mean(), d.y.mean(), w * slope + (1 - w) * g_slope)
    st = rows.State.astype(str)
    mx = st.map({k: v[0] for k, v in stats.items()}).fillna(gx)
    my = st.map({k: v[1] for k, v in stats.items()}).fillna(gy)
    sl = st.map({k: v[2] for k, v in stats.items()}).fillna(0)
    rows_out["state_slope"] = sl
    rows_out["state_curve_pred"] = my + sl * (np.log(rows.rent_inc) - mx)
    return rows_out


IDEAS = {
    "A_state_curve": {"tf": state_curve_feats},
    "B_family": {"cols": {
        "kids_x_hh": lambda d: d.AvgNumberofChildren * d.AvgHouseholdSize,
        "kids_x_gen": lambda d: d.AvgNumberofChildren * d.AvgGenerationsinHousehold,
        "kids_x_logrent": lambda d: d.AvgNumberofChildren * d.log_rent}},
    "C_humps": {"cols": {
        "senior_dev": lambda d: (d.SeniorAdultinHouseholdPercent - 13).abs(),
        "adults_dev": lambda d: (d.AvgNumberofAdults - 1.9).abs(),
        "lenres_dev": lambda d: (d.MedianHomeLengthofResidence - 9).abs()}},
    "D_rank_in_state": {"cols": {
        f"{c}_pct_state": (lambda c: lambda d: d.groupby(["m", "State"], observed=True)[c].rank(pct=True))(c)
        for c in ["AvgNumberofChildren", "log_rent", "OwnerPercent", "lir"]}},
}


def quick_cv(extra_cols=(), extra_tf=None, splits=(100, 107)):
    global target_feats
    target_feats = (lambda fz, r: pd.concat([_base_target_feats(fz, r), extra_tf(fz, r)], axis=1)) \
        if extra_tf else _base_target_feats
    feats = FEATS + list(extra_cols)
    scores = []
    for sp in splits:
        o = pd.Series(0.0, index=zips)
        for a, b in KFold(5, shuffle=True, random_state=sp).split(zips):
            za, zb = zips[a], zips[b]
            A, B = rows_for(za), rows_for(zb, (7,))
            XA = pd.concat([A[feats], inner_target_feats(za, A)], axis=1)
            XB = pd.concat([B[feats], target_feats(za, B)], axis=1)
            m = lgb.LGBMRegressor(**dict(QUICK_PARAMS, random_state=0))
            m.fit(XA, A.y, eval_set=[(XB, B.y)], callbacks=[lgb.early_stopping(200, verbose=False)])
            o.loc[B.Zip.values] = m.predict(XB)
        scores.append(rmse(y_jun.loc[zips], o))
    target_feats = _base_target_feats
    return float(np.mean(scores))


results = {}
t0 = time.time()
results["baseline"] = quick_cv()
print(f"baseline            RMSE {results['baseline']:.4f}   ({time.time() - t0:.0f}s)")
for name, idea in IDEAS.items():
    for col, fn in idea.get("cols", {}).items():
        al[col] = fn(al)
    results[name] = quick_cv(idea.get("cols", {}).keys(), idea.get("tf"))
    print(f"{name:20s}RMSE {results[name]:.4f}   gain {results[name] - results['baseline']:+.4f}")

res = pd.Series(results)
(res - res["baseline"]).drop("baseline").plot.barh(
    color=["#55A868" if v < 0 else "#C44E52" for v in (res - res["baseline"]).drop("baseline")],
    title="Step 2: change in RMSE vs baseline (green = better)", figsize=(8, 3))
plt.axvline(0, color="k", lw=0.8); plt.tight_layout(); plt.show()
