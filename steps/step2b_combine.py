# STEP 2b — Is the gain real? Combine the winning ideas and check the noise level.
# Run right after Step 2 (reuses quick_cv, IDEAS, results).
# "Noise" = how much the baseline moves when only the random seed changes.
# A gain is believable only if it is clearly bigger than that.
noise = [quick_cv(seed=s) for s in (1, 2)]
print(f"baseline with other seeds: {[round(x, 4) for x in noise]}  "
      f"(seed 0: {results['baseline']:.4f})")
base_all = np.mean([results["baseline"]] + noise)

combos = {"B+C+D": ["B_family", "C_humps", "D_rank_in_state"],
          "C+D": ["C_humps", "D_rank_in_state"]}
combo_scores = {}
for name, parts in combos.items():
    cols = [c for p in parts for c in IDEAS[p].get("cols", {})]
    sc = [quick_cv(cols, seed=s) for s in (0, 1, 2)]
    combo_scores[name] = np.mean(sc)
    print(f"{name:8s} RMSE {np.mean(sc):.4f}  gain vs baseline {np.mean(sc) - base_all:+.4f}  "
          f"(per seed {[round(x, 4) for x in sc]})")

pd.Series({"baseline": base_all, **combo_scores}).plot.bar(
    rot=0, ylim=(base_all - 0.15, base_all + 0.05), color=["grey", "#55A868", "#4C72B0"],
    title="Step 2b: average RMSE over 3 seeds × 2 splits (lower = better)", figsize=(7, 3))
plt.tight_layout(); plt.show()
