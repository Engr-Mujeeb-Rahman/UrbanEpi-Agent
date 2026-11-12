"""Bootstrap intervals for E2, added for the paper. Same seeds and code as run_e2.py; only intervals are new. Writes e2_ci.json."""
from pathlib import Path

source = Path("scripts/run_e2.py").read_text(encoding="utf8")
marker = "    medians = {arm: np.array([np.median(per_arm[arm][i]) for i, _, _ in plans]) for arm in arms}\n"
assert marker in source
extra = marker + """    boot = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 71]))
    picks = boot.integers(0, len(plans), (5000, len(plans)))
    ci_block = {arm: [round(float(x), 3) for x in np.percentile(np.median(medians[arm][picks], axis=1), [2.5, 97.5])] for arm in arms}
    diff_block = {arm: [round(float(x), 3) for x in np.percentile(np.median((medians["full"] - medians[arm])[picks], axis=1), [2.5, 97.5])] for arm in arms if arm != "full"}
"""
source = source.replace(marker, extra)
tail = '    fresh_per_full = pd.DataFrame(rows["full"])'
assert tail in source
source = source.replace(tail, '    summary["bootstrap_ci_median_error"] = ci_block\n    summary["bootstrap_ci_median_paired_difference_full_minus_arm"] = diff_block\n' + tail)
assert 'final/e2.json' in source
source = source.replace('final/e2.json', 'final/e2_ci.json')
exec(source)
