"""Added for the November 2026 review (M5, M6). Reruns E2 with common random numbers.

Each fresh atom's noise is one standard-normal draw per (noise seed, atom), scaled by that arm's sigma, so arms with the same noise level get identical answers.
Also reports how many fresh-noise answers have relative error below 0.3, by question type.
"""
import zlib
from pathlib import Path

source = Path("scripts/run_e2.py").read_text(encoding="utf8")
old = "values[atom] = (true_count(atom, exact) + rng.normal(0, sigma_of[atom]), sigma_of[atom] ** 2)"
assert old in source
source = source.replace(old, "values[atom] = (true_count(atom, exact) + sigma_of[atom] * np.random.default_rng([cfg['seed'], 72, s, zlib.crc32(repr(atom).encode())]).standard_normal(), sigma_of[atom] ** 2)")
tail = '    fresh_per_full = pd.DataFrame(rows["full"])'
assert tail in source
extra = '''    for arm_name in arms:
        frame = pd.DataFrame(rows[arm_name])
        fresh = frame[~frame["free"]]
        summary[arm_name]["fresh_share_error_below_0.3"] = round(float((fresh["error"] < 0.3).mean()), 3) if len(fresh) else None
        summary[arm_name]["fresh_share_error_below_0.3_by_category"] = fresh.groupby("category")["error"].apply(lambda e: round(float((e < 0.3).mean()), 3)).to_dict()
        summary[arm_name]["reused_share_error_below_0.3"] = round(float((frame[frame["free"]]["error"] < 0.3).mean()), 3)
        summary[arm_name]["identical_errors_to_full"] = bool(np.allclose(medians[arm_name], medians["full"])) if arm_name != "full" else None
'''
source = source.replace(tail, extra + tail)
assert "final/e2.json" in source
source = source.replace("final/e2.json", "final/e2_crn.json")
exec(source)
