"""
Summarises ms_results/*.json (from eval_multiseed.py). Model labels:
  independent / fl / fl_preagg      current design (per-junction shapes)
  independent_u / fl_u / fl_preagg_u unified-interface design
  canonical                          original per-junction checkpoints (train_seed 0)
Every model is evaluated on the same eval seeds, so comparisons are PAIRED by
eval seed. Differences are reported in the metric's own units
(negative = first model lower; lower is better for wait/travel/loss/teleports,
higher is better for trips).

Usage: python summarize_multiseed.py
"""
import glob
import json
import math
import statistics as st
from collections import defaultdict

METRICS = ["wait", "travel", "loss", "teleports", "trips"]
# wait_all / delay_all count every scheduled vehicle (finished + still on the
# road + never departed) - only present in results written with FAIR_ARGS.
FAIR_METRICS = ["wait_all", "delay_all"]

data = defaultdict(dict)   # (label, train_seed) -> {eval_seed: result}
for f in glob.glob("ms_results/*.json"):
    r = json.load(open(f))
    data[(r["mode"], r["train_seed"])][r["eval_seed"]] = r
if all(k in r for d in data.values() for r in d.values() for k in FAIR_METRICS):
    METRICS = FAIR_METRICS + METRICS
# Busy junctions = the 5 with the longest average queue under fixed time
# (map-independent; the original hard-coded the 4 busy junctions of the
# 8-junction map).
_fixed = [r["queue"] for (l, _), d in data.items() if l == "fixed" for r in d.values()]
_avgq = {j: sum(q[j] for q in _fixed) / len(_fixed) for j in (_fixed[0] if _fixed else {})}
BUSY = {j: j[:20] for j in sorted(_avgq, key=_avgq.get, reverse=True)[:5]}
labels = sorted({k[0] for k in data})
trains = lambda lab: sorted(t for (l, t) in data if l == lab)


def runs(lab):
    return [(t, e, r) for t in trains(lab) for e, r in data[(lab, t)].items()]


print("found:", {lab: len(runs(lab)) for lab in labels})

print("\n(1) Pooled over all runs (train seed x eval seed): mean (sd)")
print(f"  {'metric':10}" + "".join(f"{l:>20}" for l in labels))
for key in METRICS:
    row = f"  {key:10}"
    for lab in labels:
        v = [r[key] for _, _, r in runs(lab)]
        row += f"{st.mean(v):>12.1f} ({st.stdev(v):5.1f})" if len(v) > 1 else f"{'-':>20}"
    print(row)


def paired(a, b, key):
    """b - a, paired by eval seed. Returns per-train-seed mean diffs of b's seeds."""
    base = {}
    for t, e, r in runs(a):
        base.setdefault(e, []).append(r[key])
    base = {e: st.mean(v) for e, v in base.items()}
    per_seed = []
    for t in trains(b):
        d = [r[key] - base[e] for e, r in data[(b, t)].items() if e in base]
        if d:
            per_seed.append(st.mean(d))
    return per_seed


PAIRS = [("fixed", "independent_u2"), ("fixed", "fl_u2"),
         ("fixed", "fl_u3"), ("fl_u2", "fl_u3"), ("independent_u2", "independent_u3"),
         ("independent_u2_solo", "fl_u2_solo"), ("canonical", "independent_u2_solo"),
         ("canonical", "fl_u2_solo"), ("independent_u2", "independent_u2_solo"),
         ("fl_u2", "fl_u2_solo"),
         ("independent", "fl"), ("independent_u", "fl_u"),
         ("independent_u2", "fl_u2"), ("canonical", "independent_u2"), ("canonical", "fl_u2"),
         ("independent_u", "independent_u2"), ("independent", "independent_u2"), ("independent", "independent_u"),
         ("canonical", "independent_u"), ("canonical", "fl_u"), ("canonical", "independent"),
         ("independent", "fl_u")]
print("\n(2) Paired differences  second - first  (mean over train seeds; se = sd/sqrt(n); +/- n seeds with diff<0)")
for a, b in PAIRS:
    if a not in labels or b not in labels:
        continue
    print(f"  {b}  minus  {a}")
    for key in METRICS:
        d = paired(a, b, key)
        if len(d) < 1:
            continue
        se = st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
        ref = st.mean([r[key] for _, _, r in runs(a)])
        print(f"    {key:10} {st.mean(d):9.1f}  ({st.mean(d) / ref * 100:+6.1f}%)  se {se:6.1f}   "
              f"neg in {sum(1 for x in d if x < 0)}/{len(d)} seeds")

print("\n(3) Busy-junction average queue (m), pooled mean over runs")
print(f"  {'junction':22}" + "".join(f"{l:>16}" for l in labels))
for prefix, name in BUSY.items():
    row = f"  {name:22}"
    for lab in labels:
        v = [q for _, _, r in runs(lab) for j, q in r["queue"].items() if j.startswith(prefix)]
        row += f"{st.mean(v):>16.1f}" if v else f"{'-':>16}"
    print(row)
