"""
Summarises ms_results/*.json (from eval_multiseed.py): how much of the
FL-vs-independent comparison is evaluation noise vs training-seed variation.
Usage: python summarize_multiseed.py
"""
import glob
import json
import statistics as st
from collections import defaultdict

METRICS = ["wait", "travel", "loss", "teleports", "trips"]
data = defaultdict(dict)   # (mode, train_seed) -> {eval_seed: result}
for f in glob.glob("ms_results/*.json"):
    r = json.load(open(f))
    data[(r["mode"], r["train_seed"])][r["eval_seed"]] = r

modes = sorted({m for m, _ in data})
trains = sorted({t for _, t in data})
print("models x eval-seeds found:", {m: sum(len(data[(m, t)]) for t in trains if (m, t) in data) for m in modes})


def vals(mode, t, key):
    return [r[key] for r in data.get((mode, t), {}).values()]


print("\n(1) EVAL noise: same trained model, different traffic seed -> mean +/- sd over eval seeds")
for key in METRICS:
    print(key)
    for mode in modes:
        cells = []
        for t in trains:
            v = vals(mode, t, key)
            if len(v) > 1:
                cells.append(f"t{t}: {st.mean(v):7.1f}+/-{st.stdev(v):5.1f}")
        print(f"  {mode:10}" + "  ".join(cells))

print("\n(2) Variance split for each metric: avg within-model eval sd  vs  sd of model means across train seeds")
for key in METRICS:
    for mode in modes:
        within = [st.stdev(vals(mode, t, key)) for t in trains if len(vals(mode, t, key)) > 1]
        means = [st.mean(vals(mode, t, key)) for t in trains if vals(mode, t, key)]
        if within and len(means) > 1:
            print(f"  {key:10}{mode:10} eval-noise sd {st.mean(within):7.1f}   between-train-seed sd {st.stdev(means):7.1f}")

print("\n(3) FL minus independent, averaged over eval seeds, per train seed (negative = FL better for wait/travel/loss/teleports)")
for key in METRICS:
    d = []
    for t in trains:
        a, b = vals("independent", t, key), vals("fl", t, key)
        if a and b:
            d.append(st.mean(b) - st.mean(a))
    if len(d) > 1:
        print(f"  {key:10}" + " ".join(f"{x:8.1f}" for x in d) + f" | mean {st.mean(d):7.1f} sd {st.stdev(d):6.1f}")

print("\n(4) Pooled over ALL (train seed x eval seed) runs: mean per mode")
for key in METRICS:
    row = []
    for mode in modes:
        v = [r[key] for t in trains for r in data.get((mode, t), {}).values()]
        if v:
            row.append(f"{mode}: {st.mean(v):8.1f} (n={len(v)}, sd {st.stdev(v):5.1f})")
    print(f"  {key:10}" + "   ".join(row))
