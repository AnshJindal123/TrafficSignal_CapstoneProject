"""
Charts for new_map (74 DQN junctions, calibrated 35% demand), read straight
from ms_results/*.json (eval_multiseed.py / eval_fixed.py output):

  chart_new_map_results.png   wait / teleports / trips for fixed-time and the
                              four DQN variants - three panels, one per metric
                              (different units are never put on one axis)
  chart_new_map_junctions.png average queue at the 5 busiest junctions
                              (by fixed-time queue), fixed-time vs FL + DQN

Bars = mean over runs (fixed: 5 eval seeds; DQN variants: 2 training seeds x
5 eval seeds); whiskers = 1 standard deviation over those runs.

Usage (needs matplotlib):  python chart_new_map.py
"""
import glob
import json
import statistics as st
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BASELINE = "#a9a8a2"   # fixed-time: neutral, not a series hue
DQN = "#2a78d6"        # categorical slot 1 (blue)
FL = "#eb6834"         # categorical slot 2 (orange)

MODELS = [  # (result label, x label, colour)
    ("fixed", "Fixed-time", BASELINE),
    ("independent_u2", "DQN\n(u2)", DQN),
    ("fl_u2_g", "FL + DQN\n(u2)", FL),
    ("independent_u3", "DQN\n(u3)", DQN),
    ("fl_u3_g", "FL + DQN\n(u3)", FL),
]
METRICS = [  # (key, title, unit, lower is better)
    ("wait_all", "Average waiting time", "seconds per vehicle", True),
    ("teleports", "Teleports (gridlocked vehicles)", "vehicles", True),
    ("trips", "Trips completed in 1 hour", "of 3,876 trips", False),
]

runs = defaultdict(list)
for f in glob.glob("ms_results/*.json"):
    r = json.load(open(f))
    runs[r["mode"]].append(r)


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---- figure 1: network-wide metrics ------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), facecolor=SURFACE)
for ax, (key, title, unit, lower_better) in zip(axes, METRICS):
    style(ax)
    means = [st.mean(r[key] for r in runs[m]) for m, _, _ in MODELS]
    sds = [st.stdev(r[key] for r in runs[m]) for m, _, _ in MODELS]
    xs = range(len(MODELS))
    ax.bar(xs, means, width=0.66, color=[c for _, _, c in MODELS],
           edgecolor=SURFACE, linewidth=2, zorder=2)
    ax.errorbar(xs, means, yerr=sds, fmt="none", ecolor=TEXT_2, elinewidth=1,
                capsize=3, zorder=3)
    base = means[0]
    for x, mval, sd in zip(xs, means, sds):
        label = f"{mval:,.0f}"
        if x:
            label += f"\n{100 * (mval - base) / base:+.0f}%"
        ax.text(x, mval + sd + base * 0.02, label, ha="center", va="bottom",
                fontsize=8.5, color=TEXT, linespacing=1.15)
    ax.set_xticks(list(xs), [lab for _, lab, _ in MODELS], fontsize=8.5)
    ax.set_ylim(0, max(m + s for m, s in zip(means, sds)) * 1.28)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, fontweight="bold", pad=18)
    ax.text(0, 1.02, f"{unit} · {'lower' if lower_better else 'higher'} is better",
            transform=ax.transAxes, fontsize=8.5, color=TEXT_2)

fig.legend(handles=[Patch(color=BASELINE, label="Fixed-time (baseline)"),
                    Patch(color=DQN, label="DQN only (independent)"),
                    Patch(color=FL, label="FL + DQN (regional)")],
           loc="upper left", ncol=3, frameon=False, fontsize=9,
           labelcolor=TEXT_2, bbox_to_anchor=(0.003, 0.945))
fig.suptitle("new_map (74 DQN junctions, 35% demand): DQN halves waiting time vs fixed-time",
             x=0.01, ha="left", y=0.995, fontsize=13, color=TEXT, fontweight="bold")
fig.text(0.01, 0.01,
         "Mean of 5 eval seeds (fixed-time) or 2 training seeds x 5 eval seeds (DQN); whiskers = 1 SD. "
         "% = change vs fixed-time. u2 = unified model, u3 = u2 + per-phase upstream vehicle count.",
         fontsize=8, color=TEXT_2)
fig.tight_layout(rect=(0, 0.04, 1, 0.88))
fig.savefig("chart_new_map_results.png", dpi=200, facecolor=SURFACE)

# ---- figure 2: busiest junctions ----------------------------------------
def avg_queue(mode):
    q = defaultdict(list)
    for r in runs[mode]:
        for j, v in r["queue"].items():
            q[j].append(v)
    return {j: st.mean(v) for j, v in q.items()}


fixed_q, fl_q = avg_queue("fixed"), avg_queue("fl_u2_g")
busiest = sorted(fixed_q, key=fixed_q.get, reverse=True)[:5]

fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
style(ax)
w = 0.36
for i, j in enumerate(busiest):
    for dx, val, col in ((-w / 2, fixed_q[j], BASELINE), (w / 2, fl_q[j], FL)):
        ax.bar(i + dx, val, width=w, color=col, edgecolor=SURFACE, linewidth=2, zorder=2)
        ax.text(i + dx, val + 5, f"{val:.0f}", ha="center", va="bottom", fontsize=9, color=TEXT)
names = [j if len(j) <= 22 else j[:20] + "…" for j in busiest]
ax.set_xticks(range(len(busiest)), names, fontsize=8.5)
ax.set_ylim(0, max(fixed_q[j] for j in busiest) * 1.15)
ax.set_ylabel("average queue (m)", color=TEXT_2, fontsize=9)
ax.legend(handles=[Patch(color=BASELINE, label="Fixed-time"),
                   Patch(color=FL, label="FL + DQN (u2)")],
          frameon=False, fontsize=9, labelcolor=TEXT_2, loc="upper right")
cuts = [100 * (1 - fl_q[j] / fixed_q[j]) for j in busiest if not j.startswith("joinedS")]
ax.set_title(f"Busiest 5 junctions: queues fall {min(cuts):.0f}-{max(cuts):.0f}%, "
             "except joinedS (spillback bottleneck)",
             loc="left", fontsize=12, color=TEXT, fontweight="bold", pad=12)
fig.text(0.01, 0.01, "Busiest = longest average queue under fixed-time. "
         "Mean over the same runs as the main chart; lower is better.",
         fontsize=8, color=TEXT_2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("chart_new_map_junctions.png", dpi=200, facecolor=SURFACE)
print("wrote chart_new_map_results.png, chart_new_map_junctions.png")
