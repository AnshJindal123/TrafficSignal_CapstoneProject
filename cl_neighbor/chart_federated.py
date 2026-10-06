"""
Federated vs milestone chart for the all-8-simultaneous evaluation.

Series (fixed order, one hue each - validated with the dataviz skill's
validator on adjacent pairs, which is what grouped bars put side by side:
worst CVD dE 9.1, normal-vision dE 19.6 with all 5):
  milestone       canonical per-junction checkpoints  (seq_after_*_all8dqn.xml)
  FL personalized federated global + 5 local episodes (seq_after_*_fed_all8.xml)
  FL global       one averaged model at all 8         (seq_after_*_fedglobal__all8.xml)
  FedPer          shared hidden layers, local output  (seq_after_*_fedper__all8.xml)
  Grouped heads   output shared within phase-count    (seq_after_*_fedgrp__all8.xml)

Panels kept on separate axes (different value ranges, never one shared scale):
  1. Network-wide wait/travel/loss % improvement vs the 55-65s baseline,
     direct-labeled (FL global's aqua is 2.74:1 on the surface - labels are
     the required relief).
  2. Per-junction queue % improvement - relief for this panel is the
     table printed by compare_federated.py (24 bars, a label on each would
     be clutter).

Run after the matching evals, e.g. python eval_federated.py sim global_ /
python eval_federated.py sim per_
"""
import os
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SERIES = [
    ("Milestone (per-junction DQN)", "all8dqn", "#2a78d6"),
    ("FL personalized", "fed_all8", "#eb6834"),
    ("FL global", "fedglobal__all8", "#1baf7a"),
    ("FedPer", "fedper__all8", "#eda100"),
    ("Grouped heads", "fedgrp__all8", "#e87ba4"),
]
# Series whose eval output doesn't exist yet are skipped, so the chart can be
# regenerated before every experiment has been run.
SERIES = [s for s in SERIES if os.path.exists(f"seq_after_tripinfo_{s[1]}.xml")]

TLS_IDS = [
    "cluster_10291078963_10291078964_11497411790_11497411791_#4more",
    "cluster_11801662689_12136946746_12136946747_12136946748_#3more",
    "1213632636",
    "343322679",
    "442107275",
    "481750173",
    "GS_cluster_12692472176_12692472177_12692472178_12692472179",
    "cluster_11594569878_11909580134_3352035441_6901551440",
]
SHORT = {
    "cluster_10291078963_10291078964_11497411790_11497411791_#4more": "#4more",
    "cluster_11801662689_12136946746_12136946747_12136946748_#3more": "#3more",
    "GS_cluster_12692472176_12692472177_12692472178_12692472179": "GS_cluster",
    "cluster_11594569878_11909580134_3352035441_6901551440": "cluster_11594569878",
}

net = ET.parse("combined.net.xml").getroot()
LANES = {t: {f"{c.get('from')}_{c.get('fromLane')}" for c in net.findall("connection") if c.get("tl") == t}
         for t in TLS_IDS}


def trip(path):
    root = ET.parse(path).getroot()
    tw = td = tl = n = 0
    for tr in root.findall("tripinfo"):
        tw += float(tr.get("waitingTime", 0))
        td += float(tr.get("duration", 0))
        tl += float(tr.get("timeLoss", 0))
        n += 1
    return tw / n, td / n, tl / n


def queues(path):
    totals = {t: 0.0 for t in TLS_IDS}
    steps = 0
    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag != "data":
            continue
        steps += 1
        for lane in elem.iter("lane"):
            lid = lane.get("id")
            for t in TLS_IDS:
                if lid in LANES[t]:
                    totals[t] += float(lane.get("queueing_length_experimental", 0))
        elem.clear()
    # Rounded like chart_all8dqn.py / pick_best_general.py: 442107275's
    # near-empty baseline (0.002 avg) would otherwise read as +100%.
    return {t: round(v / steps, 2) for t, v in totals.items()}


def pct(before, after):
    return (before - after) / before * 100 if before else 0.0


before_trip = trip("before_tripinfo.xml")
before_q = queues("before_queue.xml")

network = {}
queue_pct = {}
for label, out, _ in SERIES:
    after = trip(f"seq_after_tripinfo_{out}.xml")
    network[label] = [pct(b, a) for b, a in zip(before_trip, after)]
    q = queues(f"seq_after_queue_{out}.xml")
    queue_pct[label] = [pct(before_q[t], q[t]) for t in TLS_IDS]


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=12)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def grouped(ax, categories, values_by_series, label_values):
    x = np.arange(len(categories))
    width = 0.8 / len(SERIES)
    for i, (label, _, color) in enumerate(SERIES):
        vals = values_by_series[label]
        offs = x + (i - (len(SERIES) - 1) / 2) * width
        # edgecolor=SURFACE draws the 2px surface gap between adjacent bars
        ax.bar(offs, vals, width, color=color, edgecolor=SURFACE, linewidth=2, label=label)
        if label_values:
            for xo, v in zip(offs, vals):
                ax.text(xo, v + (0.3 if v >= 0 else -0.3), f"{v:+.1f}",
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)


fig = plt.figure(figsize=(13, 9.5), facecolor=SURFACE)
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.15], hspace=0.42)

ax1 = fig.add_subplot(gs[0])
style(ax1, "Network-wide trip metrics, all 8 junctions live (improvement vs 55-65s baseline)",
      "Improvement (%)")
grouped(ax1, ["Wait", "Travel", "Time loss"], network, label_values=True)
lo = min(min(v) for v in network.values())
hi = max(max(v) for v in network.values())
ax1.set_ylim(min(0, lo) - 2, hi * 1.15 + 1)

ax2 = fig.add_subplot(gs[1])
style(ax2, "Per-junction queue improvement, all 8 junctions live", "Queue improvement (%)")
grouped(ax2, [SHORT.get(t, t) for t in TLS_IDS], queue_pct, label_values=False)
plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")

fig.suptitle("Federated learning vs per-junction DQN milestone", fontsize=14, color=INK, x=0.125,
             ha="left", y=0.995)
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.12, 0.965), ncol=len(SERIES),
           frameon=False, fontsize=9.5, labelcolor=INK)
fig.savefig("chart_federated.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
print("Chart saved to chart_federated.png")
for label, vals in network.items():
    print(f"{label:<30} wait {vals[0]:6.2f}%  travel {vals[1]:6.2f}%  loss {vals[2]:6.2f}%")
