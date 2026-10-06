"""
Full queue comparison for the all-8-simultaneous runs - the queue-output
files carry three fields per lane per second (queueing_length,
queueing_length_experimental, queueing_time), but compare_federated.py and
chart_federated.py only use the AVERAGE of queueing_length_experimental per
junction. This adds what they leave out:

  - network-wide queue: summed over the 8 controlled junctions, and over
    every lane in the network
  - peak queue (worst single second), not just the average
  - queueing_time: how long queues persist, not just how long they are
  - per-junction average AND peak, each as % change vs the 55-65s baseline

queueing_length_experimental is used for lengths, matching every other
script in this folder. One streaming pass per file (they are 100+ MB).

  python queue_analysis.py      -> prints tables, writes queue_analysis.txt
"""
import os
import xml.etree.ElementTree as ET

RUNS = [
    ("Baseline (fixed-time)", "before_queue.xml"),
    ("Milestone (per-junction DQN)", "seq_after_queue_all8dqn.xml"),
    ("FL: FedAvg global", "seq_after_queue_fedglobal__all8.xml"),
    ("FL: grouped heads", "seq_after_queue_fedgrp__all8.xml"),
    ("FL: FedPer", "seq_after_queue_fedper__all8.xml"),
    ("FL: FedAvg + personalization", "seq_after_queue_fed_all8.xml"),
    ("Test: FedAvg + milestone #3more", "seq_after_queue_fedglobal_mix_all8.xml"),
]

TLS_IDS = [
    "1213632636", "343322679", "442107275", "481750173",
    "GS_cluster_12692472176_12692472177_12692472178_12692472179",
    "cluster_10291078963_10291078964_11497411790_11497411791_#4more",
    "cluster_11594569878_11909580134_3352035441_6901551440",
    "cluster_11801662689_12136946746_12136946747_12136946748_#3more",
]
SHORT = {
    "GS_cluster_12692472176_12692472177_12692472178_12692472179": "GS_cluster",
    "cluster_10291078963_10291078964_11497411790_11497411791_#4more": "#4more",
    "cluster_11594569878_11909580134_3352035441_6901551440": "cluster_11594569878",
    "cluster_11801662689_12136946746_12136946747_12136946748_#3more": "#3more",
}

net = ET.parse("combined.net.xml").getroot()
LANE_TO_TLS = {}
for c in net.findall("connection"):
    if c.get("tl") in TLS_IDS:
        LANE_TO_TLS[f"{c.get('from')}_{c.get('fromLane')}"] = c.get("tl")


def analyse(path):
    steps = 0
    sum_ctrl = sum_net = 0.0
    peak_ctrl = peak_net = 0.0
    qtime_sum = 0.0
    qtime_n = 0
    j_sum = {t: 0.0 for t in TLS_IDS}
    j_peak = {t: 0.0 for t in TLS_IDS}
    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag != "data":
            continue
        steps += 1
        step_ctrl = step_net = 0.0
        step_j = {}
        for lane in elem.iter("lane"):
            length = float(lane.get("queueing_length_experimental", 0))
            step_net += length
            t = LANE_TO_TLS.get(lane.get("id"))
            if t is not None:
                step_ctrl += length
                step_j[t] = step_j.get(t, 0.0) + length
                qt = float(lane.get("queueing_time", 0))
                if qt > 0:
                    qtime_sum += qt
                    qtime_n += 1
        sum_ctrl += step_ctrl
        sum_net += step_net
        peak_ctrl = max(peak_ctrl, step_ctrl)
        peak_net = max(peak_net, step_net)
        for t, v in step_j.items():
            j_sum[t] += v
            j_peak[t] = max(j_peak[t], v)
        elem.clear()
    return {
        "ctrl_avg": sum_ctrl / steps, "ctrl_peak": peak_ctrl,
        "net_avg": sum_net / steps, "net_peak": peak_net,
        "qtime": qtime_sum / qtime_n if qtime_n else 0.0,
        "j_avg": {t: round(v / steps, 2) for t, v in j_sum.items()},
        "j_peak": j_peak,
    }


def pct(before, after):
    return (before - after) / before * 100 if before else 0.0


results = {}
for label, path in RUNS:
    if os.path.exists(path):
        results[label] = analyse(path)
        print(f"  parsed {path}", flush=True)
    else:
        print(f"  [missing {path}] - skipped", flush=True)

base = results["Baseline (fixed-time)"]
lines = []
lines.append("=== NETWORK-WIDE QUEUE (all 8 junctions live) - metres; % = improvement vs baseline ===")
lines.append(f"{'Run':<31}{'8-jn avg':>10}{'%':>8}{'8-jn peak':>11}{'%':>8}"
             f"{'all-lane avg':>14}{'%':>8}{'all-lane peak':>15}{'%':>8}{'q-time s':>10}{'%':>8}")
for label, r in results.items():
    lines.append(
        f"{label:<31}{r['ctrl_avg']:>10.1f}{pct(base['ctrl_avg'], r['ctrl_avg']):>7.1f}%"
        f"{r['ctrl_peak']:>11.0f}{pct(base['ctrl_peak'], r['ctrl_peak']):>7.1f}%"
        f"{r['net_avg']:>14.1f}{pct(base['net_avg'], r['net_avg']):>7.1f}%"
        f"{r['net_peak']:>15.0f}{pct(base['net_peak'], r['net_peak']):>7.1f}%"
        f"{r['qtime']:>10.1f}{pct(base['qtime'], r['qtime']):>7.1f}%")

for kind, key in (("AVERAGE", "j_avg"), ("PEAK", "j_peak")):
    lines.append("")
    lines.append(f"=== PER-JUNCTION {kind} QUEUE - % improvement vs baseline (baseline metres in header) ===")
    lines.append(f"{'Run':<31}" + "".join(f"{SHORT.get(t, t)[:12]:>13}" for t in TLS_IDS))
    lines.append(f"{'  baseline (m)':<31}" + "".join(f"{base[key][t]:>13.1f}" for t in TLS_IDS))
    for label, r in results.items():
        if label == "Baseline (fixed-time)":
            continue
        lines.append(f"{label:<31}" + "".join(f"{pct(base[key][t], r[key][t]):>12.1f}%" for t in TLS_IDS))

out = "\n".join(lines)
print(out)
with open("queue_analysis.txt", "w") as f:
    f.write(out + "\n")
print("\n=== queue_analysis.py finished ===")
