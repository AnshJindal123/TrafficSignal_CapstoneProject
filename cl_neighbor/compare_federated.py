"""
Side-by-side: canonical milestone vs federated checkpoints, using the same
% formulas as pick_best_general.py (solo) and chart_all8dqn.py
(simultaneous). Run after eval_federated.py solo/sim with the same tag.

  python compare_federated.py [tag]
"""
import sys
import xml.etree.ElementTree as ET

TAG = sys.argv[1] if len(sys.argv) > 1 else ""

net = ET.parse("combined.net.xml").getroot()
TLS_IDS = [
    "1213632636", "343322679", "442107275", "481750173",
    "GS_cluster_12692472176_12692472177_12692472178_12692472179",
    "cluster_10291078963_10291078964_11497411790_11497411791_#4more",
    "cluster_11594569878_11909580134_3352035441_6901551440",
    "cluster_11801662689_12136946746_12136946747_12136946748_#3more",
]
LANES = {t: {f"{c.get('from')}_{c.get('fromLane')}" for c in net.findall("connection") if c.get("tl") == t}
         for t in TLS_IDS}


def short(t):
    for key in ("#4more", "#3more", "GS_cluster", "cluster_11594569878"):
        if key in t:
            return key
    return t


def trip(path):
    root = ET.parse(path).getroot()
    tw = td = tl = n = 0
    for tr in root.findall("tripinfo"):
        tw += float(tr.get("waitingTime", 0))
        td += float(tr.get("duration", 0))
        tl += float(tr.get("timeLoss", 0))
        n += 1
    return tw / n, td / n, tl / n


def queues(path, tls_ids):
    totals = {t: 0.0 for t in tls_ids}
    steps = 0
    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag != "data":
            continue
        steps += 1
        for lane in elem.iter("lane"):
            lid = lane.get("id")
            for t in tls_ids:
                if lid in LANES[t]:
                    totals[t] += float(lane.get("queueing_length_experimental", 0))
        elem.clear()
    # Rounded like chart_all8dqn.py / pick_best_general.py: 442107275's
    # near-empty baseline (0.002 avg) would otherwise read as +100%.
    return {t: round(v / steps, 2) for t, v in totals.items()}


def pct(before, after):
    return (before - after) / before * 100 if before else 0.0


before_trip = trip("before_tripinfo.xml")
before_q = queues("before_queue.xml", TLS_IDS)

print(f"=== SOLO (one junction live, 7 fixed-time bystanders) - canonical vs fed{TAG} ===")
print(f"{'Junction':<22}{'':>12}{'Wait%':>9}{'Travel%':>9}{'Loss%':>9}{'Queue%':>9}")
for t in TLS_IDS:
    safe = t.replace("/", "_")
    for label, out in (("canonical", safe), (f"fed{TAG}", f"fed{TAG}_{safe}")):
        try:
            a = trip(f"seq_after_tripinfo_{out}.xml")
            q = queues(f"seq_after_queue_{out}.xml", [t])[t]
        except FileNotFoundError as e:
            print(f"{short(t):<22}{label:>12}  [missing {e.filename}]")
            continue
        print(f"{short(t):<22}{label:>12}{pct(before_trip[0], a[0]):>8.2f}%{pct(before_trip[1], a[1]):>8.2f}%"
              f"{pct(before_trip[2], a[2]):>8.2f}%{pct(before_q[t], q):>8.2f}%")

print(f"\n=== SIMULTANEOUS (all 8 live) - canonical vs fed{TAG} ===")
for label, out in (("canonical", "all8dqn"), (f"fed{TAG}", f"fed{TAG}_all8")):
    try:
        a = trip(f"seq_after_tripinfo_{out}.xml")
        q = queues(f"seq_after_queue_{out}.xml", TLS_IDS)
    except FileNotFoundError as e:
        print(f"{label:>12}  [missing {e.filename}]")
        continue
    print(f"{label:>12}  network wait {pct(before_trip[0], a[0]):6.2f}%  travel {pct(before_trip[1], a[1]):6.2f}%  "
          f"loss {pct(before_trip[2], a[2]):6.2f}%")
    print("              queue% " + "  ".join(f"{short(t)}:{pct(before_q[t], q[t]):.1f}" for t in TLS_IDS))
