"""
Compare every candidate result set (canonical '' plus any challenger
tags) per junction on the same 4 metrics logs_combined.py reports, and
copy whichever wins (highest sum of the 4 % metrics) into the canonical
filenames. The canonical files are always included as a candidate, so a
challenger that doesn't actually help never gets applied.
"""
import shutil
import xml.etree.ElementTree as ET

net = ET.parse('combined.net.xml').getroot()

CANDIDATES = {
    # All 8 are canonical-only right now - every prior challenger tag
    # ("more_", "imbalance_", etc.) predated the 55-65s baseline and was
    # deleted in the file cleanup. Add a fresh challenger tag here (e.g.
    # ["", "imbalance_"]) once a junction is retuned against the current
    # baseline.
    "cluster_10291078963_10291078964_11497411790_11497411791_#4more": [""],
    "cluster_11801662689_12136946746_12136946747_12136946748_#3more": ["", "vsdqn_"],
    "1213632636": [""],
    "343322679": ["", "retune_"],
    "442107275": [""],
    "481750173": [""],
    "GS_cluster_12692472176_12692472177_12692472178_12692472179": [""],
    "cluster_11594569878_11909580134_3352035441_6901551440": [""],
}

def controlled_lanes(tls_id):
    lanes = set()
    for conn in net.findall('connection'):
        if conn.get('tl') == tls_id:
            lanes.add(f"{conn.get('from')}_{conn.get('fromLane')}")
    return lanes

LANE_MAP = {t: controlled_lanes(t) for t in CANDIDATES}

def parse_tripinfo(file):
    tree = ET.parse(file)
    root = tree.getroot()
    total_wait = total_duration = total_time_loss = count = 0
    for trip in root.findall("tripinfo"):
        total_wait += float(trip.get("waitingTime", 0))
        total_duration += float(trip.get("duration", 0))
        total_time_loss += float(trip.get("timeLoss", 0))
        count += 1
    return {
        "avg_waiting_time": round(total_wait / count, 2) if count else 0,
        "avg_travel_time": round(total_duration / count, 2) if count else 0,
        "avg_time_loss": round(total_time_loss / count, 2) if count else 0,
    }

def junction_queue(queue_file, tls_id):
    tree = ET.parse(queue_file)
    root = tree.getroot()
    total_queue = steps = 0
    lanes = LANE_MAP[tls_id]
    for ts in root.findall('data'):
        steps += 1
        for lane in ts.findall('.//lane'):
            if lane.get('id') in lanes:
                total_queue += float(lane.get('queueing_length_experimental', 0))
    return round(total_queue / steps, 2) if steps else 0

def metrics_for(tag, tls_id, before_trip):
    safe = tag + tls_id.replace("/", "_")
    after_trip = parse_tripinfo(f"seq_after_tripinfo_{safe}.xml")
    before_q = junction_queue("before_queue.xml", tls_id)
    after_q = junction_queue(f"seq_after_queue_{safe}.xml", tls_id)
    wait_pct = (before_trip["avg_waiting_time"] - after_trip["avg_waiting_time"]) / before_trip["avg_waiting_time"] * 100
    travel_pct = (before_trip["avg_travel_time"] - after_trip["avg_travel_time"]) / before_trip["avg_travel_time"] * 100
    loss_pct = (before_trip["avg_time_loss"] - after_trip["avg_time_loss"]) / before_trip["avg_time_loss"] * 100
    queue_pct = (before_q - after_q) / before_q * 100 if before_q else 0
    return [wait_pct, travel_pct, loss_pct, queue_pct]

before_trip = parse_tripinfo("before_tripinfo.xml")

# Winner = whoever wins a majority of the 4 individual metrics (wait,
# travel, loss, queue counted equally), not the raw sum of percentages.
# Queue% swings far more widely than wait/travel/loss% in this
# simulation (-150%..+100% vs single digits), so a plain sum lets queue
# silently dominate the decision even when the other three metrics
# clearly favor the other candidate. Ties on vote count fall back to sum.
print(f"{'Junction':<55} {'Tag':<12} {'Wait%':>8} {'Travel%':>8} {'Loss%':>8} {'Queue%':>8} {'Wins':>5} {'Sum':>8}")
for tls_id, tags in CANDIDATES.items():
    rows = []
    for tag in tags:
        try:
            vals = metrics_for(tag, tls_id, before_trip)
        except FileNotFoundError as e:
            print(f"{tls_id:<55} {tag or '(canonical)':<12} [missing: {e.filename}]")
            continue
        rows.append((tag, vals))

    for tag, vals in rows:
        # count metrics where this candidate beats (or ties) every other candidate
        win_count = sum(1 for k in range(4) if all(vals[k] >= other[k] for _, other in rows))
        s = sum(vals)
        label = tag or "(canonical)"
        print(f"{tls_id:<55} {label:<12} {vals[0]:>7.2f}% {vals[1]:>7.2f}% {vals[2]:>7.2f}% {vals[3]:>7.2f}% {win_count:>5} {s:>7.2f}")

    def score(row):
        tag, vals = row
        win_count = sum(1 for k in range(4) if all(vals[k] >= other[k] for _, other in rows))
        return (win_count, sum(vals))

    best_tag, best_vals = max(rows, key=score) if rows else (None, None)
    winner_label = best_tag or "(canonical, unchanged)"
    print(f"  -> winner: {winner_label}\n")

    if best_tag:  # non-empty tag beat the canonical
        safe = tls_id.replace("/", "_")
        for kind in ["tripinfo", "summary", "queue"]:
            shutil.copy(f"seq_after_{kind}_{best_tag}{safe}.xml", f"seq_after_{kind}_{safe}.xml")
        shutil.copy(f"dqn_policy_seq_{best_tag}{safe}.pt", f"dqn_policy_seq_{safe}.pt")
        print(f"  copied '{best_tag}' files into canonical names for {tls_id}\n")
