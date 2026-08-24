import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

net = ET.parse('combined.net.xml').getroot()

def discover_controllable_tls():
    """Same rule as dqn_traffic_combined.py: keep junctions with >=2
    real green phases (a phase with no 'y'/'Y' and at least one
    'G'/'g' in its state string) - those are the ones a DQN agent was
    actually trained on."""
    ids = []
    for tl in net.findall('tlLogic'):
        states = [p.get('state') for p in tl.findall('phase')]
        green_phases = [s for s in states if 'y' not in s.lower() and 'g' in s.lower()]
        if len(green_phases) >= 2:
            ids.append(tl.get('id'))
    return ids

TLS_IDS = discover_controllable_tls()

def controlled_lanes(tls_id):
    lanes = set()
    for conn in net.findall('connection'):
        if conn.get('tl') == tls_id:
            lanes.add(f"{conn.get('from')}_{conn.get('fromLane')}")
    return lanes

LANE_MAP = {t: controlled_lanes(t) for t in TLS_IDS}

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
        "vehicles": count,
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

before_trip = parse_tripinfo("before_tripinfo.xml")

results = {}
pct_table = {}
print(f"{'Junction':<65} {'Wait%':>8} {'Travel%':>8} {'Loss%':>8} {'Queue%':>8}")
for tls_id in TLS_IDS:
    safe = tls_id.replace("/", "_")
    after_trip = parse_tripinfo(f"seq_after_tripinfo_{safe}.xml")

    before_q = junction_queue("before_queue.xml", tls_id)
    after_q = junction_queue(f"seq_after_queue_{safe}.xml", tls_id)

    wait_pct = (before_trip["avg_waiting_time"] - after_trip["avg_waiting_time"]) / before_trip["avg_waiting_time"] * 100
    travel_pct = (before_trip["avg_travel_time"] - after_trip["avg_travel_time"]) / before_trip["avg_travel_time"] * 100
    loss_pct = (before_trip["avg_time_loss"] - after_trip["avg_time_loss"]) / before_trip["avg_time_loss"] * 100
    queue_pct = (before_q - after_q) / before_q * 100 if before_q else 0

    pct_table[tls_id] = [wait_pct, travel_pct, loss_pct, queue_pct]
    print(f"{tls_id:<65} {wait_pct:>7.2f}% {travel_pct:>7.2f}% {loss_pct:>7.2f}% {queue_pct:>7.2f}%")

n = len(TLS_IDS)
ncols = 4
nrows = -(-n // ncols)  # ceil
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
metric_labels = ['Waiting', 'Travel', 'Loss', 'Queue']

axes_flat = np.array(axes).reshape(-1)
for ax, (tls_id, values) in zip(axes_flat, pct_table.items()):
    colors = ['#1f77b4' if v >= 0 else '#d62728' for v in values]
    bars = ax.bar(metric_labels, values, color=colors)
    ax.set_title(tls_id[:30], fontsize=9)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Improvement (%)')
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + (1 if v >= 0 else -3), f"{v:.1f}%",
                 ha='center', fontsize=8)
for ax in axes_flat[n:]:
    ax.axis('off')

plt.suptitle(f'Percentage Improvement After DQN Optimization - Combined Team Map ({n} junctions)')
plt.tight_layout()
plt.savefig('combined_improvement_chart.png', dpi=150)
print("\nChart saved to combined_improvement_chart.png")