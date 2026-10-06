"""
Chart for the all-8-junctions-simultaneous evaluation (eval_all_simultaneous.py):
every DQN agent live at once vs the pure 55-65s fixed-time baseline.
Separate from combined_improvement_chart.png/logs_combined.py, which cover
the one-junction-at-a-time methodology - this is a distinct experiment
(every controllable junction adaptive together, agents reacting to each
other instead of predictable fixed-time neighbors) with its own results.

Three panels, kept on separate axes since they're different units
(percent vs vehicle-count vs speed) - never mixed on one scale:
  1. Network-wide %-improvement (wait/travel/loss), diverging bars.
  2. Per-junction queue %-improvement, diverging bars.
  3. Traffic-flow stat cards (trips completed, still running, teleports,
     mean speed) as plain before/after text, not a shared-axis chart -
     these four have wildly different magnitudes and don't belong on
     one bar scale.

Colors: validated diverging pair from the dataviz skill's reference
palette (blue #2a78d6 = improvement, red #e34948 = regression), not the
arbitrary matplotlib defaults used by earlier charts in this project.
"""
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE = '#2a78d6'
RED = '#e34948'
INK = '#0b0b0b'
MUTED = '#52514e'
SURFACE = '#fcfcfb'
GRID = '#e3e2dd'

net = ET.parse('combined.net.xml').getroot()

TLS_IDS = [
    'cluster_10291078963_10291078964_11497411790_11497411791_#4more',
    'cluster_11801662689_12136946746_12136946747_12136946748_#3more',
    '1213632636',
    '343322679',
    '442107275',
    '481750173',
    'GS_cluster_12692472176_12692472177_12692472178_12692472179',
    'cluster_11594569878_11909580134_3352035441_6901551440',
]
SHORT_LABELS = {
    'cluster_10291078963_10291078964_11497411790_11497411791_#4more': '#4more',
    'cluster_11801662689_12136946746_12136946747_12136946748_#3more': '#3more',
    '1213632636': '1213632636',
    '343322679': '343322679',
    '442107275': '442107275',
    '481750173': '481750173',
    'GS_cluster_12692472176_12692472177_12692472178_12692472179': 'GS_cluster',
    'cluster_11594569878_11909580134_3352035441_6901551440': 'cluster_11594569878',
}

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

def all_junction_queues(queue_file):
    """Single pass over the (large) queue-output file, accumulating every
    junction's total at once instead of re-parsing per junction."""
    totals = {t: 0.0 for t in TLS_IDS}
    steps = 0
    for _, elem in ET.iterparse(queue_file, events=('end',)):
        if elem.tag != 'data':
            continue
        steps += 1
        for lane in elem.iter('lane'):
            lane_id = lane.get('id')
            for tls_id in TLS_IDS:
                if lane_id in LANE_MAP[tls_id]:
                    totals[tls_id] += float(lane.get('queueing_length_experimental', 0))
        elem.clear()
    return {t: round(v / steps, 2) if steps else 0 for t, v in totals.items()}

def last_step_stats(path):
    tree = ET.parse(path)
    steps = tree.getroot().findall('step')
    last = steps[-1]
    # SUMO's summary `teleports` is a running total, so the last step is the
    # whole-run count (summing every step overcounted it ~1000x).
    total_teleports = float(last.get('teleports', 0))
    return {
        'teleports_total': total_teleports,
        'running_end': int(last.get('running')),
        'ended_end': int(last.get('ended')),
        'meanSpeed_end': float(last.get('meanSpeed')),
    }

def pct(o, n):
    return (o - n) / o * 100 if o else 0

before_trip = parse_tripinfo('before_tripinfo.xml')
after_trip = parse_tripinfo('seq_after_tripinfo_all8dqn.xml')
before_flow = last_step_stats('before_summary.xml')
after_flow = last_step_stats('seq_after_summary_all8dqn.xml')

net_metrics = {
    'Wait': pct(before_trip['avg_waiting_time'], after_trip['avg_waiting_time']),
    'Travel': pct(before_trip['avg_travel_time'], after_trip['avg_travel_time']),
    'Loss': pct(before_trip['avg_time_loss'], after_trip['avg_time_loss']),
}

before_queues = all_junction_queues('before_queue.xml')
after_queues = all_junction_queues('seq_after_queue_all8dqn.xml')
queue_pcts = {SHORT_LABELS[t]: pct(before_queues[t], after_queues[t]) for t in TLS_IDS}

fig = plt.figure(figsize=(13, 9), facecolor=SURFACE)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3], hspace=0.45, wspace=0.25)

# --- Panel 1: network-wide % improvement ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(SURFACE)
labels1 = list(net_metrics.keys())
values1 = list(net_metrics.values())
colors1 = [BLUE if v >= 0 else RED for v in values1]
bars1 = ax1.bar(labels1, values1, color=colors1, width=0.55)
ax1.axhline(0, color=INK, linewidth=0.8)
ax1.set_ylim(top=max(values1) * 1.35)
for b, v in zip(bars1, values1):
    ax1.text(b.get_x() + b.get_width()/2, v + (0.4 if v >= 0 else -0.9),
              f"{v:+.2f}%", ha='center', fontsize=10, color=INK)
ax1.set_title('Network-wide trip metrics (all 8 DQN simultaneous vs baseline)',
              fontsize=11, color=INK, loc='left', pad=12)
ax1.set_ylabel('Improvement (%)', color=MUTED, fontsize=9)
ax1.grid(axis='y', color=GRID, linewidth=0.6)
ax1.set_axisbelow(True)
for spine in ['top', 'right']:
    ax1.spines[spine].set_visible(False)
ax1.tick_params(colors=MUTED)

# --- Panel 2: flow stat cards ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(SURFACE)
ax2.axis('off')
ax2.set_title('Traffic flow (baseline -> all-8-DQN)', fontsize=11, color=INK, loc='left')

cards = [
    ('Trips completed', before_flow['ended_end'], after_flow['ended_end'], True),
    ('Still running at hour-end', before_flow['running_end'], after_flow['running_end'], False),
    ('Teleports (forced, total)', int(before_flow['teleports_total']), int(after_flow['teleports_total']), False),
    ('Mean speed (m/s, end)', before_flow['meanSpeed_end'], after_flow['meanSpeed_end'], True),
]
y0 = 0.86
for label, before_v, after_v, higher_is_better in cards:
    better = (after_v > before_v) if higher_is_better else (after_v < before_v)
    color = BLUE if better else RED
    arrow = '↑' if after_v > before_v else '↓'
    ax2.text(0.0, y0, label, fontsize=9.5, color=MUTED, transform=ax2.transAxes)
    bfmt = f"{before_v:,.2f}" if isinstance(before_v, float) else f"{before_v:,}"
    afmt = f"{after_v:,.2f}" if isinstance(after_v, float) else f"{after_v:,}"
    ax2.text(0.0, y0 - 0.09, f"{bfmt}  →  {afmt}  {arrow}",
              fontsize=14, color=color, transform=ax2.transAxes, fontweight='bold')
    y0 -= 0.24

# --- Panel 3: per-junction queue % improvement ---
ax3 = fig.add_subplot(gs[1, :])
ax3.set_facecolor(SURFACE)
labels3 = list(queue_pcts.keys())
values3 = list(queue_pcts.values())
colors3 = [BLUE if v >= 0 else RED for v in values3]
bars3 = ax3.bar(labels3, values3, color=colors3, width=0.55)
ax3.axhline(0, color=INK, linewidth=0.8)
for b, v in zip(bars3, values3):
    ax3.text(b.get_x() + b.get_width()/2, v + (1.5 if v >= 0 else -3.5),
              f"{v:+.1f}%", ha='center', fontsize=9, color=INK)
ax3.set_title('Per-junction queue improvement (all-8-DQN simultaneous vs baseline)',
              fontsize=11, color=INK, loc='left')
ax3.set_ylabel('Queue improvement (%)', color=MUTED, fontsize=9)
ax3.grid(axis='y', color=GRID, linewidth=0.6)
ax3.set_axisbelow(True)
for spine in ['top', 'right']:
    ax3.spines[spine].set_visible(False)
ax3.tick_params(colors=MUTED, labelsize=8.5)
plt.setp(ax3.get_xticklabels(), rotation=20, ha='right')

fig.suptitle('All-8-Junctions-Simultaneous DQN vs 55-65s Fixed-Time Baseline',
             fontsize=14, color=INK, y=0.985)
fig.savefig('chart_all8dqn.png', dpi=150, facecolor=SURFACE)
print("Chart saved to chart_all8dqn.png")
print(net_metrics)
print(queue_pcts)
