import xml.etree.ElementTree as ET

# Lanes controlled by the main junction's traffic light
TARGET_LANES = [
    '1285910124_0',
    '30877924#4_0',
    '30877924#4_1',
    '776970601_0',
    '776970601_1',
    '776970601_2'
]

# ------------------------
# NETWORK METRICS
# ------------------------
def parse_tripinfo(file):
    tree = ET.parse(file)
    root = tree.getroot()

    total_wait = 0
    total_duration = 0
    total_time_loss = 0
    count = 0

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
        "total_waiting_time": round(total_wait, 2)
    }

def parse_summary(file):
    tree = ET.parse(file)
    root = tree.getroot()
    last_step = root.findall("step")[-1]

    return {
        "throughput": int(last_step.get("arrived", 0)),
        "mean_speed": round(float(last_step.get("meanSpeed", 0)), 2)
    }

def parse_queue_network(file):
    tree = ET.parse(file)
    root = tree.getroot()

    total_queue = 0
    steps = 0

    for timestep in root.findall("data"):
        for lane in timestep.findall(".//lane"):
            total_queue += float(lane.get("queueing_length_experimental", 0))
        steps += 1

    return round(total_queue / steps, 2) if steps else 0

# ------------------------
# JUNCTION METRICS
# ------------------------
def parse_queue_junction(file):
    tree = ET.parse(file)
    root = tree.getroot()

    total_queue = 0
    steps = 0

    for timestep in root.findall("data"):
        step_queue = 0
        for lane in timestep.findall(".//lane"):
            if lane.get("id") in TARGET_LANES:
                step_queue += float(lane.get("queueing_length_experimental", 0))

        total_queue += step_queue
        steps += 1

    return round(total_queue / steps, 2) if steps else 0

# ------------------------
# COLLECT
# ------------------------
def collect_all(trip, summary, queue):
    data = {}
    data.update(parse_tripinfo(trip))
    data.update(parse_summary(summary))

    data["network_queue"] = parse_queue_network(queue)
    data["junction_queue"] = parse_queue_junction(queue)

    return data

# ------------------------
# RUN
# ------------------------
before = collect_all("before_tripinfo.xml", "before_summary.xml", "before_queue.xml")
after = collect_all("after_tripinfo.xml", "after_summary.xml", "after_queue.xml")

print("\n====================")
print("BASELINE (BEFORE RL)")
print("====================")
for k, v in before.items():
    print(f"{k}: {v}")

print("\n====================")
print("AFTER RL")
print("====================")
for k, v in after.items():
    print(f"{k}: {v}")

print("\n====================")
print("% IMPROVEMENT")
print("====================")
for k in ["avg_waiting_time", "avg_travel_time", "avg_time_loss", "junction_queue"]:
    b, a = before[k], after[k]
    if b:
        pct = (b - a) / b * 100
        print(f"{k}: {pct:.2f}%")
for k in ["throughput", "mean_speed"]:
    b, a = before[k], after[k]
    if b:
        pct = (a - b) / b * 100
        print(f"{k}: {pct:.2f}%")
