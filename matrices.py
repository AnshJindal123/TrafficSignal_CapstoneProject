import xml.etree.ElementTree as ET
import csv

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

def parse_queue(file):
    tree = ET.parse(file)
    root = tree.getroot()

    total_queue = 0
    steps = 0

    for timestep in root.findall("data"):
        step_queue = 0
        for lane in timestep.findall(".//lane"):
            step_queue += float(lane.get("queueing_length_experimental", 0))
        total_queue += step_queue
        steps += 1

    return {
        "avg_queue_length": round(total_queue / steps, 2) if steps else 0
    }

def collect_metrics(tripinfo, summary, queue):
    data = {}
    data.update(parse_tripinfo(tripinfo))
    data.update(parse_summary(summary))
    data.update(parse_queue(queue))

    return data

metrics_before = collect_metrics("baseline_tripinfo.xml", "baseline_summary.xml", "baseline_queue.xml")
metrics_after = collect_metrics("after_tripinfo.xml", "after_summary.xml", "after_queue.xml")
print("Metrics before optimization:")
print(metrics_before)
print("Metrics after optimization:")
print(metrics_after)