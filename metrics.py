import xml.etree.ElementTree as ET


# ============================================================
# TRIPINFO METRICS
# ============================================================

def parse_tripinfo(file):

    tree = ET.parse(file)
    root = tree.getroot()

    total_wait = 0.0
    total_duration = 0.0
    total_time_loss = 0.0
    count = 0

    for trip in root.findall("tripinfo"):

        total_wait += float(trip.get("waitingTime", 0))
        total_duration += float(trip.get("duration", 0))
        total_time_loss += float(trip.get("timeLoss", 0))

        count += 1

    return {
        "vehicles": count,
        "avg_waiting_time": round(
            total_wait / count, 2
        ) if count else 0,

        "avg_travel_time": round(
            total_duration / count, 2
        ) if count else 0,

        "avg_time_loss": round(
            total_time_loss / count, 2
        ) if count else 0,

        "total_waiting_time": round(
            total_wait, 2
        )
    }


# ============================================================
# SUMMARY METRICS
# ============================================================

def parse_summary(file):

    tree = ET.parse(file)
    root = tree.getroot()

    steps = root.findall("step")

    if not steps:
        return {
            "throughput": 0,
            "mean_speed": 0
        }

    last_step = steps[-1]

    return {
        "throughput": int(
            last_step.get("arrived", 0)
        ),

        "mean_speed": round(
            float(last_step.get("meanSpeed", 0)), 2
        )
    }


# ============================================================
# QUEUE METRICS
# ============================================================

def parse_queue(file):

    tree = ET.parse(file)
    root = tree.getroot()

    total_queue = 0.0
    steps = 0

    for timestep in root.findall("data"):

        step_queue = 0.0

        for lane in timestep.findall(".//lane"):

            step_queue += float(
                lane.get(
                    "queueing_length_experimental",
                    0
                )
            )

        total_queue += step_queue
        steps += 1

    return {
        "avg_queue_length": round(
            total_queue / steps, 2
        ) if steps else 0
    }


# ============================================================
# COLLECT ALL METRICS
# ============================================================

def collect_metrics(tripinfo, summary, queue):

    data = {}

    data.update(
        parse_tripinfo(tripinfo)
    )

    data.update(
        parse_summary(summary)
    )

    data.update(
        parse_queue(queue)
    )

    return data


# ============================================================
# PERCENTAGE IMPROVEMENT
# ============================================================

def calculate_improvement(before, after):

    improvement = {}

    # Lower is better
    lower_is_better = [
        "avg_waiting_time",
        "avg_travel_time",
        "avg_time_loss",
        "total_waiting_time",
        "avg_queue_length"
    ]

    for metric in lower_is_better:

        before_value = before[metric]
        after_value = after[metric]

        if before_value != 0:

            improvement[metric] = round(
                ((before_value - after_value)
                 / before_value) * 100,
                2
            )

        else:
            improvement[metric] = 0

    # Higher is better
    higher_is_better = [
        "throughput",
        "mean_speed"
    ]

    for metric in higher_is_better:

        before_value = before[metric]
        after_value = after[metric]

        if before_value != 0:

            improvement[metric] = round(
                ((after_value - before_value)
                 / before_value) * 100,
                2
            )

        else:
            improvement[metric] = 0

    return improvement


# ============================================================
# READ RESULTS
# ============================================================

metrics_before = collect_metrics(
    "before_tripinfo.xml",
    "before_summary.xml",
    "before_queue.xml"
)

# Training episode result
metrics_training = collect_metrics(
    "after_tripinfo.xml",
    "after_summary.xml",
    "after_queue.xml"
)

# FINAL DQN EVALUATION RESULT
metrics_test = collect_metrics(
    "test_tripinfo.xml",
    "test_summary.xml",
    "test_queue.xml"
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n==============================================")
print("        BEFORE OPTIMIZATION")
print("==============================================")

for key, value in metrics_before.items():
    print(f"{key}: {value}")


print("\n==============================================")
print("        AFTER TRAINING EPISODE")
print("==============================================")

for key, value in metrics_training.items():
    print(f"{key}: {value}")


print("\n==============================================")
print("        FINAL DQN TEST")
print("==============================================")

for key, value in metrics_test.items():
    print(f"{key}: {value}")


# ============================================================
# FINAL BEFORE VS DQN COMPARISON
# ============================================================

improvement = calculate_improvement(
    metrics_before,
    metrics_test
)


print("\n==============================================")
print("     BEFORE vs FINAL DQN IMPROVEMENT")
print("==============================================")

print(
    f"Average Waiting Time : "
    f"{improvement['avg_waiting_time']}%"
)

print(
    f"Average Travel Time  : "
    f"{improvement['avg_travel_time']}%"
)

print(
    f"Average Time Loss    : "
    f"{improvement['avg_time_loss']}%"
)

print(
    f"Total Waiting Time   : "
    f"{improvement['total_waiting_time']}%"
)

print(
    f"Average Queue Length : "
    f"{improvement['avg_queue_length']}%"
)

print(
    f"Throughput           : "
    f"{improvement['throughput']}%"
)

print(
    f"Mean Speed           : "
    f"{improvement['mean_speed']}%"
)


# ============================================================
# SIMPLE FINAL TABLE
# ============================================================

print("\n==============================================")
print("             FINAL COMPARISON")
print("==============================================")

print(
    f"{'Metric':<25}"
    f"{'Before':>15}"
    f"{'After DQN':>15}"
    f"{'Improvement':>15}"
)

print("-" * 70)

metrics_to_display = [
    ("Vehicles", "vehicles"),
    ("Avg Waiting Time", "avg_waiting_time"),
    ("Avg Travel Time", "avg_travel_time"),
    ("Avg Time Loss", "avg_time_loss"),
    ("Total Waiting Time", "total_waiting_time"),
    ("Throughput", "throughput"),
    ("Mean Speed", "mean_speed"),
    ("Avg Queue Length", "avg_queue_length")
]

for name, key in metrics_to_display:

    if key == "vehicles":
        improvement_value = "-"

    else:
        improvement_value = f"{improvement[key]:.2f}%"

    print(
        f"{name:<25}"
        f"{metrics_before[key]:>15.2f}"
        f"{metrics_test[key]:>15.2f}"
        f"{improvement_value:>15}"
    )