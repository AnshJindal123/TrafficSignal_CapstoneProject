import xml.etree.ElementTree as ET


def read_tripinfo(filename):

    root = ET.parse(filename).getroot()

    vehicles = root.findall("tripinfo")

    if not vehicles:
        return {
            "completed": 0,
            "travel": 0,
            "waiting": 0,
            "loss": 0
        }

    travel = [
        float(v.get("duration", 0))
        for v in vehicles
    ]

    waiting = [
        float(v.get("waitingTime", 0))
        for v in vehicles
    ]

    loss = [
        float(v.get("timeLoss", 0))
        for v in vehicles
    ]

    return {
        "completed": len(vehicles),
        "travel": sum(travel) / len(travel),
        "waiting": sum(waiting) / len(waiting),
        "loss": sum(loss) / len(loss)
    }


def read_summary(filename):

    root = ET.parse(filename).getroot()

    steps = root.findall("step")

    halting = []
    waiting = []
    running = []
    speed = []

    for s in steps:

        halting.append(
            float(s.get("halting", 0))
        )

        waiting.append(
            float(s.get("waiting", 0))
        )

        running.append(
            float(s.get("running", 0))
        )

        speed.append(
            float(s.get("meanSpeed", 0))
        )

    return {
        "halting": sum(halting) / len(halting),
        "waiting": sum(waiting) / len(waiting),
        "running": sum(running) / len(running),

        "max_halting": max(halting),
        "max_waiting": max(waiting),

        "final_halting": halting[-1],
        "final_waiting": waiting[-1],
        "final_running": running[-1],

        "speed": sum(speed) / len(speed)
    }


def improvement_lower(baseline, dqn):

    return (
        (baseline - dqn)
        / baseline
        * 100
    )


def improvement_higher(baseline, dqn):

    return (
        (dqn - baseline)
        / baseline
        * 100
    )


# ============================================================
# READ FILES
# ============================================================

baseline_trip = read_tripinfo(
    "baseline42_m33_tripinfo.xml"
)

dqn_trip = read_tripinfo(
    "dqn42_m33_tripinfo.xml"
)


baseline_summary = read_summary(
    "baseline42_m33_summary.xml"
)

dqn_summary = read_summary(
    "dqn42_m33_summary.xml"
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("             CONTROLLED BASELINE vs DQN")
print("=" * 70)

print()

print(
    f"{'Metric':<35}"
    f"{'Baseline':>12}"
    f"{'DQN':>12}"
    f"{'Improvement':>15}"
)

print("-" * 70)


metrics = [

    (
        "Average Travel Time",
        baseline_trip["travel"],
        dqn_trip["travel"],
        improvement_lower
    ),

    (
        "Average Waiting Time",
        baseline_trip["waiting"],
        dqn_trip["waiting"],
        improvement_lower
    ),

    (
        "Average Time Loss",
        baseline_trip["loss"],
        dqn_trip["loss"],
        improvement_lower
    ),

    (
        "Average Network Queue",
        baseline_summary["halting"],
        dqn_summary["halting"],
        improvement_lower
    ),

    (
        "Average Network Waiting",
        baseline_summary["waiting"],
        dqn_summary["waiting"],
        improvement_lower
    ),

    (
        "Average Network Speed",
        baseline_summary["speed"],
        dqn_summary["speed"],
        improvement_higher
    ),

    (
        "Completed Vehicles",
        baseline_trip["completed"],
        dqn_trip["completed"],
        improvement_higher
    )
]


for name, baseline, dqn, func in metrics:

    imp = func(
        baseline,
        dqn
    )

    print(
        f"{name:<35}"
        f"{baseline:>12.2f}"
        f"{dqn:>12.2f}"
        f"{imp:>14.2f}%"
    )


# ============================================================
# NETWORK DETAILS
# ============================================================

print()
print("=" * 70)
print("                    NETWORK DETAILS")
print("=" * 70)

print()

network_metrics = [

    (
        "Maximum Halting Vehicles",
        baseline_summary["max_halting"],
        dqn_summary["max_halting"],
        improvement_lower
    ),

    (
        "Maximum Waiting Vehicles",
        baseline_summary["max_waiting"],
        dqn_summary["max_waiting"],
        improvement_lower
    ),

    (
        "Final Halting Vehicles",
        baseline_summary["final_halting"],
        dqn_summary["final_halting"],
        improvement_lower
    ),

    (
        "Final Waiting Vehicles",
        baseline_summary["final_waiting"],
        dqn_summary["final_waiting"],
        improvement_lower
    ),

    (
        "Final Running Vehicles",
        baseline_summary["final_running"],
        dqn_summary["final_running"],
        improvement_lower
    )
]


for name, baseline, dqn, func in network_metrics:

    imp = func(
        baseline,
        dqn
    )

    print(
        f"{name:<35}"
        f"{baseline:>12.2f}"
        f"{dqn:>12.2f}"
        f"{imp:>14.2f}%"
    )


print()
print("=" * 70)