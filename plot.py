import matplotlib.pyplot as plt

# ============================================================
# BEFORE vs FINAL DQN TEST
# ============================================================

metrics = {
    "Waiting": (1.89, 1.83),
    "Travel": (69.18, 69.03),
    "Loss": (20.53, 20.36),
    "Queue": (9.19, 8.93),
    "Speed": (8.87, 8.88)
}

names = []
improvements = []

for metric, (before, after) in metrics.items():

    # Lower is better for these metrics
    if metric in ["Waiting", "Travel", "Loss", "Queue"]:
        change = ((before - after) / before) * 100

    # Higher is better for speed
    elif metric == "Speed":
        change = ((after - before) / before) * 100

    names.append(metric)
    improvements.append(change)


# ============================================================
# GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(names, improvements)

plt.title("Percentage Improvement After DQN Optimization")
plt.xlabel("Traffic Performance Metric")
plt.ylabel("Improvement (%)")

plt.axhline(0)

# Add percentage values
for i, value in enumerate(improvements):

    if value >= 0:
        vertical_position = value + 0.05
        alignment = "bottom"
    else:
        vertical_position = value - 0.05
        alignment = "top"

    plt.text(
        i,
        vertical_position,
        f"{value:.2f}%",
        ha="center",
        va=alignment
    )

plt.tight_layout()
plt.show()