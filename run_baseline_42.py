import os
import sys
import traci

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not set")


SUMO_CMD = [
    "sumo",
    "-c",
    "junction_m33.sumocfg",

    "--seed",
    "42",

    "--tripinfo-output",
    "baseline42_m33_tripinfo.xml",

    "--summary",
    "baseline42_m33_summary.xml",

    "--queue-output",
    "baseline42_m33_queue.xml",

    "--no-step-log",

    "--duration-log.disable"
]


print()
print("=" * 60)
print("M3 CONTROLLED BASELINE")
print("=" * 60)
print()

traci.start(SUMO_CMD)

while traci.simulation.getTime() < 3600:
    traci.simulationStep()

traci.close()

print()
print("=" * 60)
print("BASELINE COMPLETE")
print("=" * 60)
print()
print("Created:")
print("  baseline42_tripinfo.xml")
print("  baseline42_summary.xml")
print("  baseline42_queue.xml")
print()