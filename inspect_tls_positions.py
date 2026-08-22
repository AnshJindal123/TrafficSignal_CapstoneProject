import os
import sys
import traci

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not set")

traci.start([
    "sumo",
    "-n",
    "network.net.xml"
])

print()
print("========== M3 TLS POSITIONS ==========")
print()

for tls_id in traci.trafficlight.getIDList():

    lanes = traci.trafficlight.getControlledLanes(tls_id)
    pos = traci.junction.getPosition(tls_id)

    print(f"TLS: {tls_id}")
    print(f"Position: x={pos[0]:.2f}, y={pos[1]:.2f}")
    print(f"Controlled lanes: {len(lanes)}")

    for lane in lanes:
        print(f"    {lane}")

    print()

traci.close()