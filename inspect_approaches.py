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

TLS_IDS = [
    "12551629147",
    "343322679"
]

print()
print("==========================================================")
print("                 APPROACH ANALYSIS")
print("==========================================================")

for tls_id in TLS_IDS:

    print()
    print(f"TLS: {tls_id}")
    print("-" * 60)

    lanes = traci.trafficlight.getControlledLanes(tls_id)
    links = traci.trafficlight.getControlledLinks(tls_id)

    for i, link_group in enumerate(links):

        if not link_group:
            continue

        incoming_lane = link_group[0][0]
        outgoing_lane = link_group[0][1]

        incoming_edge = traci.lane.getEdgeID(incoming_lane)
        outgoing_edge = traci.lane.getEdgeID(outgoing_lane)

        incoming_shape = traci.lane.getShape(incoming_lane)
        outgoing_shape = traci.lane.getShape(outgoing_lane)

        print()
        print(f"LINK {i}")
        print(f"  incoming lane : {incoming_lane}")
        print(f"  incoming edge : {incoming_edge}")
        print(f"  outgoing lane : {outgoing_lane}")
        print(f"  outgoing edge : {outgoing_edge}")

        if incoming_shape:
            print(
                f"  approach start: "
                f"x={incoming_shape[0][0]:.2f}, "
                f"y={incoming_shape[0][1]:.2f}"
            )

            print(
                f"  approach end  : "
                f"x={incoming_shape[-1][0]:.2f}, "
                f"y={incoming_shape[-1][1]:.2f}"
            )

print()
print("==========================================================")

traci.close()