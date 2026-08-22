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

TARGET_TLS = [
    "12551629147",
    "343322679"
]

print()
print("==========================================================")
print("              JOINED TLS CONNECTION ANALYSIS")
print("==========================================================")

for tls_id in TARGET_TLS:

    print()
    print("##########################################################")
    print(f"TLS: {tls_id}")
    print("##########################################################")

    lanes = traci.trafficlight.getControlledLanes(tls_id)

    links = traci.trafficlight.getControlledLinks(tls_id)

    print()
    print("CONTROLLED LANES")
    print("----------------")

    for i, lane in enumerate(lanes):
        print(f"Link {i}: {lane}")

    print()
    print("CONTROLLED LINKS")
    print("----------------")

    for i, link_group in enumerate(links):

        print()
        print(f"LINK {i}")

        for link in link_group:

            incoming = link[0]
            outgoing = link[1]
            via = link[2]

            print(f"    incoming : {incoming}")
            print(f"    outgoing : {outgoing}")
            print(f"    via      : {via}")

print()
print("==========================================================")

traci.close()