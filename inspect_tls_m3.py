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
print("========== M3 TRAFFIC LIGHTS ==========")
print()

for tls_id in traci.trafficlight.getIDList():

    print(f"TLS ID: {tls_id}")

    lanes = traci.trafficlight.getControlledLanes(tls_id)

    print("Controlled lanes:")
    for i, lane in enumerate(lanes):
        print(f"    Link {i}: {lane}")

    print()

    programs = traci.trafficlight.getAllProgramLogics(tls_id)

    for program in programs:
        print(f"Program: {program.programID}")

        for i, phase in enumerate(program.phases):
            print(
                f"    Phase {i}: "
                f"duration={phase.duration}, "
                f"state={phase.state}"
            )

    print()
    print("--------------------------------------")
    print()

traci.close()