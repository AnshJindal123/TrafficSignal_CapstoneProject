"""
Fixed-time reference on the merged 9-tile map: every controllable junction
on the realistic 55-65s program (m.install_realistic_program), 1 sim-hour.

Writes tripinfo with unfinished + undeparted vehicles (the fair metric -
every scheduled vehicle counted) and a summary, then prints throughput and
gridlock indicators so demand can be sanity-checked before any training.

  python run_baseline.py                        -> before_{tripinfo,summary}.xml
  python run_baseline.py demand_1of2.rou.xml 1of2   -> before_1of2_{tripinfo,summary}.xml
                                                   (demand-calibration runs)
"""
import sys
import xml.etree.ElementTree as ET

import traci

import dqn_traffic_combined as m

FAIR_ARGS = ["--tripinfo-output.write-unfinished", "true",
             "--tripinfo-output.write-undeparted", "true"]


def main(routes=None, tag=""):
    out = f"before_{tag}_" if tag else "before_"
    config = m.discover_controllable_tls()
    traci.start(["sumo", "-c", "combined.sumocfg", *(["--route-files", routes] if routes else []),
                 "--tripinfo-output", out + "tripinfo.xml", "--summary", out + "summary.xml", "--no-step-log", "--duration-log.disable",
                 "--seed", str(m.SEED), *FAIR_ARGS])
    for tls_id, info in config.items():
        m.apply_phase_merge(tls_id)
        m.install_realistic_program(tls_id, info["green_phases"], info["lane_groups"])
    while traci.simulation.getTime() < 3600:
        traci.simulationStep()
    traci.close()

    trips = ET.parse(out + "tripinfo.xml").getroot().findall("tripinfo")
    finished = [t for t in trips if float(t.get("arrival", -1)) >= 0]
    undeparted = [t for t in trips if float(t.get("depart", -1)) < 0]
    last = ET.parse(out + "summary.xml").getroot().findall("step")[-1]
    wait = sum(float(t.get("waitingTime", 0)) for t in trips) / len(trips)
    print(f"demand: {routes or 'combined.sumocfg'} | controllable junctions: {len(config)}")
    print(f"vehicles {len(trips)} | finished {len(finished)} | on road {len(trips) - len(finished) - len(undeparted)}"
          f" | never entered {len(undeparted)}")
    # teleports in the summary is cumulative - read the last step, never sum
    print(f"teleports {last.get('teleports')} | avg wait (all vehicles) {wait:.1f}s"
          f" | mean speed at end {float(last.get('meanSpeed', 0)):.2f} m/s")
    print("=== run_baseline.py finished ===")


if __name__ == "__main__":
    main(*sys.argv[1:3])
