"""
Replaces the master before_tripinfo/summary/queue.xml baseline (used
throughout this project by pick_best_general.py and logs_combined.py)
with one where the 8 controllable junctions run a realistic Bangalore-
style fixed-time program instead of SUMO's generic default: green
45-50s per phase, compulsory/flat regardless of how many vehicles are
queued (per explicit user direction - this is the non-DQN comparison
baseline, so it deliberately does NOT react to demand), yellow 5s flat.
Red isn't set directly - it's whatever the OTHER phases' green+yellow
add up to, same as any real fixed-time signal.

The other 10 junctions (single green phase, nothing to control) are
left on their existing default program - they're identical in every
before/after run regardless, so they don't affect the comparison.

The BASELINE_GREEN_MIN/MAX/YELLOW constants and install_realistic_program()
now live in dqn_traffic_combined.py (imported as m below) so this exact
same program can also be installed on bystander junctions during
"after" runs (see install_realistic_bystanders() there) - this script
just re-uses that single source of truth for the "before" side.
"""
import traci

import dqn_traffic_combined as m

if __name__ == "__main__":
    config = m.discover_controllable_tls()

    traci.start(["sumo", "-c", "combined.sumocfg",
                 "--tripinfo-output", "before_tripinfo.xml",
                 "--summary", "before_summary.xml",
                 "--queue-output", "before_queue.xml",
                 "--no-step-log", "--duration-log.disable", "--seed", str(m.SEED)])

    for tls_id, info in config.items():
        durations = m.install_realistic_program(tls_id, info["green_phases"], info["lane_groups"])
        print(f"{tls_id}: {len(info['green_phases'])} phases, "
              f"durations(green,yellow,...)={durations}")

    while traci.simulation.getTime() < 3600:
        traci.simulationStep()
    traci.close()

    print("\n=== generate_realistic_baseline.py finished ===")
