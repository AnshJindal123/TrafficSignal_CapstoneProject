"""
Fixed-time reference: all 8 controllable junctions on the realistic
fixed-time program (55-65 s green by lane count, 5 s yellow - the same
baseline generate_realistic_baseline.py uses) evaluated on one SUMO eval
seed, so the DQN systems can be measured against static timers on exactly the
same traffic (eval seeds 11,22,33,44,55) as every other model.

Usage: FL_EVAL_SEED=<n> python eval_fixed.py
Writes ms_results/fixed_e<n>.json (label "fixed", train_seed 0), the same
format eval_multiseed.py writes, so summarize_multiseed.py picks it up.
"""
import json
import os

import traci

import dqn_traffic_combined as m
import eval_fl_compare as ef
import federated_regional as fr


def main():
    os.makedirs("ms_eval_tmp", exist_ok=True)
    os.makedirs("ms_results", exist_ok=True)
    config = m.discover_controllable_tls()
    tag = f"fixed_e{fr.EVAL_SEED}"
    paths = {k: f"ms_eval_tmp/{k}_{tag}.xml" for k in ("tripinfo", "summary", "queue")}
    traci.start(["sumo", "-c", "combined.sumocfg", "--no-step-log", "--duration-log.disable",
                 "--seed", str(fr.EVAL_SEED),
                 "--tripinfo-output", paths["tripinfo"], "--summary", paths["summary"],
                 "--queue-output", paths["queue"]])
    for tls_id, info in config.items():
        m.apply_phase_merge(tls_id)
        m.install_realistic_program(tls_id, info["green_phases"], info["lane_groups"])
    while traci.simulation.getTime() < 3600:
        traci.simulationStep()
    traci.close()
    out = {"train_seed": 0, "eval_seed": fr.EVAL_SEED, "mode": "fixed",
           **ef.trip_metrics(paths["tripinfo"]), **ef.summary_metrics(paths["summary"]),
           "queue": {t: ef.junction_queue(paths["queue"], ef.controlled_lanes(t)) for t in config}}
    json.dump(out, open(f"ms_results/{tag}.json", "w"))
    for p in paths.values():
        os.remove(p)
    print("done", tag)


if __name__ == "__main__":
    main()
