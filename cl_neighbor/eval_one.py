"""
Evaluates ONE junction's checkpoint, frozen weights, in its OWN fresh
process: `python eval_one.py <tls_id> [tag]`. tag defaults to "" (today's
canonical dqn_policy_seq_{tls_id}.pt); pass a tag like "imbalance_" to
evaluate a not-yet-promoted challenger checkpoint
(dqn_policy_seq_imbalance_{tls_id}.pt) with the exact same methodology,
writing seq_after_*_imbalance_{tls_id}.xml for pick_best_general.py to
compare against canonical before any promotion.

FINAL CONVENTION (explicit user decision): epsilon=0.0, fully greedy,
no exploration. Every prior seq_after_*.xml this project produced came
from training's own recorded last episode at epsilon_min=0.05 - which
turned out to be the real source of a large, previously-unrecognized
run-to-run variance (343322679 alone swung from +11%/+6%/+15% to
-5%/-3%/-10% between two nominally-"same" epsilon=0.05 evaluations of
the identical checkpoint), because which random numbers get consumed
at 5% exploration depends on the RNG's position, which itself depends
on everything that happened in every earlier training episode -
different episode counts/configs never reliably reproduce the same
sequence. epsilon=0.0 needs no RNG at all for action selection, so it
is exactly reproducible regardless of process/loop context (verified:
identical results whether run standalone, repeated, or inside an
8-junction loop) - it is also the policy that would actually run in
real deployment, not an artificial 5%-random-mistakes version kept
only for historical reporting convention.
"""
import sys

import torch
import traci

import dqn_traffic_combined as m
import discover_neighbor_topology as nt

TLS_ID = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else ""

if __name__ == "__main__":
    config = m.discover_controllable_tls()
    info = config[TLS_ID]
    safe_tls = TAG + TLS_ID.replace("/", "_")
    neighbor_specs = nt.discover_neighbor_inflows(config)[TLS_ID]

    agent = m.SingleJunctionAgent(TLS_ID, info["green_phases"], info["lane_groups"],
                                   neighbor_specs=neighbor_specs)
    ckpt = torch.load(f"dqn_policy_seq_{safe_tls}.pt", weights_only=True)
    agent.policy_net.load_state_dict(ckpt)
    agent.epsilon = 0.0

    traci.start(["sumo", "-c", "combined.sumocfg",
                 "--tripinfo-output", f"seq_after_tripinfo_{safe_tls}.xml",
                 "--summary", f"seq_after_summary_{safe_tls}.xml",
                 "--queue-output", f"seq_after_queue_{safe_tls}.xml",
                 "--no-step-log", "--duration-log.disable", "--seed", str(m.SEED)])
    m.apply_phase_merge(TLS_ID)
    m.install_realistic_bystanders(config, TLS_ID)

    while traci.simulation.getTime() < 3600:
        state = agent.get_state()
        action = agent.choose_action(state)
        agent.apply_action(action)
    traci.close()
    print(f"=== eval_one.py finished: {TLS_ID} ===")
