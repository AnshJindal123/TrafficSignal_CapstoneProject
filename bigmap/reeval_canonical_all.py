"""
Re-evaluates the CURRENT canonical checkpoint for all 8 junctions fresh
(frozen weights, epsilon=0, verified green_bounds_for(), consistent
bystander timing), overwriting seq_after_*_<tls>.xml (no tag) - the
same trustworthy methodology as reeval_more_all.py, applied to canonical
this time. Needed because two junctions (1213632636,
cluster_11801662689...#3more) were promoted from 'more_' during the
FIRST (buggy) pick_best_general.py run this round, before the
442107275 discrepancy was found - their canonical .pt weights are fine
(promotion just copies the file), but their canonical seq_after_*.xml
came from the same suspect recorded-episode mechanism, so it needs
re-verification too, not just the more_ challenger side.
"""
import torch
import traci

import dqn_traffic_combined as m
import discover_neighbor_topology as nt

if __name__ == "__main__":
    config = m.discover_controllable_tls()
    neighbor_topology = nt.discover_neighbor_inflows(config)

    for tls_id, info in config.items():
        safe_tls = tls_id.replace("/", "_")
        print(f"green_bounds_for({tls_id}):", m.green_bounds_for(tls_id))

        agent = m.SingleJunctionAgent(tls_id, info["green_phases"], info["lane_groups"],
                                       neighbor_specs=neighbor_topology[tls_id])
        ckpt = torch.load(f"dqn_policy_seq_{safe_tls}.pt", weights_only=True)
        agent.policy_net.load_state_dict(ckpt)
        agent.epsilon = 0.0

        traci.start(["sumo", "-c", "combined.sumocfg",
                     "--tripinfo-output", f"seq_after_tripinfo_{safe_tls}.xml",
                     "--summary", f"seq_after_summary_{safe_tls}.xml",
                     "--queue-output", f"seq_after_queue_{safe_tls}.xml",
                     "--no-step-log", "--duration-log.disable", "--seed", str(m.SEED)])
        m.apply_phase_merge(tls_id)
        m.install_realistic_bystanders(config, tls_id)

        while traci.simulation.getTime() < 3600:
            state = agent.get_state()
            action = agent.choose_action(state)
            agent.apply_action(action)
        traci.close()
        print(f"  re-eval done: {tls_id}")

    print("=== reeval_canonical_all.py finished ===")
