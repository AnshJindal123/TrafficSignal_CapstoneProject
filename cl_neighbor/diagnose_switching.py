"""
Diagnostic (not part of the training/eval pipeline): counts how often
the DQN policy actually switches phase at a junction, frozen weights,
one episode, vs. the fixed-time baseline's cycle-implied switch count -
testing the hypothesis that 442107275/481750173's bad wait/travel/loss
despite near-zero queue is caused by excessive reactive switching
(each switch costs a flat 5s yellow "tax" and disrupts flow), not a
geometry/phase-count bottleneck (ruled out separately - both are plain
2-phase/2-approach junctions).
"""
import sys

import torch
import traci

import dqn_traffic_combined as m

TARGETS = ["442107275", "481750173"]

if __name__ == "__main__":
    config = m.discover_controllable_tls()

    for tls_id in TARGETS:
        info = config[tls_id]
        agent = m.SingleJunctionAgent(tls_id, info["green_phases"], info["lane_groups"])
        safe_tls = tls_id.replace("/", "_")
        ckpt = torch.load(f"dqn_policy_seq_{safe_tls}.pt", weights_only=True)
        agent.policy_net.load_state_dict(ckpt)
        agent.epsilon = 0.0

        traci.start(["sumo", "-c", "combined.sumocfg", "--no-step-log",
                     "--duration-log.disable", "--seed", str(m.SEED)])
        m.apply_phase_merge(tls_id)
        m.install_realistic_bystanders(config, tls_id)

        switch_count = 0
        last_phase = None
        queue_samples = []
        while traci.simulation.getTime() < 3600:
            state = agent.get_state()
            action = agent.choose_action(state)
            if last_phase is not None and action != last_phase:
                switch_count += 1
            last_phase = action
            agent.apply_action(action)
            lanes = traci.trafficlight.getControlledLanes(tls_id)
            queue_samples.append(sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes))
        traci.close()

        avg_q = sum(queue_samples) / len(queue_samples)
        print(f"\n{tls_id}: {switch_count} phase switches in 3600s "
              f"(1 switch = {3600/switch_count:.1f}s avg between switches), "
              f"avg queue={avg_q:.2f}, max queue={max(queue_samples)}")

    print("\n=== diagnose_switching.py finished ===")
