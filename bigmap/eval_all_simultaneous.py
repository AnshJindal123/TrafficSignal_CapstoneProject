"""
Runs all 8 controllable junctions under their own canonical DQN
checkpoints AT THE SAME TIME, in one shared simulation - unlike every
other script in this project, which runs exactly one junction under
DQN control with the other 7 held on the realistic fixed-time baseline
("bystanders", see install_realistic_bystanders() in
dqn_traffic_combined.py). This answers a different question than the
per-junction before/after tables: what happens network-wide if every
adaptive signal is live simultaneously, reacting to each other instead
of to a predictable fixed-time neighbor?

Caveat worth keeping in mind when reading the results: every one of
these 8 policies was trained/tuned with its neighbors on fixed-time
baseline, never against another RL agent's phase changes. Running them
together is out-of-distribution for all 8 - could be better (adaptive
neighbors relieve pressure faster) or worse (uncoordinated simultaneous
switching, phases fighting each other) than the one-at-a-time numbers
already measured. That is exactly what this script is for finding out.

apply_action() in dqn_traffic_combined.py is NOT reused here because it
calls traci.simulationStep() itself, which only works when a single
agent owns the entire simulation clock. Instead this script advances a
single shared clock 1 step at a time and gives each agent its own small
state machine (mid-yellow-transition or not, steps until next decision)
so all 8 decide independently but on the exact same 5-sim-second
cadence apply_action() used - min_green/max_green/imbalance semantics
are preserved exactly, just driven by a shared loop instead of each
agent's own private one.

Usage: python eval_all_simultaneous.py
Writes seq_after_{tripinfo,summary,queue}_all8dqn.xml (separate name,
does not touch any per-junction canonical/challenger files).
"""
import torch
import traci

import dqn_traffic_combined as m
import discover_neighbor_topology as nt

DECISION_INTERVAL = 5   # matches apply_action()'s 5-step hold/decide cadence
YELLOW_STEPS = 5         # matches apply_action()'s yellow-transition length


class Runner:
    def __init__(self, agent):
        self.agent = agent
        self.pending_action = None
        self.transition_remaining = 0


def main():
    config = m.discover_controllable_tls()
    neighbor_topology = nt.discover_neighbor_inflows(config)

    runners = {}
    for tls_id, info in config.items():
        safe = tls_id.replace("/", "_")
        agent = m.SingleJunctionAgent(tls_id, info["green_phases"], info["lane_groups"],
                                       neighbor_specs=neighbor_topology[tls_id])
        ckpt = torch.load(f"dqn_policy_seq_{safe}.pt", weights_only=True)
        agent.policy_net.load_state_dict(ckpt)
        agent.epsilon = 0.0
        runners[tls_id] = Runner(agent)

    traci.start(["sumo", "-c", "combined.sumocfg",
                 "--tripinfo-output", "seq_after_tripinfo_all8dqn.xml",
                 "--summary", "seq_after_summary_all8dqn.xml",
                 "--queue-output", "seq_after_queue_all8dqn.xml",
                 "--no-step-log", "--duration-log.disable", "--seed", str(m.SEED)])

    for tls_id in config:
        m.apply_phase_merge(tls_id)
    # No install_realistic_bystanders() call - every controllable junction
    # is DQN-controlled this run, so there are no bystanders. The other 10
    # (single-phase, nothing to control) junctions stay on their default
    # program, same as every other script.

    while traci.simulation.getTime() < 3600:
        traci.simulationStep()

        for tls_id, r in runners.items():
            agent = r.agent

            if r.transition_remaining > 0:
                r.transition_remaining -= 1
                if r.transition_remaining == 0:
                    traci.trafficlight.setPhase(tls_id, r.pending_action)
                    agent.current_phase = r.pending_action
                    agent.phase_time = 0
                continue

            agent.phase_time += 1
            action_idx = agent.green_phases.index(agent.current_phase)
            for i in range(agent.n_actions):
                agent.wait_since_served[i] = 0 if i == action_idx else agent.wait_since_served[i] + 1

            if agent.phase_time % DECISION_INTERVAL != 0:
                continue

            state = agent.get_state()
            action = agent.choose_action(state)
            if action != agent.current_phase:
                current = traci.trafficlight.getPhase(tls_id)
                traci.trafficlight.setPhase(tls_id, current + 1)  # yellow
                r.pending_action = action
                r.transition_remaining = YELLOW_STEPS
            else:
                # SUMO's own program auto-advances phases on its built-in
                # timer unless re-asserted - apply_action() in
                # dqn_traffic_combined.py does this every single call
                # (even on a hold), so without this the live phase drifts
                # away from agent.current_phase between decisions.
                traci.trafficlight.setPhase(tls_id, agent.current_phase)

        t = traci.simulation.getTime()
        if t % 900 == 0:
            print(f"t={t:.0f}s  " + "  ".join(
                f"{tls_id.split('_')[0]}:{r.agent.current_phase}" for tls_id, r in runners.items()
            ))

    traci.close()
    print("=== eval_all_simultaneous.py finished ===")


if __name__ == "__main__":
    main()
