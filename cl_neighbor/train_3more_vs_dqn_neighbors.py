"""
cl_neighbor version of ../cl_retrain/train_3more_vs_dqn_neighbors.py: fine-tunes
#3more's freshly-trained (neighbor-outflow-augmented) canonical checkpoint
against its REAL neighbors - the other 7 junctions' own canonical DQN
policies (each also neighbor-outflow-augmented where applicable), frozen
(epsilon=0, deterministic) - instead of the fixed-time bystander program
train_all.py used.

Why: the base 5-junction neighbor-outflow retrain reproduced the same
solo-vs-simultaneous gap #3more had before - it does fine one-at-a-time
against fixed-time bystanders but the comparison under
eval_all_simultaneous.py is confounded, since cl_retrain's canonical
#3more there is already the vsdqn_ joint-trained fix while this folder's
#3more is still a plain bystander-trained model on the new state. This
script closes that gap the same way it was closed in cl_retrain, applied
on top of the new state shape.

Architecture: identical shared-clock multi-agent driver + fused training
loop as cl_retrain's version - see that file's docstring for the full
rationale. Only difference here: every SingleJunctionAgent (target and
frozen neighbors alike) is constructed with neighbor_specs from
discover_neighbor_topology.py, matching how these checkpoints were
actually trained/sized in this folder.

Warm-started from cl_neighbor's own current #3more checkpoint (not
cl_retrain's) with a reduced starting epsilon (0.3, not 1.0). Tagged
'vsdqn_' so it never overwrites canonical; compare (both solo AND under
eval_all_simultaneous.py) before promoting.
"""
import sys

import torch
import traci

import dqn_traffic_combined as m
import discover_neighbor_topology as nt

TARGET_TLS = "cluster_11801662689_12136946746_12136946747_12136946748_#3more"
NUM_EPISODES = 80
START_EPSILON = 0.3
EPSILON_MIN = 0.05
EXPLORE_FRACTION = 0.85
DECISION_INTERVAL = 5
YELLOW_STEPS = 5
TAG = "vsdqn_"
REWARD_WEIGHTS = (0.005, 0.002, 1)  # same as #3more's established weights


def custom_epsilon_decay(start, end, num_episodes, steps_per_episode=600, explore_fraction=0.85):
    target_steps = max(1, num_episodes * steps_per_episode * explore_fraction)
    return (end / start) ** (1.0 / target_steps)


class Runner:
    def __init__(self, agent, is_target):
        self.agent = agent
        self.is_target = is_target
        self.pending_action = None
        self.transition_remaining = 0
        self.pending_train = None  # (state, action) awaiting next_state/reward


def build_runners(config, neighbor_topology):
    runners = {}
    for tls_id, info in config.items():
        specs = neighbor_topology[tls_id]
        is_target = (tls_id == TARGET_TLS)
        if is_target:
            agent = m.SingleJunctionAgent(
                tls_id, info["green_phases"], info["lane_groups"],
                queue_weight=REWARD_WEIGHTS[0], wait_weight=REWARD_WEIGHTS[1],
                queue_power=REWARD_WEIGHTS[2],
                epsilon_decay=custom_epsilon_decay(START_EPSILON, EPSILON_MIN, NUM_EPISODES,
                                                    explore_fraction=EXPLORE_FRACTION),
                neighbor_specs=specs)
            safe = tls_id.replace("/", "_")
            ckpt = torch.load(f"dqn_policy_seq_{safe}.pt", weights_only=True)
            agent.policy_net.load_state_dict(ckpt)
            agent.target_net.load_state_dict(ckpt)
            agent.epsilon = START_EPSILON
        else:
            agent = m.SingleJunctionAgent(tls_id, info["green_phases"], info["lane_groups"],
                                           neighbor_specs=specs)
            safe = tls_id.replace("/", "_")
            ckpt = torch.load(f"dqn_policy_seq_{safe}.pt", weights_only=True)
            agent.policy_net.load_state_dict(ckpt)
            agent.epsilon = 0.0
        runners[tls_id] = Runner(agent, is_target)
    return runners


def reset_episode_state(runners, config):
    for tls_id, r in runners.items():
        info = config[tls_id]
        r.agent.current_phase = info["green_phases"][0]
        r.agent.phase_time = 0
        r.agent.wait_since_served = [0] * r.agent.n_actions
        r.agent.prev_queue = 0
        r.agent.prev_queues = [0] * r.agent.n_actions
        r.pending_action = None
        r.transition_remaining = 0
        r.pending_train = None


def run_episode(runners, config, is_last, step_counter):
    safe = TARGET_TLS.replace("/", "_")
    record_cmd = ["sumo", "-c", "combined.sumocfg",
                  "--tripinfo-output", f"seq_after_tripinfo_{TAG}{safe}.xml",
                  "--summary", f"seq_after_summary_{TAG}{safe}.xml",
                  "--queue-output", f"seq_after_queue_{TAG}{safe}.xml",
                  "--no-step-log", "--duration-log.disable", "--seed", str(m.SEED)]
    quiet_cmd = ["sumo", "-c", "combined.sumocfg", "--no-step-log",
                 "--duration-log.disable", "--seed", str(m.SEED)]

    traci.start(record_cmd if is_last else quiet_cmd)
    for tls_id in config:
        m.apply_phase_merge(tls_id)

    reset_episode_state(runners, config)
    target_runner = runners[TARGET_TLS]
    target_agent = target_runner.agent

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

            current_state = agent.get_state()

            if r.is_target and r.pending_train is not None:
                prev_state, prev_action = r.pending_train
                reward = agent.get_reward()
                agent.memory.append((prev_state, prev_action, reward, current_state))
                agent.train()
                agent.maybe_update_target(step_counter[0])
                agent.decay_epsilon()
                step_counter[0] += 1

            action = agent.choose_action(current_state)

            if r.is_target:
                r.pending_train = (current_state, action)

            if action != agent.current_phase:
                current = traci.trafficlight.getPhase(tls_id)
                traci.trafficlight.setPhase(tls_id, current + 1)  # yellow
                r.pending_action = action
                r.transition_remaining = YELLOW_STEPS
            else:
                traci.trafficlight.setPhase(tls_id, agent.current_phase)

    traci.close()


if __name__ == "__main__":
    config = m.discover_controllable_tls()
    neighbor_topology = nt.discover_neighbor_inflows(config)
    runners = build_runners(config, neighbor_topology)
    step_counter = [0]

    print(f"\n### tag='{TAG}' episodes={NUM_EPISODES} weights={REWARD_WEIGHTS} "
          f"start_epsilon={START_EPSILON} epsilon_min={EPSILON_MIN} "
          f"explore_fraction={EXPLORE_FRACTION} on {TARGET_TLS} "
          f"(vs 7 real frozen DQN neighbors) ###\n")

    for ep in range(NUM_EPISODES):
        is_last = (ep == NUM_EPISODES - 1)
        run_episode(runners, config, is_last, step_counter)
        target_agent = runners[TARGET_TLS].agent
        print(f"Episode {ep} done | epsilon={target_agent.epsilon:.3f} | "
              f"memory={len(target_agent.memory)}")
        sys.stdout.flush()

    safe = TARGET_TLS.replace("/", "_")
    torch.save(runners[TARGET_TLS].agent.policy_net.state_dict(), f"dqn_policy_seq_{TAG}{safe}.pt")
    print(f"\n=== train_3more_vs_dqn_neighbors.py finished: {TARGET_TLS} ===")
