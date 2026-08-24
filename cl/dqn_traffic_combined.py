import os
import sys
import traci
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ------------------------
# Reproducibility
# ------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------------
# SUMO setup
# ------------------------
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not set")

gamma = 0.95
epsilon_min = 0.05
epsilon_decay = 0.995
lr = 0.001
batch_size = 64
memory_size = 5000
target_update_freq = 50

MIN_GREEN = 20
MAX_GREEN = 60
MIN_IMBALANCE = 3 / 20.0

NUM_EPISODES = 10

DISCOVER_CMD = ["sumo", "-c", "combined.sumocfg", "--no-step-log",
                "--duration-log.disable", "--seed", str(SEED)]


def discover_controllable_tls():
    """
    Inspect every traffic light in the network and keep the ones that
    actually have a decision to make: 2 or more real green phases
    (a phase whose state has no 'y'/'Y' and at least one 'G'/'g').
    Junctions with only one green phase have nothing for an RL agent
    to choose between and are left on their default program.

    For each kept junction, also compute which controlled lanes belong
    to each green phase (the lanes that are 'G'/'g' in that phase's
    state string), so the state/reward can be generalized to N phases
    instead of assuming exactly two directions.
    """
    traci.start(DISCOVER_CMD)
    config = {}
    for tls_id in traci.trafficlight.getIDList():
        logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        phases = logic.phases
        green_phase_indices = [
            i for i, p in enumerate(phases)
            if 'y' not in p.state.lower() and 'g' in p.state.lower()
        ]
        if len(green_phase_indices) < 2:
            continue

        lanes = traci.trafficlight.getControlledLanes(tls_id)
        lane_groups = []
        for gp in green_phase_indices:
            state = phases[gp].state
            group = sorted({lane for lane, ch in zip(lanes, state) if ch in ('G', 'g')})
            lane_groups.append(group)

        config[tls_id] = {
            "green_phases": green_phase_indices,
            "lane_groups": lane_groups,
        }
    traci.close()
    return config


# ------------------------
# Neural Network
# ------------------------
class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_actions + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.fc(x)

# ------------------------
# Same agent logic proven on the earlier 3-junction map (fairness-
# override fix needing a real 3-vehicle imbalance, not just a ratio,
# min/max green constraints), generalized from a fixed 2-way choice
# to an arbitrary number of green phases per junction.
# ------------------------
class SingleJunctionAgent:
    def __init__(self, tls_id, green_phases, lane_groups):
        self.tls_id = tls_id
        self.green_phases = green_phases
        self.lane_groups = lane_groups
        self.n_actions = len(green_phases)

        self.policy_net = DQN(self.n_actions)
        self.target_net = DQN(self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=memory_size)
        self.current_phase = green_phases[0]
        self.phase_time = 0
        self.epsilon = 1.0
        # How long (in the same units as phase_time) each phase has gone
        # without being served. Used to force a phase to get its turn
        # instead of letting a persistently busy phase win the queue-based
        # comparison indefinitely - a real risk once there are 3+ phases
        # competing, since "switch to whoever has the biggest queue" alone
        # has no bound on how long a smaller-but-steady queue can be skipped.
        self.wait_since_served = [0] * self.n_actions

    def get_state(self):
        queues = [sum(traci.lane.getLastStepHaltingNumber(l) for l in group)
                  for group in self.lane_groups]
        phase = traci.trafficlight.getPhase(self.tls_id)
        state = [q / 20.0 for q in queues] + [phase / 10.0]
        return np.array(state, dtype=np.float32)

    def get_reward(self):
        lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
        wait = sum(traci.lane.getWaitingTime(l) for l in lanes)
        return -(queue / 100.0 + wait / 1000.0)

    def choose_action(self, state):
        queues = state[:-1]
        current_idx = self.green_phases.index(self.current_phase)

        # Fairness bound: no phase WITH ACTUAL DEMAND should wait longer
        # than every other phase getting a full MAX_GREEN turn ahead of
        # it. Requiring queues[i] > 0 avoids burning a green+yellow cycle
        # on a phase nobody is waiting on - that wasted-cycle case is what
        # regressed the 3-action junction when this first tried an
        # unconditional starvation check. For a 2-action junction the
        # bound is exactly MAX_GREEN, and the pre-existing unconditional
        # "phase_time > MAX_GREEN" check below still forces the same
        # switch even when the other phase is empty, so 2-action behavior
        # is unchanged either way.
        fairness_bound = MAX_GREEN * (self.n_actions - 1)
        starved = [i for i in range(self.n_actions)
                   if i != current_idx and self.wait_since_served[i] > fairness_bound
                   and queues[i] > 0]
        if starved:
            target_idx = max(starved, key=lambda i: self.wait_since_served[i])
            return self.green_phases[target_idx]

        cur_q = queues[current_idx]
        best_idx = int(np.argmax(queues))
        best_q = queues[best_idx]

        significant_imbalance = (
            best_idx != current_idx and
            best_q > cur_q * 1.5 and
            (best_q - cur_q) >= MIN_IMBALANCE
        )

        if self.phase_time < MIN_GREEN and not significant_imbalance:
            return self.current_phase
        if self.phase_time > MAX_GREEN:
            target_idx = best_idx if best_idx != current_idx else (current_idx + 1) % self.n_actions
            return self.green_phases[target_idx]
        if significant_imbalance:
            return self.green_phases[best_idx]
        if random.random() < self.epsilon:
            return random.choice(self.green_phases)
        state_t = torch.tensor(state, dtype=torch.float32)
        q_vals = self.policy_net(state_t)
        return self.green_phases[int(torch.argmax(q_vals))]

    def apply_action(self, action):
        current = traci.trafficlight.getPhase(self.tls_id)
        if current != action:
            yellow_phase = current + 1
            traci.trafficlight.setPhase(self.tls_id, yellow_phase)
            for _ in range(5):
                traci.simulationStep()
            self.current_phase = action
            self.phase_time = 0
        traci.trafficlight.setPhase(self.tls_id, action)
        for _ in range(5):
            traci.simulationStep()
            self.phase_time += 1

        action_idx = self.green_phases.index(action)
        for i in range(self.n_actions):
            self.wait_since_served[i] = 0 if i == action_idx else self.wait_since_served[i] + 5

    def train(self):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        q_vals = self.policy_net(states)
        next_q_vals = self.target_net(next_states).detach()
        target = q_vals.clone()
        for i in range(batch_size):
            action_index = self.green_phases.index(actions[i])
            target[i][action_index] = rewards[i] + gamma * torch.max(next_q_vals[i])
        loss = self.loss_fn(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        if self.epsilon > epsilon_min:
            self.epsilon *= epsilon_decay

    def maybe_update_target(self, step):
        if step % target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

# ------------------------
# Train ONE junction for NUM_EPISODES, others stay on default program
# ------------------------
def train_one_junction(tls_id, green_phases, lane_groups):
    safe_name = tls_id.replace("/", "_")
    agent = SingleJunctionAgent(tls_id, green_phases, lane_groups)

    record_cmd = ["sumo", "-c", "combined.sumocfg",
                  "--tripinfo-output", f"seq_after_tripinfo_{safe_name}.xml",
                  "--summary", f"seq_after_summary_{safe_name}.xml",
                  "--queue-output", f"seq_after_queue_{safe_name}.xml",
                  "--no-step-log", "--duration-log.disable",
                  "--seed", str(SEED)]
    quiet_cmd = ["sumo", "-c", "combined.sumocfg", "--no-step-log", "--duration-log.disable",
                 "--seed", str(SEED)]

    for ep in range(NUM_EPISODES):
        is_last = (ep == NUM_EPISODES - 1)
        agent.current_phase = green_phases[0]
        agent.phase_time = 0
        agent.wait_since_served = [0] * agent.n_actions

        traci.start(record_cmd if is_last else quiet_cmd)

        step = 0
        while traci.simulation.getTime() < 3600:
            state = agent.get_state()
            action = agent.choose_action(state)
            agent.apply_action(action)

            next_state = agent.get_state()
            reward = agent.get_reward()

            agent.memory.append((state, action, reward, next_state))
            agent.train()
            agent.maybe_update_target(step)
            agent.decay_epsilon()

            if step % 60 == 0:
                print(f"[{tls_id[:20]}] Episode {ep} | Time {traci.simulation.getTime():.1f} | "
                      f"Reward={reward:.3f} | Epsilon={agent.epsilon:.3f}")
            step += 1

        traci.close()

    torch.save(agent.policy_net.state_dict(), f"dqn_policy_seq_{safe_name}.pt")
    print(f"== Finished {tls_id} ({agent.n_actions} actions) ==\n")

def run():
    config = discover_controllable_tls()
    print(f"Discovered {len(config)} controllable junctions (>=2 real green phases):")
    for tls_id, info in config.items():
        print(f"  {tls_id} -> {len(info['green_phases'])} actions")

    for tls_id, info in config.items():
        train_one_junction(tls_id, info["green_phases"], info["lane_groups"])

if __name__ == "__main__":
    run()

# ------------------------
# NOTE on junctions NOT included here:
# The combined map has 18 signalised junctions total. 10 of them have
# only ONE real green phase - meaning there's nothing for an RL agent
# to choose between, so they're left on their default program. The
# remaining 8 are discovered and trained automatically above, whether
# they have 2, 3, or 6 real green phases.
# ------------------------
