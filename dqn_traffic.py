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
# SUMO setup
# ------------------------
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not set")

SUMO_CMD_RECORD = ["sumo", "-c", "junction.sumocfg",
            "--tripinfo-output", "after_tripinfo.xml",
            "--summary", "after_summary.xml",
            "--queue-output", "after_queue.xml",
            "--no-step-log", "--duration-log.disable"]
SUMO_CMD_QUIET = ["sumo", "-c", "junction.sumocfg",
            "--no-step-log", "--duration-log.disable"]

# Main junction on the new map (highest-degree signalised cluster)
TLS_ID = "cluster_11594569878_11909580134_3352035441_6901551440"
GREEN_PHASES = [0, 2]

# ------------------------
# Hyperparameters (same as Phase 2 / Silk Board run)
# ------------------------
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
lr = 0.001
batch_size = 64
memory_size = 5000
target_update_freq = 50

MIN_GREEN = 20
MAX_GREEN = 60

NUM_EPISODES = 8  # epsilon carries across episodes so the agent gets to exploit by the end

# ------------------------
# Replay Buffer
# ------------------------
memory = deque(maxlen=memory_size)

# ------------------------
# Neural Network
# ------------------------
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.fc(x)

policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# ------------------------
# State
# ------------------------
def get_state():
    lanes = traci.trafficlight.getControlledLanes(TLS_ID)
    queues = [traci.lane.getLastStepHaltingNumber(l) for l in lanes]

    half = len(queues)//2
    dir1 = sum(queues[:half])
    dir2 = sum(queues[half:])

    phase = traci.trafficlight.getPhase(TLS_ID)

    return np.array([dir1/20, dir2/20, phase/10], dtype=np.float32)

# ------------------------
# Reward
# ------------------------
def get_reward():
    lanes = traci.trafficlight.getControlledLanes(TLS_ID)
    queues = [traci.lane.getLastStepHaltingNumber(l) for l in lanes]

    return -sum(queues) / 100.0

# ------------------------
# Action selection
# ------------------------
current_phase = 0
phase_time = 0

def choose_action(state):
    global epsilon, current_phase, phase_time

    dir1, dir2, _ = state
    # 3 vehicles, expressed in the same normalized units as get_state() (raw/20)
    MIN_IMBALANCE = 3 / 20.0

    significant_imbalance = (
        (dir1 > dir2 * 1.5 and (dir1 - dir2) >= MIN_IMBALANCE) or
        (dir2 > dir1 * 1.5 and (dir2 - dir1) >= MIN_IMBALANCE)
    )

    # min green constraint takes priority over the fairness override -
    # a 1-vs-2-vehicle noise blip should never cut a green phase short
    if phase_time < MIN_GREEN and not significant_imbalance:
        return current_phase

    if phase_time > MAX_GREEN:
        return 2 if current_phase == 0 else 0

    if dir1 > dir2 * 1.5 and (dir1 - dir2) >= MIN_IMBALANCE:
        return 0
    if dir2 > dir1 * 1.5 and (dir2 - dir1) >= MIN_IMBALANCE:
        return 2

    # epsilon-greedy
    if random.random() < epsilon:
        return random.choice(GREEN_PHASES)

    state_t = torch.tensor(state, dtype=torch.float32)
    q_vals = policy_net(state_t)

    return GREEN_PHASES[int(torch.argmax(q_vals))]

# ------------------------
# Apply action
# ------------------------
def apply_action(action):
    global current_phase, phase_time

    current = traci.trafficlight.getPhase(TLS_ID)

    if current != action:
        yellow_phase = current + 1
        traci.trafficlight.setPhase(TLS_ID, yellow_phase)

        for _ in range(5):
            traci.simulationStep()

        current_phase = action
        phase_time = 0

    traci.trafficlight.setPhase(TLS_ID, action)

    for _ in range(5):
        traci.simulationStep()
        phase_time += 1

# ------------------------
# Training step
# ------------------------
def train():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)

    states, actions, rewards, next_states = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)

    q_vals = policy_net(states)
    next_q_vals = target_net(next_states).detach()

    target = q_vals.clone()

    for i in range(batch_size):
        action_index = GREEN_PHASES.index(actions[i])
        target[i][action_index] = rewards[i] + gamma * torch.max(next_q_vals[i])

    loss = loss_fn(q_vals, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ------------------------
# ONE EPISODE
# ------------------------
def run_episode(episode_idx, record_output=False):
    global epsilon, current_phase, phase_time

    current_phase = 0
    phase_time = 0

    cmd = SUMO_CMD_RECORD if record_output else SUMO_CMD_QUIET
    traci.start(cmd)

    step = 0
    while traci.simulation.getTime() < 3600:

        state = get_state()
        action = choose_action(state)

        apply_action(action)

        next_state = get_state()
        reward = get_reward()

        memory.append((state, action, reward, next_state))

        train()

        if step % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if step % 40 == 0:
            print(f"Episode {episode_idx} | Time {traci.simulation.getTime():.1f} | Reward={reward:.3f} | Epsilon={epsilon:.3f}")

        step += 1

    traci.close()

# ------------------------
# MULTI-EPISODE RUN
# ------------------------
def run():
    for ep in range(NUM_EPISODES):
        is_last = (ep == NUM_EPISODES - 1)
        run_episode(ep, record_output=is_last)
    torch.save(policy_net.state_dict(), "dqn_policy.pt")

if __name__ == "__main__":
    run()
