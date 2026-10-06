"""
Federated-learning variant of SingleJunctionAgent (dqn_traffic_combined.py).

Standard FedAvg needs every client's model to be the same shape so their
weights can be averaged - this project's 8 junctions aren't: n_actions
ranges 2-5, neighbor count 0-2, and 2 junctions (343322679, 481750173)
use a richer 4-feature-per-phase state while the other 6 use a thin
1-feature one (see RICHER_STATE_TLS in dqn_traffic_combined.py).

Fix: give every junction the SAME padded architecture.
  - MAX_ACTIONS = 5 (from #3more, the largest junction) - every DQN has
    a 5-wide output layer and a richer per-phase state block, even for a
    2-phase junction. Phases past this junction's real n_actions are
    zero-padded in the state and simply never read from or written to in
    the output (see FederatedJunctionAgent.choose_action/train - both
    slice to [:self.n_actions] before argmax/target assignment, so a
    padding slot's Q-value never influences behavior and never gets a
    gradient: MSE loss on an untouched slot is target==prediction, i.e.
    exactly zero).
  - MAX_NEIGHBORS = 2 (already the max any junction has) - neighbor
    inflow features are zero-padded the same way.
  - Every junction computes the RICHER state (queue, wait, speed, trend
    per phase) regardless of its original RICHER_STATE_TLS membership -
    thin-state junctions can compute these exactly as easily, they just
    didn't before. This is a deliberate divergence from the canonical
    per-junction checkpoints (dqn_policy_seq_*.pt), which are untouched
    by this file - federated checkpoints are saved under a distinct
    'fed_' naming scheme in train_federated.py, never overwriting them.

This file only defines the agent/state representation and one local
training episode (local_train_episode). The FedAvg round loop and
weight averaging live in train_federated.py.
"""
import random
from collections import deque

import numpy as np
import torch
import torch.optim as optim
import traci

import dqn_traffic_combined as m

MAX_ACTIONS = 5
MAX_NEIGHBORS = 2
STATE_SIZE = MAX_ACTIONS * 4 + 2 + MAX_NEIGHBORS  # = 24
HIDDEN_SIZE = 128


class FederatedJunctionAgent:
    def __init__(self, tls_id, green_phases, lane_groups, neighbor_specs=None,
                 queue_weight=0.01, wait_weight=0.001, queue_power=1,
                 epsilon_decay=m.epsilon_decay, delta_weight=0.02):
        self.tls_id = tls_id
        self.green_phases = green_phases
        self.lane_groups = lane_groups
        self.n_actions = len(green_phases)  # this junction's REAL action count
        self.neighbor_specs = neighbor_specs or []
        self.n_neighbors = len(self.neighbor_specs)
        if self.n_actions > MAX_ACTIONS:
            raise ValueError(f"{tls_id} has {self.n_actions} actions > MAX_ACTIONS={MAX_ACTIONS}")
        if self.n_neighbors > MAX_NEIGHBORS:
            raise ValueError(f"{tls_id} has {self.n_neighbors} neighbors > MAX_NEIGHBORS={MAX_NEIGHBORS}")

        self.queue_weight = queue_weight
        self.wait_weight = wait_weight
        self.queue_power = queue_power
        self.delta_weight = delta_weight
        self.prev_queue = 0
        self.prev_queues = [0] * self.n_actions

        self.state_size = STATE_SIZE
        self.policy_net = m.DQN(MAX_ACTIONS, STATE_SIZE, HIDDEN_SIZE)
        self.target_net = m.DQN(MAX_ACTIONS, STATE_SIZE, HIDDEN_SIZE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=m.lr)
        self.loss_fn = torch.nn.MSELoss()
        self.memory = deque(maxlen=m.memory_size)
        self.current_phase = green_phases[0]
        self.phase_time = 0
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.wait_since_served = [0] * self.n_actions
        self.min_green, self.max_green = m.green_bounds_for(tls_id)
        self.min_imbalance, self.imbalance_ratio = m.imbalance_for(tls_id)

    def _neighbor_outflow_features(self):
        raw = [sum(traci.lane.getLastStepVehicleNumber(l) for l in lanes) / m.NEIGHBOR_OUTFLOW_SCALE
               for _, lanes in self.neighbor_specs]
        return raw + [0.0] * (MAX_NEIGHBORS - len(raw))

    def get_state(self):
        state = []
        for i in range(MAX_ACTIONS):
            if i < self.n_actions:
                group = self.lane_groups[i]
                queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in group)
                wait = sum(traci.lane.getWaitingTime(l) for l in group)
                speeds = [traci.lane.getLastStepMeanSpeed(l) for l in group]
                speed = sum(speeds) / len(speeds) if speeds else 0.0
                trend = self.prev_queues[i] - queue
                self.prev_queues[i] = queue
                state.extend([queue / 20.0, wait / 300.0, speed / 15.0, trend / 20.0])
            else:
                state.extend([0.0, 0.0, 0.0, 0.0])
        phase = traci.trafficlight.getPhase(self.tls_id)
        state.append(phase / 10.0)
        state.append(self.phase_time / m.MAX_GREEN)
        state.extend(self._neighbor_outflow_features())
        return np.array(state, dtype=np.float32)

    def get_reward(self):
        lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
        wait = sum(traci.lane.getWaitingTime(l) for l in lanes)
        base = -(self.queue_weight * (queue ** self.queue_power) + self.wait_weight * wait)
        shaping = self.delta_weight * (self.prev_queue - queue)
        self.prev_queue = queue
        return base + shaping

    def choose_action(self, state):
        queues = [state[i * 4] for i in range(self.n_actions)]
        current_idx = self.green_phases.index(self.current_phase)

        fairness_bound = self.max_green * (self.n_actions - 1)
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
            best_q > cur_q * self.imbalance_ratio and
            (best_q - cur_q) >= self.min_imbalance
        )

        if self.phase_time < self.min_green and not significant_imbalance:
            return self.current_phase
        if self.phase_time > self.max_green:
            target_idx = best_idx if best_idx != current_idx else (current_idx + 1) % self.n_actions
            return self.green_phases[target_idx]
        if significant_imbalance:
            return self.green_phases[best_idx]
        if random.random() < self.epsilon:
            return random.choice(self.green_phases)
        state_t = torch.tensor(state, dtype=torch.float32)
        q_vals = self.policy_net(state_t)[:self.n_actions]  # padding slots never considered
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
        if len(self.memory) < m.batch_size:
            return
        batch = random.sample(self.memory, m.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        q_vals = self.policy_net(states)
        # Double DQN, restricted to this client's real action slots so a
        # padding slot can never be selected as the "best next action" -
        # see module docstring for why untouched padding slots get zero
        # gradient automatically (target == prediction there).
        next_actions = self.policy_net(next_states).detach()[:, :self.n_actions].argmax(dim=1)
        next_q_vals_target = self.target_net(next_states).detach()
        target = q_vals.clone()
        for i in range(m.batch_size):
            action_index = self.green_phases.index(actions[i])
            target[i][action_index] = rewards[i] + m.gamma * next_q_vals_target[i][next_actions[i]]
        loss = self.loss_fn(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        if self.epsilon > m.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def maybe_update_target(self, step):
        if step % m.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


QUIET_CMD = ["sumo", "-c", "combined.sumocfg", "--no-step-log", "--duration-log.disable",
             "--seed", str(m.SEED)]


def local_train_episode(agent, bystander_config, port=None):
    """One local-training episode for ONE federated client, reusing the
    exact simulation/bystander setup train_one_junction() uses (same
    SUMO seed, same realistic fixed-time bystanders). Resets per-episode
    bookkeeping the same way. Epsilon is NOT reset - it keeps decaying
    across the whole federated run, round boundaries included.

    port: explicit per-client SUMO port, so 8 clients starting in
    parallel processes never race for the same auto-picked free port.

    Returns the episode's total reward (for progress logging only)."""
    agent.current_phase = agent.green_phases[0]
    agent.phase_time = 0
    agent.wait_since_served = [0] * agent.n_actions
    agent.prev_queue = 0
    agent.prev_queues = [0] * agent.n_actions

    traci.start(QUIET_CMD, port=port)
    m.apply_phase_merge(agent.tls_id)
    if bystander_config is not None:
        m.install_realistic_bystanders(bystander_config, agent.tls_id)

    step = 0
    total_reward = 0.0
    while traci.simulation.getTime() < 3600:
        state = agent.get_state()
        action = agent.choose_action(state)
        agent.apply_action(action)

        next_state = agent.get_state()
        reward = agent.get_reward()
        total_reward += reward

        agent.memory.append((state, action, reward, next_state))
        agent.train()
        agent.maybe_update_target(step)
        agent.decay_epsilon()
        step += 1
    traci.close()
    return total_reward
