import os
import sys
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import libsumo as traci


# ============================================================
# SUMO SETUP
# ============================================================

if "SUMO_HOME" in os.environ:

    tools = os.path.join(
        os.environ["SUMO_HOME"],
        "tools"
    )

    sys.path.append(tools)

else:
    sys.exit("SUMO_HOME not set")


CONFIG_FILE = "junction_m33.sumocfg"

# Fixed seed so baseline and DQN use the same traffic realization.
SUMO_SEED = 42


# ============================================================
# TRAFFIC LIGHT
# ============================================================

# M3 CENTER JUNCTION
#
# This is the joined TLS consisting of:
#
#   13075780553
#   1839764547
#   343322679
#
# The joined TLS is represented by:
#
#   343322679
#
TLS_ID = "343322679"


# The center junction has four phases:
#
# Phase 0: GGrrrGGG
# Phase 1: yyrrryyy
# Phase 2: rrGGGrrr
# Phase 3: rryyyrrr
#
# Therefore the two actual green decisions are:
#
#   0 -> first traffic movement
#   2 -> second traffic movement
#
GREEN_PHASES = [0, 2]


# ============================================================
# M3 LANE GROUPS
# ============================================================
#
# The joined TLS has 8 controlled links:
#
# Link 0 -> 1315288638#0_0
# Link 1 -> 1315288638#0_1
# Link 2 -> 376662041#12_0
# Link 3 -> 376662041#12_1
# Link 4 -> 376662041#12_2
# Link 5 -> 617589873#0_0
# Link 6 -> 617589873#0_1
# Link 7 -> 617589873#0_2
#
# Phase 0 = GGrrrGGG
# Therefore Phase 0 controls:
#
#   0, 1, 5, 6, 7
#
# Phase 2 = rrGGGrrr
# Therefore Phase 2 controls:
#
#   2, 3, 4
#
# These groups MUST match the actual signal phases.

GROUP_A = [0, 1, 5, 6, 7]
GROUP_B = [2, 3, 4]


# ============================================================
# SIMULATION
# ============================================================

SIMULATION_TIME = 3600

NUM_EPISODES = 30


# ============================================================
# SIGNAL CONSTRAINTS
# ============================================================

MIN_GREEN = 20
MAX_GREEN = 60
MAX_RED = 60

YELLOW_TIME = 5


# ============================================================
# DQN
# ============================================================

gamma = 0.95

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.997

learning_rate = 0.001

batch_size = 64

memory_size = 10000

target_update_freq = 100


# ============================================================
# REPLAY MEMORY
# ============================================================

memory = deque(
    maxlen=memory_size
)


# ============================================================
# NETWORK
# ============================================================

class DQN(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc = nn.Sequential(

            nn.Linear(5, 64),

            nn.ReLU(),

            nn.Linear(64, 64),

            nn.ReLU(),

            nn.Linear(64, 2)

        )

    def forward(self, x):

        return self.fc(x)


policy_net = DQN()

target_net = DQN()

target_net.load_state_dict(
    policy_net.state_dict()
)

target_net.eval()


optimizer = optim.Adam(
    policy_net.parameters(),
    lr=learning_rate
)

loss_fn = nn.MSELoss()


# ============================================================
# CONTROLLER VARIABLES
# ============================================================

current_phase = 0

phase_time = 0

red_time = {
    0: 0,
    2: 0
}


# ============================================================
# SUMO COMMAND
# ============================================================

def start_sumo(record=False):

    if record:

        command = [

            "sumo",

            "-c",
            CONFIG_FILE,

            "--seed",
            str(SUMO_SEED),

            "--tripinfo-output",
            "dqn42_m33_tripinfo.xml",

            "--summary",
            "dqn42_m33_summary.xml",

            "--queue-output",
            "dqn42_m33_queue.xml",

            "--no-step-log",

            "--duration-log.disable"

        ]

    else:

        command = [

            "sumo",

            "-c",
            CONFIG_FILE,

            "--seed",
            str(SUMO_SEED),

            "--no-step-log",

            "--duration-log.disable"

        ]

    traci.start(command)


# ============================================================
# QUEUES
# ============================================================

def get_lane_queues():

    lanes = traci.trafficlight.getControlledLanes(
        TLS_ID
    )

    # M3 center TLS must have exactly 8 controlled lanes.
    if len(lanes) != 8:

        raise RuntimeError(
            f"Expected 8 controlled lanes for "
            f"TLS {TLS_ID}, "
            f"found {len(lanes)}"
        )

    return [

        traci.lane.getLastStepHaltingNumber(
            lane
        )

        for lane in lanes

    ]


def get_group_data():

    queues = get_lane_queues()

    group_a_total = sum(
        queues[i]
        for i in GROUP_A
    )

    group_b_total = sum(
        queues[i]
        for i in GROUP_B
    )

    group_a_avg = (
        group_a_total / len(GROUP_A)
    )

    group_b_avg = (
        group_b_total / len(GROUP_B)
    )

    return (
        group_a_total,
        group_b_total,
        group_a_avg,
        group_b_avg
    )


def get_total_queue():

    queues = get_lane_queues()

    return sum(queues)


# ============================================================
# STATE
# ============================================================

def get_state(
    recent_arrivals,
    running_vehicles
):

    (
        group_a_total,
        group_b_total,
        group_a_avg,
        group_b_avg
    ) = get_group_data()

    phase = traci.trafficlight.getPhase(
        TLS_ID
    )

    #
    # Five inputs:
    #
    # 1. Group A average queue
    # 2. Group B average queue
    # 3. Recent network arrivals
    # 4. Current running vehicles
    # 5. Current phase
    #

    state = np.array(

        [

            group_a_avg / 20.0,

            group_b_avg / 20.0,

            recent_arrivals / 10.0,

            running_vehicles / 1000.0,

            phase / 3.0

        ],

        dtype=np.float32

    )

    return state


# ============================================================
# ACTION SELECTION
# ============================================================

def choose_action(state):

    global current_phase
    global phase_time

    #
    # Minimum green.
    #

    if phase_time < MIN_GREEN:

        return current_phase


    #
    # Maximum green.
    #

    if phase_time >= MAX_GREEN:

        if current_phase == 0:

            return 2

        else:

            return 0


    #
    # Maximum red / starvation protection.
    #

    if current_phase == 0:

        if red_time[2] >= MAX_RED:

            return 2

    elif current_phase == 2:

        if red_time[0] >= MAX_RED:

            return 0


    #
    # Epsilon-greedy.
    #

    if random.random() < epsilon:

        return random.choice(
            GREEN_PHASES
        )


    #
    # DQN.
    #

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32
    )

    with torch.no_grad():

        q_values = policy_net(
            state_tensor
        )

    action_index = int(
        torch.argmax(q_values).item()
    )

    return GREEN_PHASES[
        action_index
    ]


# ============================================================
# APPLY ACTION
# ============================================================

def apply_action(action):

    global current_phase
    global phase_time
    global red_time

    current = traci.trafficlight.getPhase(
        TLS_ID
    )

    #
    # We only ever request green phase 0 or 2.
    #

    if action not in GREEN_PHASES:

        raise RuntimeError(
            f"Invalid action: {action}"
        )


    #
    # Switch phase.
    #

    if current != action:

        #
        # 0 -> 1 -> 2
        #

        if action == 2:

            traci.trafficlight.setPhase(
                TLS_ID,
                1
            )

            for _ in range(
                YELLOW_TIME
            ):

                traci.simulationStep()


        #
        # 2 -> 3 -> 0
        #

        else:

            traci.trafficlight.setPhase(
                TLS_ID,
                3
            )

            for _ in range(
                YELLOW_TIME
            ):

                traci.simulationStep()


        #
        # Activate new green.
        #

        traci.trafficlight.setPhase(
            TLS_ID,
            action
        )

        current_phase = action

        phase_time = 0


    #
    # Advance 5 seconds.
    #

    for _ in range(5):

        traci.simulationStep()

        phase_time += 1


    #
    # Update red timers.
    #

    if current_phase == 0:

        red_time[0] = 0
        red_time[2] += 5

    else:

        red_time[2] = 0
        red_time[0] += 5


# ============================================================
# TRAINING
# ============================================================

def train_network():

    if len(memory) < batch_size:

        return


    batch = random.sample(
        memory,
        batch_size
    )


    states, actions, rewards, next_states = zip(
        *batch
    )


    states = torch.tensor(
        np.array(states),
        dtype=torch.float32
    )

    next_states = torch.tensor(
        np.array(next_states),
        dtype=torch.float32
    )

    rewards = torch.tensor(
        rewards,
        dtype=torch.float32
    )


    #
    # Current Q values.
    #

    q_values = policy_net(
        states
    )


    #
    # Target Q values.
    #

    with torch.no_grad():

        next_q_values = target_net(
            next_states
        )


    targets = q_values.clone().detach()


    for i in range(batch_size):

        action_index = GREEN_PHASES.index(
            actions[i]
        )

        targets[i][action_index] = (

            rewards[i]

            +

            gamma
            *
            torch.max(
                next_q_values[i]
            )

        )


    loss = loss_fn(
        q_values,
        targets
    )


    optimizer.zero_grad()

    loss.backward()


    #
    # Prevent extremely large updates.
    #

    torch.nn.utils.clip_grad_norm_(
        policy_net.parameters(),
        1.0
    )

    optimizer.step()


# ============================================================
# REWARD
# ============================================================

def calculate_reward(
    previous_queue,
    current_queue,
    arrived_delta,
    current_phase,
    phase_time
):

    #
    # 1. Queue improvement.
    #
    # Positive when queue decreases.
    #

    queue_reward = (

        previous_queue
        -
        current_queue

    ) / 10.0


    #
    # 2. Throughput reward.
    #
    # Reward vehicles that actually complete
    # their trips.
    #

    throughput_reward = (
        arrived_delta * 1.0
    )


    #
    # 3. Mild congestion penalty.
    #
    # Prevents the agent from exploiting tiny queue
    # reductions while allowing a large queue to build.
    #

    congestion_penalty = (
        current_queue / 200.0
    )


    #
    # 4. Mild starvation penalty.
    #
    # If an approach has been red for a long time,
    # discourage continuing to ignore it.
    #

    if current_phase == 0:

        starving_time = red_time[2]

    else:

        starving_time = red_time[0]


    starvation_penalty = 0.0


    if starving_time > 40:

        starvation_penalty = (
            (starving_time - 40)
            / 40.0
        )


    reward = (

        queue_reward

        +

        throughput_reward

        -

        congestion_penalty

        -

        starvation_penalty

    )


    return reward


# ============================================================
# TRAINING EPISODE
# ============================================================

def run_training_episode(
    episode
):

    global epsilon
    global current_phase
    global phase_time
    global red_time


    current_phase = 0

    phase_time = 0

    red_time = {
        0: 0,
        2: 0
    }


    start_sumo(
        record=False
    )


    previous_queue = get_total_queue()

    previous_arrived = (
        traci.simulation.getArrivedNumber()
    )


    step = 0


    while (
        traci.simulation.getTime()
        <
        SIMULATION_TIME
    ):


        running = (
            traci.simulation.getDepartedNumber()
            -
            traci.simulation.getArrivedNumber()
        )


        state = get_state(
            recent_arrivals=0,
            running_vehicles=running
        )


        action = choose_action(
            state
        )


        apply_action(
            action
        )


        current_queue = get_total_queue()


        current_arrived = (
            traci.simulation.getArrivedNumber()
        )


        arrived_delta = (
            current_arrived
            -
            previous_arrived
        )


        running = (
            traci.simulation.getDepartedNumber()
            -
            traci.simulation.getArrivedNumber()
        )


        reward = calculate_reward(

            previous_queue,

            current_queue,

            arrived_delta,

            current_phase,

            phase_time

        )


        next_state = get_state(

            recent_arrivals=arrived_delta,

            running_vehicles=running

        )


        memory.append(

            (

                state,

                action,

                reward,

                next_state

            )

        )


        train_network()


        if step % target_update_freq == 0:

            target_net.load_state_dict(
                policy_net.state_dict()
            )


        #
        # Slowly decay exploration.
        #

        if epsilon > epsilon_min:

            epsilon *= epsilon_decay

            epsilon = max(
                epsilon,
                epsilon_min
            )


        if step % 60 == 0:

            print(

                f"Episode {episode} | "
                f"Time {traci.simulation.getTime():.0f}s | "
                f"Phase {current_phase} | "
                f"Queue {current_queue} | "
                f"Arrived +{arrived_delta} | "
                f"Reward {reward:.3f} | "
                f"Epsilon {epsilon:.3f}"

            )


        previous_queue = current_queue

        previous_arrived = current_arrived

        step += 1


    traci.close()


# ============================================================
# EVALUATION
# ============================================================

def run_evaluation():

    global epsilon
    global current_phase
    global phase_time
    global red_time


    epsilon = 0.0

    current_phase = 0

    phase_time = 0

    red_time = {
        0: 0,
        2: 0
    }


    #
    # IMPORTANT:
    #
    # This creates:
    #
    # dqn42_m33_tripinfo.xml
    # dqn42_m33_summary.xml
    # dqn42_m33_queue.xml
    #

    start_sumo(
        record=True
    )


    total_reward = 0.0
    phase_counts = {
        0: 0,
        2: 0
    }

    phase_switches = 0
    previous_action = None

    evaluation_queue_sum = 0
    evaluation_steps = 0

    previous_queue = get_total_queue()

    previous_arrived = (
        traci.simulation.getArrivedNumber()
    )


    step = 0


    while (
        traci.simulation.getTime()
        <
        SIMULATION_TIME
    ):


        running = (
            traci.simulation.getDepartedNumber()
            -
            traci.simulation.getArrivedNumber()
        )


        state = get_state(

            recent_arrivals=0,

            running_vehicles=running

        )


        action = choose_action(
            state
        )
        phase_counts[action] += 1

        if previous_action is not None and action != previous_action:
            phase_switches += 1

        previous_action = action


        apply_action(
            action
        )


        current_queue = get_total_queue()
        evaluation_queue_sum += current_queue
        evaluation_steps += 1

        current_arrived = (
            traci.simulation.getArrivedNumber()
        )


        arrived_delta = (
            current_arrived
            -
            previous_arrived
        )


        running = (
            traci.simulation.getDepartedNumber()
            -
            traci.simulation.getArrivedNumber()
        )


        reward = calculate_reward(

            previous_queue,

            current_queue,

            arrived_delta,

            current_phase,

            phase_time

        )


        total_reward += reward


        if step % 60 == 0:

            print(

                f"Evaluation | "
                f"Time {traci.simulation.getTime():.0f}s | "
                f"Phase {current_phase} | "
                f"Queue {current_queue} | "
                f"Arrived +{arrived_delta} | "
                f"Reward {reward:.3f}"

            )


        previous_queue = current_queue

        previous_arrived = current_arrived

        step += 1

    print()
    print("=" * 60)
    print("DQN POLICY DIAGNOSTICS")
    print("=" * 60)

    print(
        f"Phase 0 decisions: {phase_counts[0]}"
    )

    print(
        f"Phase 2 decisions: {phase_counts[2]}"
    )

    print(
        f"Phase switches: {phase_switches}"
    )

    if evaluation_steps > 0:
        print(
            f"Average observed queue: "
            f"{evaluation_queue_sum / evaluation_steps:.3f}"
        )

    print("=" * 60)


    traci.close()


    print()

    print("=" * 60)

    print("DQN EVALUATION COMPLETE")

    print("=" * 60)

    print(
        f"Total evaluation reward: "
        f"{total_reward:.3f}"
    )

    print()

    print("Created:")

    print("  dqn42_m33_tripinfo.xml")

    print("  dqn42_m33_summary.xml")

    print("  dqn42_m33_queue.xml")

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    global epsilon


    print()

    print("=" * 60)

    print("DQN TRAFFIC LIGHT OPTIMIZATION")

    print("=" * 60)

    print()

    print(
        f"TLS: {TLS_ID}"
    )

    print(
        f"Seed: {SUMO_SEED}"
    )

    print(
        f"Training episodes: {NUM_EPISODES}"
    )

    print(
        f"Simulation time: {SIMULATION_TIME}s"
    )

    print()


    #
    # Verify TLS before training.
    #

    start_sumo(
        record=False
    )


    tls_ids = traci.trafficlight.getIDList()


    if TLS_ID not in tls_ids:

        traci.close()

        raise RuntimeError(

            f"TLS '{TLS_ID}' not found.\n"
            f"Available TLS IDs:\n"
            f"{tls_ids}"

        )


    print(
        "TLS verified successfully."
    )


    lanes = traci.trafficlight.getControlledLanes(
        TLS_ID
    )


    print(
        "Controlled lanes:"
    )


    for lane in lanes:

        print(
            f"  {lane}"
        )


    traci.close()


    #
    # TRAINING
    #

    for episode in range(
        NUM_EPISODES
    ):

        run_training_episode(
            episode
        )


        print()

        print(
            f"Training episode "
            f"{episode} complete."
        )

        print()


    #
    # Save trained model.
    #

    torch.save(

        policy_net.state_dict(),

        "dqn42_m33_policy.pt"

    )


    print(
        "Model saved: "
        "dqn42_m33_policy.pt"
    )


    #
    # FINAL EVALUATION
    #
    policy_net.load_state_dict(
        torch.load(
            "dqn42_m33_policy.pt",
            map_location="cpu"
        )
    )

    target_net.load_state_dict(
        policy_net.state_dict()
    )
    run_evaluation()


if __name__ == "__main__":

    main()