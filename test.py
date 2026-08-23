import os
import sys
import traci
import numpy as np
import torch
import torch.nn as nn

# ============================================================
# SUMO SETUP
# ============================================================

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not set")

SUMO_CMD = [
    "sumo-gui",
    "-c", "junction.sumocfg",

    # Output files for final DQN evaluation
    "--tripinfo-output", "test_tripinfo.xml",
    "--summary", "test_summary.xml",
    "--queue-output", "test_queue.xml",

    "--no-step-log",
    "--duration-log.disable"
]

# ============================================================
# TRAFFIC LIGHT
# ============================================================

TLS_ID = "11338711006"

# Keep this as required for the project
GREEN_PHASES = [0, 2]

# ============================================================
# DQN MODEL
# ============================================================

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


# Create the same network architecture used during training
policy_net = DQN()

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_FILE = "dqn_policy.pt"

if not os.path.exists(MODEL_FILE):
    sys.exit(
        f"ERROR: {MODEL_FILE} not found. "
        "Run dqn_traffic.py successfully first."
    )

policy_net.load_state_dict(
    torch.load(MODEL_FILE, map_location=torch.device("cpu"))
)

# Evaluation mode
policy_net.eval()

print("Trained DQN model loaded successfully.")


# ============================================================
# STATE REPRESENTATION
# ============================================================

def get_state():

    # Get lanes controlled by our traffic light
    lanes = traci.trafficlight.getControlledLanes(TLS_ID)

    # Remove duplicate lane IDs
    lanes = list(dict.fromkeys(lanes))

    # Number of stopped/halting vehicles on each lane
    queues = [
        traci.lane.getLastStepHaltingNumber(lane)
        for lane in lanes
    ]

    # Split lanes into two groups
    half = len(queues) // 2

    dir1 = sum(queues[:half])
    dir2 = sum(queues[half:])

    # Current traffic-light phase
    phase = traci.trafficlight.getPhase(TLS_ID)

    # Fixed-size state vector
    state = np.array(
        [
            dir1 / 20.0,
            dir2 / 20.0,
            phase / 10.0
        ],
        dtype=np.float32
    )

    return state


# ============================================================
# APPLY DQN ACTION
# ============================================================

current_phase = 0
phase_time = 0

# For this signal:
# Phase 1 = yellow
YELLOW_PHASE = 1


def apply_action(action):

    global current_phase
    global phase_time

    current = traci.trafficlight.getPhase(TLS_ID)

    # --------------------------------------------------------
    # If already on requested phase
    # --------------------------------------------------------

    if current == action:

        traci.trafficlight.setPhase(
            TLS_ID,
            action
        )

        for _ in range(5):

            traci.simulationStep()

            phase_time += 1

        return

    # --------------------------------------------------------
    # Change phase
    # First go through yellow phase
    # --------------------------------------------------------

    if current != YELLOW_PHASE:

        traci.trafficlight.setPhase(
            TLS_ID,
            YELLOW_PHASE
        )

        for _ in range(5):

            traci.simulationStep()

    # --------------------------------------------------------
    # Set DQN-selected phase
    # --------------------------------------------------------

    current_phase = action
    phase_time = 0

    traci.trafficlight.setPhase(
        TLS_ID,
        action
    )

    # Let the selected phase run
    for _ in range(5):

        traci.simulationStep()

        phase_time += 1


# ============================================================
# TEST / EVALUATION
# ============================================================

def test():

    global current_phase
    global phase_time

    current_phase = 0
    phase_time = 0

    print("Starting DQN evaluation...")

    traci.start(SUMO_CMD)

    step = 0

    # Run the same 3600-second simulation
    while traci.simulation.getTime() < 3600:

        # ----------------------------------------------------
        # 1. Observe current traffic
        # ----------------------------------------------------

        state = get_state()

        # ----------------------------------------------------
        # 2. Convert state to PyTorch tensor
        # ----------------------------------------------------

        state_t = torch.tensor(
            state,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # 3. DQN predicts Q-values
        # ----------------------------------------------------

        with torch.no_grad():

            q_values = policy_net(state_t)

        # ----------------------------------------------------
        # 4. Choose best action
        #
        # NO RANDOM EXPLORATION HERE.
        # The trained DQN is only being evaluated.
        # ----------------------------------------------------

        action_index = int(
            torch.argmax(q_values).item()
        )

        action = GREEN_PHASES[action_index]

        # ----------------------------------------------------
        # 5. Apply selected signal phase
        # ----------------------------------------------------

        apply_action(action)

        # ----------------------------------------------------
        # Progress information
        # ----------------------------------------------------

        if step % 40 == 0:

            print(
                f"Time={traci.simulation.getTime():.1f} | "
                f"State={state} | "
                f"Q-values={q_values.numpy()} | "
                f"Action={action}"
            )

        step += 1

    # --------------------------------------------------------
    # End simulation
    # --------------------------------------------------------

    traci.close()

    print()
    print("========================================")
    print("DQN testing completed successfully.")
    print("========================================")
    print("Generated files:")
    print("  test_tripinfo.xml")
    print("  test_summary.xml")
    print("  test_queue.xml")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    test()