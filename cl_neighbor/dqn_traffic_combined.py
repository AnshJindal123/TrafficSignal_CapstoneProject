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

from sumolib.net import Phase as m_Phase

gamma = 0.95
epsilon_min = 0.05
epsilon_decay = 0.995
lr = 0.001
batch_size = 64
memory_size = 5000
target_update_freq = 50

MIN_GREEN = 20
MAX_GREEN = 60
# A branch-usage diagnostic tried raising these to give the trained
# network more of the "contestable" decisions (those past the mandatory
# min-green hold). It measurably backfired - the network's share of
# contestable decisions dropped from 64.3% to 54.2% in a real run - so
# this is reverted to the original values. Recorded here as a checked
# and rejected hypothesis, not an untested guess.
MIN_IMBALANCE = 3 / 20.0
IMBALANCE_RATIO = 1.5

# apply_action holds the yellow phase for exactly this many simulation
# steps (see the range(5) loop there) - kept as a named constant for the
# cycle-length reasoning below.
YELLOW_DURATION = 5

# Per-junction (min_green, max_green) override, same pattern as
# REWARD_WEIGHTS_OVERRIDES. Default is the global MIN_GREEN/MAX_GREEN.
#
# First hypothesis tried here (train_cycle_fix.py) SHRANK both bounds
# as phase count grew, on the theory that the 6-phase junction's
# single-lane approaches were starved by waiting up to 390s between
# turns (6 * (20..60 + yellow)). Tested and rejected: it made both
# multi-phase junctions substantially worse, not better - queue% on
# the 6-phase junction collapsed from -0.18% to -56.80%. Root cause:
# shrinking green time raises the FRACTION of every cycle lost to the
# fixed 5s yellow transition (20% of a 25s phase-slot vs 7.7% of a 65s
# one) - exactly backwards for a junction that queue time-series show
# is already demand-constrained (sustained 60-155 vehicle plateaus for
# most of the hour). Standard signal-timing practice says oversaturated
# intersections need LONGER cycles to minimize lost-time overhead and
# maximize usable green - so the counter-hypothesis (train_longcycle.py,
# (20, 90) for the 6-phase junction) was tried too. It also lost - worse
# than canonical on all 4 metrics (queue% -0.18% -> -31.03%), just less
# catastrophically than the shrink attempt. Both directions of adjusting
# these bounds hurt this junction, which is itself evidence the original
# (20, 60) tuning is already near a local optimum here - the bottleneck
# isn't green-time tuning at all, most likely the fixed phase count/
# geometry itself (3 of 5 approaches are single-lane against 6 phases).
# Left empty; a genuine fix would mean changing the number of phases
# (merging non-conflicting ones) or the lane geometry, not this knob.
DEFAULT_GREEN_BOUNDS = (MIN_GREEN, MAX_GREEN)
# 442107275: dwell-time diagnostic (diagnose_dwell.py) showed a clean
# bimodal ~27s/~67s cycle with the long side landing exactly at the
# MAX_GREEN(60s)-forced switch every cycle, 0% of switches near
# MIN_GREEN - i.e. the agent achieves a perfect local queue (never a
# single halted vehicle all hour) by holding one direction as long as
# it's allowed, which likely dumps one large platoon into the network
# per ~94s cycle instead of the baseline's smaller, more frequent
# releases (35s/25s alternating). That correlates with a real
# network-wide gridlock signature when this junction is DQN-controlled
# (before_summary.xml vs seq_after_summary_442107275.xml: teleports
# 1307->1539 (+18%), vehicles still running at hour-end 1677->1995,
# completed trips 5332->5013, mean speed 0.74->0.50 m/s). Capping
# max_green near the baseline's own ~35s tests whether forcing more
# frequent/smaller releases fixes the downstream burst effect - a
# different mechanism than the shrink/grow experiments tried earlier
# on the 6-phase junction (that was an oversaturation/phase-count
# problem, not a light-demand burst-release one).
# REVERTED (explicit user direction): DQN junctions must keep the full
# global (20,60) range and switch early based on genuine demand
# imbalance, not a shrunk max_green cap. The (20,25)/(20,35) overrides
# above are gone; the replacement mechanism for the same burst-release
# problem is IMBALANCE_OVERRIDES / imbalance_for() below, which makes
# the existing significant_imbalance heuristic (see choose_action)
# actually sensitive enough to fire at these two junctions' real (very
# light, ~0-2 vehicle) demand levels instead of never firing and always
# falling through to the max_green-forced switch.
GREEN_BOUNDS_OVERRIDES = {}


def green_bounds_for(tls_id):
    return GREEN_BOUNDS_OVERRIDES.get(tls_id, DEFAULT_GREEN_BOUNDS)


# Per-junction (min_imbalance, imbalance_ratio) override, same pattern as
# GREEN_BOUNDS_OVERRIDES/REWARD_WEIGHTS_OVERRIDES. Default is the global
# MIN_IMBALANCE/IMBALANCE_RATIO.
#
# 442107275 and 481750173: with GREEN_BOUNDS_OVERRIDES reverted to the
# global (20,60), diagnose_dwell.py's original bimodal always-hits-
# max_green pattern will return unless significant_imbalance actually
# starts firing at these junctions' real demand levels. It currently
# never does: max observed queue here is only ~0-2 vehicles all hour,
# so the global MIN_IMBALANCE (3 raw vehicles = 3/20.0 normalized) is
# never reached. Also, since the current phase's queue is very often
# exactly 0 here, "best_q > cur_q * ratio" reduces to "best_q > 0" and
# IMBALANCE_RATIO does little real work in that dominant case - only
# MIN_IMBALANCE actually gates the decision. Lowering MIN_IMBALANCE to
# 2/20.0 (2 raw vehicles) lets the common "0 waiting here, 2 waiting
# there" case fire the real imbalance path instead of only ever ending
# via the max_green timer; IMBALANCE_RATIO is loosened to 1.2 for the
# less common case where the current phase already has 1+ vehicles.
# This is an informed starting guess, not a proven value - verify with
# diagnose_dwell.py after retraining, not just the trip-metric outcome.
# Scoped to just these two junctions so the other 6 junctions' proven
# global thresholds are untouched.
DEFAULT_IMBALANCE = (MIN_IMBALANCE, IMBALANCE_RATIO)
IMBALANCE_OVERRIDES = {
    "442107275": (2 / 20.0, 1.2),
    "481750173": (2 / 20.0, 1.2),
}


def imbalance_for(tls_id):
    return IMBALANCE_OVERRIDES.get(tls_id, DEFAULT_IMBALANCE)


NUM_EPISODES = 20

# Reward weights are (queue_weight, wait_weight, queue_power) - see
# SingleJunctionAgent.get_reward. A controlled per-junction reward
# experiment (reward_experiment.py, with per-junction RNG reseeding so
# the comparison isn't confounded by leftover random state) showed the
# default weights win clearly on typical 2-action junctions, but the
# hardest 6-action junction does much better - including flipping queue
# length from strongly negative to +22% - when wait time is weighted
# more heavily relative to queue length.
DEFAULT_REWARD_WEIGHTS = (0.01, 0.001, 1)
REWARD_WEIGHTS_OVERRIDES = {
    "cluster_11801662689_12136946746_12136946747_12136946748_#3more": (0.005, 0.002, 1),
}

# Richer state ([queue,wait,speed,trend] per phase + [phase,phase_time])
# was tried on ALL 8 junctions (train_richer_state.py) but only won
# against the canonical thin state ([queue_per_phase] + [phase]) on
# these two - it made the other 6 worse, badly so on the two hardest
# junctions. The canonical .pt checkpoint for every junction NOT in
# this set is still thin-state-shaped; loading a checkpoint with the
# wrong mode's network will crash on a shape mismatch, so this set
# must stay in sync with whichever mode's tag is actually canonical
# for each junction (see pick_best_general.py).
#
# 343322679: reverting to thin state was requested (richer state won
# there on the majority rule but made queue% worse: +22.45% -> -5.28%).
# NOT applied: train_revert_343322679.py retrained under this file's
# thin-state code and got wait=-2.08%/travel=-0.40%/loss=-0.64%/
# queue=-13.31% - worse than richer state on EVERY metric, not a match
# for the +2.89/+0.40/+0.89/+22.45% on record from when richer state
# last beat thin state here. The likely explanation: this file's thin-
# state get_state()/choose_action() branch was reconstructed from
# indirect evidence (checkpoint input-shape, MIN_IMBALANCE calibration)
# after the original thin-state result files were already overwritten
# by richer state's win - not recovered from the original source - so
# it isn't guaranteed byte-for-byte identical to whatever code actually
# produced the on-record thin-state numbers. Left as richer state
# (still canonical, still the better of the two available options)
# until the discrepancy is resolved.
RICHER_STATE_TLS = {"343322679", "481750173"}

# Normalization for the neighbor-outflow state feature (see
# discover_neighbor_topology.py) - one scalar per neighboring
# controllable junction, summing traci.lane.getLastStepVehicleNumber()
# over that neighbor's source lanes. Validated via check_neighbor_scale.py
# against a full-hour run with every junction on the realistic fixed-time
# baseline: raw sums ranged 0-22 across all discovered neighbor pairs
# (mean 0.09-14.57 depending on the pair), close enough to the existing
# queue features' 0-20 scale to reuse the same divisor - only the
# busiest pair (#4more -> #3more) occasionally exceeds 1.0 by ~10%,
# which is fine for a neural net input.
NEIGHBOR_OUTFLOW_SCALE = 20.0

# (green_phase_index_a, green_phase_index_b) pairs to merge into one
# simultaneous-green phase, per junction - only for pairs verified to
# have zero conflicting movements. Verified for the 6-phase junction by
# resolving each connection's local junction request index (sumolib
# Connection.getJunctionIndex()) and checking every pair of green
# phases with sumolib Node.areFoes() against SUMO's own computed
# conflict/foe matrix (phase_conflict_check.py methodology) - of 15
# possible pairs, phases 0 and 2 (both movements from the same
# incoming edge: through/right from one lane, protected lefts from the
# other) were the only pair with zero foes; every other pair genuinely
# conflicts. Merging them drops this junction from 6 phases to 5 (a
# real ~17% cycle-length reduction) without touching MIN_GREEN/
# MAX_GREEN, which already failed in both directions for this
# junction (see GREEN_BOUNDS_OVERRIDES above).
#
# Tested (train_merge.py): fixed the local queue metric dramatically
# (-0.18% -> +49.57%, confirming the phase-count theory was right
# about that dimension) but made the trip-level wait/travel/loss
# metrics worse across the board (-2.55%->-12.60%, +1.58%->-1.92%,
# +1.05%->-5.45%) - 3 of 4 metrics regressed, which the standard
# majority-of-4-metrics rule would reject despite the higher raw sum.
# Deliberately overridden for this one junction (explicit call, not
# the general pick_best_general rule) because every other lever tried
# here - reward weights, exploration schedule, bigger net, richer
# state, green bounds in both directions - also failed, and a +49.57
# point queue swing against a -20-point combined cost on the other
# three was judged worth it for this specific, otherwise-stuck
# junction. Not a precedent for other junctions.
PHASE_MERGE_TLS = {
    "cluster_11801662689_12136946746_12136946747_12136946748_#3more": (0, 2),
}


def apply_phase_merge(tls_id):
    """
    Must be called once right after every traci.start() for a junction
    in PHASE_MERGE_TLS - a custom program logic doesn't persist across
    SUMO restarts, so this rebuilds it each time. No-op otherwise.
    """
    pair = PHASE_MERGE_TLS.get(tls_id)
    if pair is None:
        return
    a, b = pair
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    old_phases = logic.phases

    def merge_state(s1, s2):
        return "".join(c1 if c1 != 'r' else c2 for c1, c2 in zip(s1, s2))

    merged_green = m_Phase(old_phases[a].duration, merge_state(old_phases[a].state, old_phases[b].state))
    merged_yellow = m_Phase(old_phases[a + 1].duration,
                             merge_state(old_phases[a + 1].state, old_phases[b + 1].state))
    skip = {a, a + 1, b, b + 1}
    keep = [p for i, p in enumerate(old_phases) if i not in skip]
    new_logic = traci.trafficlight.Logic(logic.programID, logic.type, 0,
                                          phases=[merged_green, merged_yellow] + keep)
    traci.trafficlight.setProgramLogic(tls_id, new_logic)


# Realistic Bangalore-style fixed-time program (green scaled by lane
# count per phase, compulsory - never responsive to real-time queue
# length, only each phase's static lane count, same as any true
# fixed-time signal; yellow flat 5s; red emergent from the other
# phases) - canonical definition, also used by
# generate_realistic_baseline.py for the master before_*.xml and by
# install_realistic_bystanders() below so every "after" run gives every
# non-target junction this same timing instead of silently reverting
# them to SUMO's default program (the inconsistency found while
# investigating whether traffic flows correctly between
# cluster_11594569878... and GS_cluster_12692472176... as neighbors -
# before_*.xml had all 8 junctions realistic, but every "after" run only
# ever installed a custom program on the one junction being trained/
# evaluated, leaving the other 7 on default for that run).
#
# History: 25-35s (original) -> 45-50s (explicit user direction) ->
# 55-65s (current). The 45-50s change was measured (via
# generate_baseline_25_35.py/45_50/55_65 tagged side-by-side runs) to
# improve network-wide throughput (+192 trips completed, -5% teleports,
# ~2x mean speed) despite worsening queue at 7/8 junctions - the
# opposite of "worse baseline" if throughput counts. 55-65s was tested
# next and found to worsen BOTH queue (7/8 junctions) AND throughput
# (-613 trips completed vs 45-50s, +26% teleports, mean speed roughly
# halved again) relative to 45-50s - a harsher, more clogged baseline
# on every axis, promoted to canonical per explicit user direction.
# Backups of the superseded 25-35s/45-50s canonical before_*.xml are
# kept as before_*_25_35.xml / before_*_45_50.xml.
BASELINE_GREEN_MIN = 55
BASELINE_GREEN_MAX = 65
BASELINE_YELLOW = 5


def baseline_green_duration_for(lane_groups, phase_idx):
    lane_counts = [len(g) for g in lane_groups]
    lo, hi = min(lane_counts), max(lane_counts)
    this_count = lane_counts[phase_idx]
    if hi == lo:
        return (BASELINE_GREEN_MIN + BASELINE_GREEN_MAX) // 2
    frac = (this_count - lo) / (hi - lo)
    return round(BASELINE_GREEN_MIN + frac * (BASELINE_GREEN_MAX - BASELINE_GREEN_MIN))


def install_realistic_program(tls_id, green_phases, lane_groups):
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    old_phases = logic.phases
    new_phases = []
    for i, gp in enumerate(green_phases):
        green_dur = baseline_green_duration_for(lane_groups, i)
        new_phases.append(m_Phase(green_dur, old_phases[gp].state))
        yellow_idx = gp + 1
        new_phases.append(m_Phase(BASELINE_YELLOW, old_phases[yellow_idx].state))
    new_logic = traci.trafficlight.Logic(logic.programID, logic.type, 0, phases=new_phases)
    traci.trafficlight.setProgramLogic(tls_id, new_logic)
    return [p.duration for p in new_phases]


def install_realistic_bystanders(config, exclude_tls_id):
    """
    Installs the realistic baseline program on every controllable
    junction in config EXCEPT exclude_tls_id (the one under active
    DQN/agent control this run). Must be called after traci.start(),
    same as apply_phase_merge - a custom program doesn't persist across
    restarts. Safe/no-op-equivalent to call even if a bystander is the
    phase-merge junction: apply_phase_merge is applied first so its
    green_phases indices (already post-merge, from discover_controllable_tls)
    line up with the live program.
    """
    for tls_id, info in config.items():
        if tls_id == exclude_tls_id:
            continue
        apply_phase_merge(tls_id)
        install_realistic_program(tls_id, info["green_phases"], info["lane_groups"])


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
        apply_phase_merge(tls_id)
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
    def __init__(self, n_actions, state_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
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
    def __init__(self, tls_id, green_phases, lane_groups,
                 queue_weight=0.01, wait_weight=0.001, queue_power=1,
                 epsilon_decay=epsilon_decay, delta_weight=0.02,
                 neighbor_specs=None):
        self.tls_id = tls_id
        self.green_phases = green_phases
        self.lane_groups = lane_groups
        self.n_actions = len(green_phases)
        # [(neighbor_tls_id, source_lane_ids), ...] from
        # discover_neighbor_topology.discover_neighbor_outflows(), sorted
        # by neighbor_tls_id - this ordering is load-bearing (baked into
        # state_size/the checkpoint's input layer), must be identical at
        # train and eval/load time. Junctions with no neighbor within
        # NEIGHBOR_MAX_DISTANCE_M get an empty list, unaffected.
        self.neighbor_specs = neighbor_specs or []
        self.n_neighbors = len(self.neighbor_specs)
        # Reward = -(queue_weight * queue^queue_power + wait_weight * wait)
        #          + delta_weight * (prev_queue - queue).
        # The delta term is potential-based shaping: it gives the agent
        # immediate credit/blame for what its LAST action actually did to
        # the queue, instead of only ever seeing a flat penalty on
        # absolute state that can't distinguish "I just fixed this" from
        # "I just caused this." Kept configurable so different penalty
        # balances can be compared without duplicating the agent/training
        # code.
        self.queue_weight = queue_weight
        self.wait_weight = wait_weight
        self.queue_power = queue_power
        self.delta_weight = delta_weight
        self.prev_queue = 0
        # Per-phase previous queue count, used only for the state's
        # queue-trend feature (is this direction getting worse or
        # better) - separate from prev_queue above, which tracks the
        # junction-wide total for reward shaping.
        self.prev_queues = [0] * self.n_actions
        # State mode is per-junction (see RICHER_STATE_TLS): richer state
        # is [queue, wait, avg_speed, queue_trend] per phase plus
        # [current_phase_index, phase_elapsed_time] - the network can see
        # how long vehicles have waited (not just how many), whether a
        # queue is building or clearing, approach speed, and how long the
        # current phase has already run. Thin state (the default, and
        # still canonical for 6 of 8 junctions) is just
        # [queue_per_phase..., phase_index].
        self.use_richer_state = tls_id in RICHER_STATE_TLS
        if self.use_richer_state:
            self.state_size = self.n_actions * 4 + 2 + self.n_neighbors
            self.hidden_size = 128
        else:
            self.state_size = self.n_actions + 1 + self.n_neighbors
            self.hidden_size = 64

        self.policy_net = DQN(self.n_actions, self.state_size, self.hidden_size)
        self.target_net = DQN(self.n_actions, self.state_size, self.hidden_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=memory_size)
        self.current_phase = green_phases[0]
        self.phase_time = 0
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        # How long (in the same units as phase_time) each phase has gone
        # without being served. Used to force a phase to get its turn
        # instead of letting a persistently busy phase win the queue-based
        # comparison indefinitely - a real risk once there are 3+ phases
        # competing, since "switch to whoever has the biggest queue" alone
        # has no bound on how long a smaller-but-steady queue can be skipped.
        self.wait_since_served = [0] * self.n_actions
        # Per-junction green bounds (see GREEN_BOUNDS_OVERRIDES) - equal
        # to the global MIN_GREEN/MAX_GREEN unless overridden.
        self.min_green, self.max_green = green_bounds_for(tls_id)
        # Per-junction imbalance sensitivity (see IMBALANCE_OVERRIDES) -
        # equal to the global MIN_IMBALANCE/IMBALANCE_RATIO unless overridden.
        self.min_imbalance, self.imbalance_ratio = imbalance_for(tls_id)

    def _neighbor_outflow_features(self):
        # One scalar per neighboring controllable junction (not per lane) -
        # sum of vehicles that just left THIS junction on the lanes heading
        # toward that neighbor, giving it advance notice of an incoming
        # surge. Stateless snapshot (traci.lane.getLastStepVehicleNumber),
        # unlike prev_queue/prev_queues - needs no per-episode reset.
        return [sum(traci.lane.getLastStepVehicleNumber(l) for l in lanes) / NEIGHBOR_OUTFLOW_SCALE
                for _, lanes in self.neighbor_specs]

    def get_state(self):
        if not self.use_richer_state:
            state = []
            for group in self.lane_groups:
                queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in group)
                state.append(queue / 20.0)
            phase = traci.trafficlight.getPhase(self.tls_id)
            state.append(phase / 10.0)
            state.extend(self._neighbor_outflow_features())
            return np.array(state, dtype=np.float32)

        state = []
        for i, group in enumerate(self.lane_groups):
            queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in group)
            wait = sum(traci.lane.getWaitingTime(l) for l in group)
            speeds = [traci.lane.getLastStepMeanSpeed(l) for l in group]
            speed = sum(speeds) / len(speeds) if speeds else 0.0
            trend = self.prev_queues[i] - queue
            self.prev_queues[i] = queue
            state.extend([queue / 20.0, wait / 300.0, speed / 15.0, trend / 20.0])
        phase = traci.trafficlight.getPhase(self.tls_id)
        state.append(phase / 10.0)
        state.append(self.phase_time / MAX_GREEN)
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
        # Thin state is [queue_per_phase..., phase_index]; richer state is
        # [queue,wait,speed,trend] per phase then [phase_index,
        # phase_elapsed_time]. Either way the heuristic checks below only
        # need the per-phase queue, normalized the same way (/20.0), so
        # MIN_IMBALANCE's calibration is unaffected by which mode this
        # junction uses (now read from self.min_imbalance/self.imbalance_ratio,
        # set per-junction via IMBALANCE_OVERRIDES - same reasoning still applies).
        if self.use_richer_state:
            queues = [state[i * 4] for i in range(self.n_actions)]
        else:
            queues = state[:self.n_actions]
        current_idx = self.green_phases.index(self.current_phase)

        # Fairness bound: no phase WITH ACTUAL DEMAND should wait longer
        # than every other phase getting a full max_green turn ahead of
        # it. Requiring queues[i] > 0 avoids burning a green+yellow cycle
        # on a phase nobody is waiting on - that wasted-cycle case is what
        # regressed the 3-action junction when this first tried an
        # unconditional starvation check. For a 2-action junction
        # max_green == MAX_GREEN, and the pre-existing unconditional
        # "phase_time > max_green" check below still forces the same
        # switch even when the other phase is empty, so 2-action behavior
        # is unchanged either way.
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
        # Double DQN: pick the best next action with the policy net (the
        # one still being updated) but evaluate its value with the target
        # net. Using the same network for both, as vanilla DQN does,
        # systematically overestimates Q-values - a likely contributor to
        # the drift we saw when training ran for many more episodes than
        # necessary instead of converging.
        next_actions = self.policy_net(next_states).detach().argmax(dim=1)
        next_q_vals_target = self.target_net(next_states).detach()
        target = q_vals.clone()
        for i in range(batch_size):
            action_index = self.green_phases.index(actions[i])
            target[i][action_index] = rewards[i] + gamma * next_q_vals_target[i][next_actions[i]]
        loss = self.loss_fn(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        if self.epsilon > epsilon_min:
            self.epsilon *= self.epsilon_decay

    def maybe_update_target(self, step):
        if step % target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

# ------------------------
# Train ONE junction for NUM_EPISODES, others stay on default program
# ------------------------
def scaled_epsilon_decay(num_episodes, steps_per_episode=600, explore_fraction=0.8):
    """
    epsilon_decay=0.995 reaches epsilon_min within ~700 steps - i.e.
    inside the FIRST episode - regardless of num_episodes, so extra
    episodes beyond the first were mostly just more gradient updates on
    an already-greedy policy, not more exploration. Scale the decay so
    epsilon reaches epsilon_min at explore_fraction of the full training
    budget instead, so "train longer" actually means "explore more."
    """
    target_steps = max(1, num_episodes * steps_per_episode * explore_fraction)
    return epsilon_min ** (1.0 / target_steps)

def train_one_junction(tls_id, green_phases, lane_groups, num_episodes=NUM_EPISODES,
                        reward_weights=(0.01, 0.001, 1), tag="", epsilon_decay_value=None,
                        delta_weight=0.02, bystander_config=None, neighbor_specs=None):
    # Re-seed right before each junction's training so every run starts
    # from an identical RNG state (network init, epsilon-greedy draws)
    # regardless of how many other junctions/variants were trained earlier
    # in this same process. Without this, comparisons across reward
    # variants (or even just "same junction, different run order") are
    # confounded by leftover random state instead of isolating the one
    # thing actually being varied.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    safe_name = tag + tls_id.replace("/", "_")
    queue_weight, wait_weight, queue_power = reward_weights
    eps_decay = epsilon_decay_value if epsilon_decay_value is not None else scaled_epsilon_decay(num_episodes)
    agent = SingleJunctionAgent(tls_id, green_phases, lane_groups,
                                 queue_weight=queue_weight, wait_weight=wait_weight,
                                 queue_power=queue_power, epsilon_decay=eps_decay,
                                 delta_weight=delta_weight, neighbor_specs=neighbor_specs)

    record_cmd = ["sumo", "-c", "combined.sumocfg",
                  "--tripinfo-output", f"seq_after_tripinfo_{safe_name}.xml",
                  "--summary", f"seq_after_summary_{safe_name}.xml",
                  "--queue-output", f"seq_after_queue_{safe_name}.xml",
                  "--no-step-log", "--duration-log.disable",
                  "--seed", str(SEED)]
    quiet_cmd = ["sumo", "-c", "combined.sumocfg", "--no-step-log", "--duration-log.disable",
                 "--seed", str(SEED)]

    for ep in range(num_episodes):
        is_last = (ep == num_episodes - 1)
        agent.current_phase = green_phases[0]
        agent.phase_time = 0
        agent.wait_since_served = [0] * agent.n_actions
        agent.prev_queue = 0
        agent.prev_queues = [0] * agent.n_actions

        traci.start(record_cmd if is_last else quiet_cmd)
        apply_phase_merge(tls_id)
        if bystander_config is not None:
            install_realistic_bystanders(bystander_config, tls_id)

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
    print(f"== Finished {tls_id} ({agent.n_actions} actions, tag='{tag}') ==\n")

def run():
    config = discover_controllable_tls()
    print(f"Discovered {len(config)} controllable junctions (>=2 real green phases):")
    for tls_id, info in config.items():
        print(f"  {tls_id} -> {len(info['green_phases'])} actions")

    for tls_id, info in config.items():
        weights = REWARD_WEIGHTS_OVERRIDES.get(tls_id, DEFAULT_REWARD_WEIGHTS)
        train_one_junction(tls_id, info["green_phases"], info["lane_groups"],
                            num_episodes=NUM_EPISODES, reward_weights=weights,
                            bystander_config=config)

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
