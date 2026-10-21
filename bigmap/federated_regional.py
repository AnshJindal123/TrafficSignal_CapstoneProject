"""
Regional Federated Learning layer for the 8 per-junction DQN agents.

All 8 controllable junctions form ONE region (see CLAUDE.md). Each junction
keeps its own SingleJunctionAgent (existing DQN, replay buffer, optimizer,
epsilon schedule) and trains locally. After every round of local training
only model PARAMETERS cross the node boundary to the RegionalAggregator -
replay buffers / observations / traffic data never do.

Heterogeneity: nodes differ in state size and action count, so their input
layer (state -> hidden) and output head (hidden -> actions) cannot be
averaged. Following the "FedPer" pattern, only the hidden->hidden layer
(SHARED_KEYS) is federated; input adapter and Q-head stay personalised per
node. To make that layer shape-compatible every node uses FL_HIDDEN_SIZE
(the two richer-state junctions would otherwise be 128 wide).

Extensibility: RegionalAggregator is keyed by region_id and exposes
get_regional_model() (a plain state_dict of the shared layers), which a
future GlobalAggregator can average across regions the same way.

Two modes, identical in everything except the aggregation step, so they are
directly comparable (same seed, episodes, hidden size, reward weights):
    python federated_regional.py train fl            # regional FedAvg
    python federated_regional.py train independent   # no aggregation
Then: python eval_fl_compare.py

Round structure (both modes): `rounds` rounds of [local training -> (fl only)
aggregation], then ONE final local-only round with no aggregation, so every
node's input layer / Q-head re-adapts to the last averaged layer before the
model is saved/evaluated. In fl mode the models from just BEFORE the last
aggregation are also saved (mode "fl_preagg") to measure what averaging
itself did.

Aggregation refinements (fl mode):
  * BLEND_ALPHA: a node moves only part-way to the regional layer
    (new = (1-a)*local + a*regional) instead of being overwritten, since the
    hidden units of different nodes are not feature-aligned.
  * Reward-scale weighting: busy nodes have |reward| ~2-3 per step, idle
    ones ~0, so a plain step-weighted average is dominated by the busy
    nodes' gradient scale. Weight = steps / clip(mean|reward|, *REWARD_SCALE_CLIP).

Checkpoints: dqn_policy_{mode}_{tls_id}.pt, regional_model_{region}.pt
(canonical dqn_policy_seq_* files are never touched).
"""
import os
import random
import sys

import numpy as np
import torch
import traci

import dqn_traffic_combined as m
import discover_neighbor_topology as nt
import unified_dqn as ud

REGION_ID = "region_main"

# Training seed (python/numpy/torch RNG and the SUMO demand seed used while
# TRAINING), set via the FL_SEED env var. Default 42 = the original runs, whose
# filenames are unchanged; any other seed appends _s<seed> to every output so
# multi-seed runs never overwrite each other. Evaluation uses EVAL_SEED (below)
# so every model is judged on the same traffic.
TRAIN_SEED = int(os.environ.get("FL_SEED", m.SEED))
# Agent/network design (FL_DESIGN): "current" = per-junction shapes, only the
# hidden->hidden layer is shared (original FL layer); "unified" = the common
# interface in unified_dqn.py, whole network shared; "canonical" is evaluation
# only (the original per-junction checkpoints dqn_policy_seq_*.pt).
DESIGN = os.environ.get("FL_DESIGN", "current")
assert DESIGN in ("current", "unified", "canonical")
# Training regime (FL_REGIME): "joint" = all 8 agents learn at once in one shared
# simulation (this file's train()); "solo" = each node trains ALONE against
# fixed-time neighbors in its own simulation, like the canonical checkpoints
# were (federated_solo.py). The tag keeps their outputs separate.
REGIME = os.environ.get("FL_REGIME", "joint")
assert REGIME in ("joint", "solo")
# Regional grouping (FL_GROUPING): "none" = one region of all 8 nodes (original);
# "load" = nodes are split into regions by how congested they are under the
# fixed-time baseline (groups_by_load below), one RegionalAggregator per region,
# so nodes only learn from nodes that see similar traffic. Each region's model
# is a separate state_dict - exactly what a future global aggregator would
# average across regions.
GROUPING = os.environ.get("FL_GROUPING", "none")
assert GROUPING in ("none", "load")
BUSY_QUEUE_M = 20.0     # fixed-time avg queue (m) above which a node is "busy"
DESIGN_TAG = ((ud.TAG if DESIGN == "unified" else "") + ("_solo" if REGIME == "solo" else "")
              + ("_g" if GROUPING == "load" else ""))
SEED_SUFFIX = DESIGN_TAG + ("" if TRAIN_SEED == m.SEED else f"_s{TRAIN_SEED}")
# Aggregation weights: "reward" = steps / clipped mean|reward| (original FL
# runs), "steps" = plain FedAvg by local step count. Default follows design so
# old runs stay reproducible.
WEIGHTING = os.environ.get("FL_WEIGHTING", "steps" if DESIGN == "unified" else "reward")
assert WEIGHTING in ("reward", "steps")
# SUMO seed used for greedy EVALUATION episodes (FL_EVAL_SEED, default = m.SEED),
# so the same checkpoints can be re-tested on different traffic realisations.
EVAL_SEED = int(os.environ.get("FL_EVAL_SEED", m.SEED))
FL_HIDDEN_SIZE = 64
SHARED_KEYS = ("fc.2.weight", "fc.2.bias")   # hidden -> hidden layer of DQN.fc
DECISION_INTERVAL = 5
YELLOW_STEPS = 5

ROUNDS = 5
EPISODES_PER_ROUND = 4
BLEND_ALPHA = 0.5
REWARD_SCALE_CLIP = (0.5, 4.0)


def groups_by_load(config, min_queue_m=None):
    """{region_id: [tls_id, ...]}: nodes split into a 'busy' and a 'light' region
    by their average queue under the non-adaptive fixed-time program
    (ms_results/fixed_e*.json, written by eval_fixed.py). This is known before
    any learning and needs no DQN results; the two sets are far apart on this
    map (busy 58-246 m vs light 0-9 m). For a new area, run eval_fixed.py first."""
    import glob
    import json
    thr = BUSY_QUEUE_M if min_queue_m is None else min_queue_m
    qs = {}
    for f in glob.glob("ms_results/fixed_e*.json"):
        for j, v in json.load(open(f))["queue"].items():
            qs.setdefault(j, []).append(v)
    if not all(t in qs for t in config):
        raise RuntimeError("fixed-time queues missing - run eval_fixed.py first")
    groups = {"busy": [], "light": []}
    for t in config:
        groups["busy" if sum(qs[t]) / len(qs[t]) > thr else "light"].append(t)
    return {g: ts for g, ts in groups.items() if ts}


class RegionalAggregator:
    """FedAvg over the shared layers of the nodes in one region."""

    def __init__(self, region_id, shared_keys=SHARED_KEYS):
        self.region_id = region_id
        self.shared_keys = tuple(shared_keys)
        self.regional_model = None

    def aggregate(self, updates):
        """updates: {node_id: (shared_state_dict, weight)}. Weighted average
        (weight = e.g. number of local training steps this round)."""
        total = float(sum(w for _, w in updates.values()))
        if total <= 0:
            total = float(len(updates))
            updates = {k: (sd, 1.0) for k, (sd, _) in updates.items()}
        agg = {}
        for key in self.shared_keys:
            shapes = {tuple(sd[key].shape) for sd, _ in updates.values()}
            if len(shapes) != 1:
                raise ValueError(f"shared layer {key} has mismatched shapes {shapes}")
            agg[key] = sum(sd[key].detach().clone() * (w / total)
                           for sd, w in updates.values())
        self.regional_model = agg
        return agg

    def get_regional_model(self):
        return self.regional_model


def shared_keys_for(agent):
    """Unified agents share the WHOLE network; current-design agents share only
    the hidden->hidden layer (the only shape-compatible one)."""
    if isinstance(agent, ud.UnifiedJunctionAgent):
        return tuple(k for k in agent.policy_net.state_dict()
                     if not k.startswith(ud.LOCAL_KEY_PREFIXES))
    return SHARED_KEYS


def get_shared_params(agent):
    sd = agent.policy_net.state_dict()
    return {k: sd[k].detach().clone() for k in shared_keys_for(agent)}


def set_shared_params(agent, shared, alpha=1.0):
    """alpha=1 overwrites the node's shared layer with the regional one (used
    for the common initial model); alpha<1 blends regional into local."""
    sd = agent.policy_net.state_dict()
    for k in shared_keys_for(agent):
        sd[k].copy_((1.0 - alpha) * sd[k] + alpha * shared[k])
    agent.target_net.load_state_dict(agent.policy_net.state_dict())


class Runner:
    """Per-agent shared-clock state machine (same cadence as apply_action)."""

    def __init__(self, agent):
        self.agent = agent
        self.pending_action = None
        self.transition_remaining = 0
        self.pending_train = None   # (state, action) awaiting next_state/reward
        self.train_steps = 0        # local steps since last aggregation
        self.reward_sum = 0.0
        self.reward_abs_sum = 0.0
        self.reward_n = 0


def build_agent(tls_id, info, neighbor_specs, decay, design=None, phase_inflow_lanes=None):
    design = design or DESIGN
    qw, ww, qp = m.REWARD_WEIGHTS_OVERRIDES.get(tls_id, m.DEFAULT_REWARD_WEIGHTS)
    kw = dict(queue_weight=qw, wait_weight=ww, queue_power=qp,
              epsilon_decay=decay, neighbor_specs=neighbor_specs)
    if design == "unified":
        return ud.UnifiedJunctionAgent(tls_id, info["green_phases"], info["lane_groups"],
                                       phase_inflow_lanes=phase_inflow_lanes, **kw)
    if design == "canonical":
        return m.SingleJunctionAgent(tls_id, info["green_phases"], info["lane_groups"], **kw)
    return m.SingleJunctionAgent(tls_id, info["green_phases"], info["lane_groups"],
                                 hidden_size=FL_HIDDEN_SIZE, **kw)


def build_runners(config, topology, total_episodes, design=None):
    decay = m.scaled_epsilon_decay(total_episodes)
    design = design or DESIGN
    phase_inflows = (nt.discover_phase_inflows(config)
                     if design == "unified" and ud.VERSION == "v3" else {})
    return {tls_id: Runner(build_agent(tls_id, info, topology[tls_id], decay, design,
                                       phase_inflows.get(tls_id)))
            for tls_id, info in config.items()}


def reset_episode_state(runners, config):
    for tls_id, r in runners.items():
        a = r.agent
        a.current_phase = config[tls_id]["green_phases"][0]
        a.phase_time = 0
        a.wait_since_served = [0] * a.n_actions
        a.prev_queue = 0
        a.prev_queues = [0] * a.n_actions
        r.pending_action = None
        r.transition_remaining = 0
        r.pending_train = None


def run_episode(runners, config, learn, sumo_extra=()):
    """One 3600 s simulation with ALL agents on one shared clock. learn=True:
    every agent stores transitions and trains (local DQN training, unchanged);
    learn=False: pure greedy evaluation (caller sets epsilon=0)."""
    traci.start(["sumo", "-c", "combined.sumocfg", "--no-step-log",
                 "--duration-log.disable",
                 "--seed", str(TRAIN_SEED if learn else EVAL_SEED), *sumo_extra])
    for tls_id in config:
        m.apply_phase_merge(tls_id)
    reset_episode_state(runners, config)

    step_counter = 0
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
            idx = agent.green_phases.index(agent.current_phase)
            for i in range(agent.n_actions):
                agent.wait_since_served[i] = 0 if i == idx else agent.wait_since_served[i] + 1
            if agent.phase_time % DECISION_INTERVAL != 0:
                continue

            state = agent.get_state()
            if learn and r.pending_train is not None:
                prev_state, prev_action = r.pending_train
                reward = agent.get_reward()
                agent.memory.append((prev_state, prev_action, reward, state))
                agent.train()
                agent.maybe_update_target(r.train_steps)
                agent.decay_epsilon()
                r.train_steps += 1
                r.reward_sum += reward
                r.reward_abs_sum += abs(reward)
                r.reward_n += 1

            action = agent.choose_action(state)
            if learn:
                r.pending_train = (state, action)

            if action != agent.current_phase:
                traci.trafficlight.setPhase(tls_id, traci.trafficlight.getPhase(tls_id) + 1)
                r.pending_action = action
                r.transition_remaining = YELLOW_STEPS
            else:
                traci.trafficlight.setPhase(tls_id, agent.current_phase)
    traci.close()


def ckpt_name(mode, tls_id):
    if mode == "canonical":   # original per-junction checkpoints, any seed/design
        return f"dqn_policy_seq_{tls_id.replace('/', '_')}.pt"
    return f"dqn_policy_{mode}{SEED_SUFFIX}_{tls_id.replace('/', '_')}.pt"


def train(mode, rounds=ROUNDS, episodes_per_round=EPISODES_PER_ROUND):
    assert mode in ("fl", "independent")
    # Same seeding for both modes -> they differ only by the aggregation step.
    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)

    config = m.discover_controllable_tls()
    topology = nt.discover_neighbor_inflows(config)
    runners = build_runners(config, topology, (rounds + 1) * episodes_per_round)
    first_agent = next(iter(runners.values())).agent
    keys = shared_keys_for(first_agent)
    if GROUPING == "load":
        groups = groups_by_load(config)
        aggregators = {f"{REGION_ID}_{g}": (RegionalAggregator(f"{REGION_ID}_{g}", keys), ts)
                       for g, ts in groups.items()}
        print("regions:", {k: [t[:12] for t in ts] for k, (_, ts) in aggregators.items()}, flush=True)
    else:
        aggregators = {REGION_ID: (RegionalAggregator(REGION_ID, keys), list(runners))}

    if mode == "fl" or DESIGN == "unified":
        # Step 1: common initial model distributed to every node. For the
        # unified design (identical architectures) this is done in BOTH modes,
        # so independent vs fl differ only by the aggregation step.
        init = get_shared_params(first_agent)
        for r in runners.values():
            set_shared_params(r.agent, init)

    print(f"### mode={mode} region={REGION_ID} nodes={len(runners)} rounds={rounds} "
          f"+1 final local round, episodes/round={episodes_per_round} hidden={FL_HIDDEN_SIZE} "
          f"blend={BLEND_ALPHA} seed={TRAIN_SEED} design={DESIGN} weighting={WEIGHTING} ###", flush=True)

    def save_all(prefix_mode):
        for tls_id, r in runners.items():
            torch.save(r.agent.policy_net.state_dict(), ckpt_name(prefix_mode, tls_id))

    # `rounds` federated rounds, then one final local-only round (rnd == rounds).
    for rnd in range(rounds + 1):
        final_local = (rnd == rounds)
        for r in runners.values():
            r.train_steps = 0
        for ep in range(episodes_per_round):
            run_episode(runners, config, learn=True)
            eps = np.mean([r.agent.epsilon for r in runners.values()])
            print(f"round {rnd} episode {ep} done | mean epsilon={eps:.3f}", flush=True)

        note = "final local-only round, no aggregation"
        if mode == "fl" and not final_local:
            if rnd == rounds - 1:
                save_all("fl_preagg")   # models just before the last aggregation
            # Steps 4-6: nodes send parameters only -> aggregate -> redistribute.
            updates = {}
            for t, r in runners.items():
                scale = 1.0 if WEIGHTING == "steps" else min(
                    max(r.reward_abs_sum / max(1, r.reward_n), REWARD_SCALE_CLIP[0]),
                    REWARD_SCALE_CLIP[1])
                updates[t] = (get_shared_params(r.agent), r.train_steps / scale)
            wnote = []
            for rid, (aggr, members) in aggregators.items():
                sub = {t: updates[t] for t in members}
                regional = aggr.aggregate(sub)         # one aggregation per region
                for t in members:
                    set_shared_params(runners[t].agent, regional, BLEND_ALPHA)
                total_w = sum(w for _, w in sub.values())
                wnote += [f"{t.split('_')[0][:6]}:{w / total_w:.2f}" for t, (_, w) in sub.items()]
            note = "aggregated, weights " + " ".join(wnote)
        elif mode == "independent":
            note = "no aggregation"

        print(f"== round {rnd} complete ({note}) | avg reward "
              + " ".join(f"{t.split('_')[0][:6]}:{r.reward_sum / max(1, r.reward_n):.3f}"
                         for t, r in runners.items()), flush=True)
        for r in runners.values():
            r.reward_sum = r.reward_abs_sum = 0.0
            r.reward_n = 0

    save_all(mode)
    if mode == "fl":
        for rid, (aggr, _) in aggregators.items():
            torch.save(aggr.get_regional_model(), f"regional_model_{rid}{SEED_SUFFIX}.pt")
    print(f"=== federated_regional.py finished mode={mode} ===")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "train":
        sys.exit("usage: python federated_regional.py train <fl|independent> [rounds] [episodes_per_round]")
    r_ = int(sys.argv[3]) if len(sys.argv) > 3 else ROUNDS
    e_ = int(sys.argv[4]) if len(sys.argv) > 4 else EPISODES_PER_ROUND
    train(sys.argv[2], r_, e_)
