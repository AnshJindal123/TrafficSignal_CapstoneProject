# CLAUDE.md — cl_neighbor (Federated RL for Smart Traffic Lights)

Capstone project: **Federated Reinforcement Learning for Smart Traffic Lights using SUMO + TraCI + PyTorch DQN.**
This folder (`cl_neighbor`) is the latest iteration: 8 per-junction DQN agents on one combined SUMO map, now with a
"neighbor inflow" state feature. Branch: `combined_map`. Spec for the next task: [README_REGIONAL_FL.md](README_REGIONAL_FL.md).

## Current task (not started yet)
Add a **Regional Federated Learning layer** *around* the existing DQN training (details in README_REGIONAL_FL.md):
init common model -> distribute -> local training per node -> send **parameters only** (never raw traffic data) ->
Regional Aggregator -> redistribute -> repeat for several rounds. Must:
- reuse the existing `DQN` / `SingleJunctionAgent` (no second DQN implementation), preserve local learning;
- keep the design extensible to a future Global Aggregator over multiple regions (do NOT build the global one now);
- ship a minimal reproducible demo + allow comparison **Independent local DQN vs Regional Federated DQN** using the existing metrics.

## Environment
- Windows, Python with torch 2.8 (CPU) + traci/sumolib. `SUMO_HOME` must be set (`C:\Program Files (x86)\Eclipse\Sumo\`); the
  module calls `sys.exit` on import if it is missing. Scripts call the `sumo` binary and must be **run from this folder**
  (relative paths: `combined.sumocfg`, `combined.net.xml`, `dqn_policy_seq_*.pt`).
- No requirements file, no tests. Verification = running SUMO sims (1 hour sim = 3600 steps) and comparing metrics.
- Git-ignored outputs: `before_*.xml`, `seq_after_*.xml`. **`before_*.xml` is NOT present right now** (needed by
  `logs_combined.py`/`pick_best_general.py`) -> regenerate with `python generate_realistic_baseline.py`.

## Key files
| File | Role |
|---|---|
| `dqn_traffic_combined.py` | **Core.** `DQN` net, `SingleJunctionAgent`, `discover_controllable_tls()`, `train_one_junction()`, baseline/bystander programs, per-junction override tables. Imported as `m` everywhere. |
| `discover_neighbor_topology.py` | `discover_neighbor_inflows(config)` -> `{tls_id: [(upstream_tls, lanes), ...]}`; gives each junction its neighbor-outflow state features (700 m cutoff, Dijkstra over `combined.net.xml`). |
| `train_all.py` | Retrains the 4 neighbor-receiving junctions (GS_cluster, #4more, cluster_11594569878…, #3more) one at a time vs fixed-time bystanders; copies 1213632636's checkpoint from `../cl_retrain`. Writes canonical `.pt`. |
| `train_3more_vs_dqn_neighbors.py` | **Best template for multi-agent loop**: shared-clock driver with a `Runner` per agent, fused train loop; only #3more learns, other 7 frozen. Tag `vsdqn_`. |
| `eval_all_simultaneous.py` | All 8 DQNs live at once in one sim (shared clock, per-agent yellow/decision state machine). Writes `seq_after_*_all8dqn.xml`. |
| `eval_one.py <tls> [tag]` | Evaluate one checkpoint, epsilon=0, others on fixed-time baseline. |
| `reeval_canonical_all.py` | Re-evaluate all 8 canonical checkpoints (epsilon=0). |
| `generate_realistic_baseline.py` | Produces `before_*.xml` (all 8 on 55-65 s fixed-time) = the comparison baseline. |
| `logs_combined.py`, `pick_best_general.py`, `chart_all8dqn.py` | Metric tables/charts; challenger-vs-canonical promotion (majority of 4 metrics). |
| `check_neighbor_scale.py`, `diagnose_*.py` | One-off diagnostics, not pipeline. |
| `combined.net.xml`, `combined*.rou.xml`, `combined*.sumocfg` | SUMO map + demand. `combined.sumocfg` is the default (1 h); `combined_demand_{m30,m15,p15,p30,p50}` are demand-scaled variants. |
| `*.log` | Huge historical logs (40 MB + 18 MB) — don't read whole; grep/tail only. |

## Architecture of the existing system
- **8 controllable junctions** (of 18 signalised; the other 10 have a single green phase and stay on default). Found automatically:
  a TLS qualifies if it has >=2 "real" green phases (no `y`, has `g`). `config[tls_id] = {green_phases, lane_groups}`.
- One `SingleJunctionAgent` per junction: own `policy_net`, `target_net`, Adam (lr 1e-3), replay `deque(5000)`, epsilon-greedy,
  **Double DQN** target, batch 64, gamma 0.95, target sync every 50 steps. `DQN` = MLP `state -> h -> h -> n_actions`.
- **Action = which green phase to serve next.** Decisions every 5 sim-seconds; switching inserts a 5 s yellow.
  `choose_action` is a **hybrid heuristic + DQN**: starvation fairness override, min_green(20)/max_green(60) constraints,
  `significant_imbalance` rule, and only otherwise epsilon-greedy / argmax Q. So the net only decides "contestable" moments.
- **State** (per junction, shape fixed at construction):
  - thin (6 junctions): `[queue/20 per phase] + [phase/10] + [neighbor_outflow/20 per upstream neighbor]`, hidden 64
  - richer (`RICHER_STATE_TLS = {343322679, 481750173}`): `[queue,wait,speed,trend]` per phase + `[phase, phase_time]` + neighbors, hidden 128
- **Reward**: `-(qw*queue^p + ww*wait) + 0.02*(prev_queue - queue)`; weights per junction via `REWARD_WEIGHTS_OVERRIDES` (#3more differs).
- Per-junction overrides: `IMBALANCE_OVERRIDES` (442107275, 481750173), `PHASE_MERGE_TLS` (#3more merges phases 0&2 -> 5 actions), `GREEN_BOUNDS_OVERRIDES` (empty).
- **Bystanders**: when training/evaluating one junction, the other 7 run the realistic fixed-time program (green 55-65 s by lane count, 5 s yellow).
- Training: 20-60 episodes of 3600 sim-s (one SUMO restart per episode), epsilon decay scaled so it reaches 0.05 at 80% of budget
  (`scaled_epsilon_decay`), re-seeded with `SEED=42` before each junction. Checkpoint = `policy_net.state_dict()` only
  -> `dqn_policy_seq_{tag}{tls_id}.pt` (tls_id `/`->`_`).
- **Evaluation convention (final)**: epsilon=0, fully greedy, frozen weights. Metrics from tripinfo/queue XML vs `before_*.xml`:
  avg waiting time, travel time, time loss (network-wide, per trip) and junction queue length -> %-improvement; winner = majority of 4 metrics.

## Checkpoint shapes (verified from the `.pt` files) — CRITICAL for federation
Models are **heterogeneous**, so plain FedAvg across all 8 is impossible as-is:

| Junction | in -> hidden -> out |
|---|---|
| 1213632636 | 3 -> 64 -> 2 |
| 442107275 | 3 -> 64 -> 2 |
| 343322679 | 10 -> 128 -> 2 (richer) |
| 481750173 | 10 -> 128 -> 2 (richer) |
| GS_cluster_1269… | 4 -> 64 -> 2 (1 neighbor) |
| cluster_11594569878… | 5 -> 64 -> 2 (2 neighbors) |
| #4more | 6 -> 64 -> 3 (2 neighbors) |
| #3more | 7 -> 64 -> 5 (1 neighbor, 5 actions after merge) |

Only compatible same-architecture groups can be averaged directly: **{1213632636, 442107275}** (3-64-2) and
**{343322679, 481750173}** (10-128-2). Anything else needs a design decision (see below).

## Design notes / decisions pending for the Regional FL layer
- The regional aggregator needs a **homogeneous** set of nodes. Options to discuss with the user before coding:
  (a) regions = groups with identical architecture (the two pairs above); (b) force a common state/action space (padding/masking, uniform
  thin state, fixed action count) for demo nodes; (c) average only shared/compatible layers (e.g. hidden-hidden layer). Don't silently pick one.
- Likely shape of the implementation: a `RegionalAggregator` (FedAvg, weighting by local sample/step count; keep a `region_id` and a
  `get_regional_model()` hook so a future global aggregator can consume several regional state_dicts) + a federated-round driver that
  wraps `SingleJunctionAgent` (`get_state_dict`/`load_state_dict` of `policy_net`, then re-sync `target_net`). Do not modify DQN behaviour.
- Federated round needs a **local-training primitive that can be called per round** — currently `train_one_junction` runs a full
  multi-episode loop and saves a file; it needs to be factored (or a new function written reusing the agent) to run N episodes/steps and
  return state without restarting from scratch. Replay buffers and optimizers stay local (never aggregated); decide whether epsilon
  continues across rounds (probably yes).
- Training environment choice: (i) each node trains alone vs fixed-time bystanders (as `train_one_junction`), or (ii) all region nodes live
  in one shared sim (as `train_3more_vs_dqn_neighbors.py` / `eval_all_simultaneous.py`, where each agent runs on a shared clock).
  Since all 8 junctions are in one SUMO network, "raw data stays local" = only state_dicts cross the node boundary; keep that boundary explicit.
- Comparison for the demo: Independent DQN (existing canonical flow) vs Regional FL DQN, evaluated with the existing epsilon=0 pipeline
  (`eval_one.py`-style / `eval_all_simultaneous.py`) and the same 4 metrics. Use tagged filenames (e.g. `fl_`) so canonical `.pt`/`.xml` are
  never overwritten (this repo's established convention: challenger tags, promote only after comparison).
- Note `get_state` mutates `prev_queues` (richer mode) and checkpoints save only `policy_net` — keep that in mind when snapshotting/restoring.

## Conventions / gotchas
- Match the existing style: heavy explanatory docstrings/comments describing *why* and recorded failed experiments; module-level constants;
  scripts import `dqn_traffic_combined as m`; per-junction behaviour via override dicts + `*_for(tls_id)` helpers.
- Reproducibility matters: re-seed (`SEED=42`) before runs; evaluation at epsilon=0 is exactly reproducible, training-episode metrics are not.
- `torch.load(..., weights_only=True)` is used for checkpoints.
- Neighbor feature ordering (sorted by upstream tls_id) is baked into each network's input layer — train and eval must use the same `neighbor_specs`.
- `apply_phase_merge()` and any custom program must be re-applied after **every** `traci.start()`.
- `apply_action()` advances the sim itself (single-agent owner of the clock); multi-agent scripts instead use their own shared-clock state machine.
- Don't commit generated `before_*.xml`/`seq_after_*.xml`; `README_REGIONAL_FL.md` is currently untracked.
- Commit attribution: end commits with the `Co-Authored-By` trailer given by the harness; only commit when asked.
