# CLAUDE.md — cl_neighbor (Federated RL for Smart Traffic Lights)

Capstone project: **Federated Reinforcement Learning for Smart Traffic Lights using SUMO + TraCI + PyTorch DQN.**
This folder (`cl_neighbor`) holds 8 per-junction DQN agents on one combined SUMO map, a **unified-interface DQN**
(one network shape for every junction) and a **regional federated learning layer**. Branch: `unified_fl`.
Original spec (a rough guide, not a strict rule): [README_REGIONAL_FL.md](README_REGIONAL_FL.md).
Scope note: this 8-junction map is a deliberately small FIRST area. Next step is a bigger map with more nodes; keep everything
extensible (several regions, later a global aggregator over regional models) and do not hard-code "8 nodes / one region".

## Status (what exists and what it showed)
- **Unified DQN** (`unified_dqn.py`) works and **matches the original per-junction DQNs** within noise.
- **Regional FL** (aggregator, rounds, grouping into regions, joint + solo training regimes) works end to end, parameters only.
- **On this small map federation gave no measurable gain** over independent training (see Results). Reasons we believe, not proven:
  5 of 8 junctions are nearly idle, rules (min/max green, imbalance, fairness) make ~80-90% of decisions, and each busy node already
  learns well alone. Expanding with **busy, similar, adjacent** junctions is the next experiment.
- Not built (deliberately): the Global Aggregator. `RegionalAggregator.get_regional_model()` returns a plain state_dict per region for it.

## Environment
- Windows, Python + torch (CPU) + traci/sumolib. `SUMO_HOME` must be set (`C:\Program Files (x86)\Eclipse\Sumo\`); modules `sys.exit`
  on import without it. Scripts call the `sumo` binary and must be **run from this folder** (relative paths).
- No requirements file, no tests. Verification = SUMO sims (1 sim-hour = 3600 steps) + metrics. Run long jobs in the background.
- Compute: one joint training episode ~3-8 min depending on parallel load; solo regime = 8 sims per episode, ~4 GB RAM per seed.
  Many parallel sims are fine on 16 cores; solo-regime SUMO start-up has a retry guard (a connection race once killed a seed).
- Git-ignored: `before_*.xml`, `seq_after_*.xml`, `fl_eval_*.xml`, FL checkpoints (`dqn_policy_fl_*`, `dqn_policy_independent_*`,
  `regional_model_*`), run logs, `ms_results/`, `ms_logs/`, `ms_eval_tmp/`, `diag/`. Checkpoints are map-specific: retrain on a new map.

## Key files
| File | Role |
|---|---|
| `dqn_traffic_combined.py` (`m`) | **Core.** Original `DQN`, `SingleJunctionAgent` (heuristics + DQN in `choose_action`, reward, Double DQN), `discover_controllable_tls()`, override tables, fixed-time baseline programs. Has a `_queues_from_state` hook + optional `hidden_size` so subclasses reuse it. |
| `unified_dqn.py` | **Unified interface.** `UnifiedDQN` + `UnifiedJunctionAgent` (see below). `UNIFIED_VERSION` env: `v2` default, `v1` (worse, kept for reference). |
| `federated_regional.py` (`fr`) | `RegionalAggregator`, federated round driver, **joint regime** (all agents in one shared sim), grouping by load, seeds/tags via env vars. `python federated_regional.py train <fl\|independent> [rounds] [episodes_per_round]` |
| `federated_solo.py` | **Solo regime**: each node is its own process, trains alone vs fixed-time neighbours (like the original checkpoints), only parameters go to the aggregator. |
| `eval_multiseed.py`, `summarize_multiseed.py` | Evaluate checkpoints (epsilon=0, all agents live) on one SUMO eval seed per run -> `ms_results/*.json`; summarize with paired differences + error bars. |
| `eval_fixed.py` | Fixed-time reference on the same eval seeds (also feeds `groups_by_load`). |
| `eval_fl_compare.py` | Older 5-seed independent-vs-FL comparison; its helpers are reused by the scripts above. |
| `headroom_check.py`, `diag_343.py` | Read-only diagnostics: share of decisions made by DQN vs rules; per-decision log at one junction. |
| `discover_neighbor_topology.py` | Neighbour-inflow features (`discover_neighbor_inflows`, 700 m, Dijkstra over the net). |
| older pipeline | `train_all.py`, `train_3more_vs_dqn_neighbors.py`, `eval_all_simultaneous.py`, `eval_one.py`, `reeval_canonical_all.py`, `generate_realistic_baseline.py`, `logs_combined.py`, `pick_best_general.py`: how the original ("canonical", `dqn_policy_seq_*.pt`) checkpoints were made. |
| `combined.net.xml`, `combined*.rou.xml`, `combined*.sumocfg` | The map + demand (`combined_demand_{m30,m15,p15,p30,p50}` = scaled variants). |

## Original system (unchanged)
- Controllable junction = TLS with >=2 real green phases (8 of 18 on this map). One `SingleJunctionAgent` each: Adam lr 1e-3, replay 5000,
  Double DQN, batch 64, gamma 0.95. Action = next green phase; decision every 5 sim-s; switching inserts 5 s yellow.
- `choose_action` is hybrid: starvation override, min_green 20 / max_green 60, significant-imbalance rule, and only otherwise the network.
  Per-junction overrides: `REWARD_WEIGHTS_OVERRIDES`, `IMBALANCE_OVERRIDES`, `PHASE_MERGE_TLS` (#3more merges phases 0&2), `RICHER_STATE_TLS`.
- Original per-junction nets have **different shapes** (input 3-10, hidden 64/128, outputs 2/3/5), which is why only one layer could be shared.
- Reward `-(qw*queue^p + ww*wait) + 0.02*(prev_queue - queue)`. Neighbour feature order (sorted upstream tls id) is baked into inputs.

## Unified interface (`unified_dqn.py`, v2)
- State = per phase `[queue/20, wait/300, speed/15, trend/20, is_current, lane_count/4, cycle_position/4, time_since_served/300]` + context
  `[phase_time/MAX_GREEN, summed neighbour inflow]`. Thin-state junctions get wait/speed/trend/phase_time zero-filled (information parity
  with the original: richer features were measured to hurt them).
- Net: shared per-phase encoder + shared head on `[phase embedding, mean embedding]` -> one Q per phase. Parameters do not depend on the
  number of phases (~13.5k params), so **every node has identical shapes and the WHOLE network is federated**.
- `phase_embed` (per-node phase identity) is **local, never aggregated** (`LOCAL_KEY_PREFIXES`). v1 lacked phase identity and broke #3more.
- Reuses all agent logic through `SingleJunctionAgent`; only state encoding + network differ.
- **Expansion gotchas:** `MAX_PHASES = 5` (size of `phase_embed`): a junction with more than 5 green phases will crash or must raise it
  (changes shapes -> retrain). Neighbour inflow is a single summed number (a 2-slot variant was considered, expected effect small).

## Running (env vars read by `federated_regional.py`)
`FL_DESIGN=current|unified|canonical` (canonical = eval only), `UNIFIED_VERSION=v2`, `FL_REGIME=joint|solo`, `FL_SEED=<n>` (training seed),
`FL_EVAL_SEED=<n>` (SUMO seed for evaluation), `FL_GROUPING=none|load`, `FL_WEIGHTING=steps|reward` (default steps for unified), `MS_MODES=...`.
Checkpoint names: `dqn_policy_{mode}{design tag}{_s<seed>}_{tls}.pt`; tags `_u2` (unified v2), `_solo`, `_g` (grouped); seed 42 has no suffix.
```
FL_DESIGN=unified FL_SEED=7 python federated_regional.py train fl            # joint regime, one region
FL_DESIGN=unified FL_SEED=7 FL_GROUPING=load python federated_regional.py train fl   # two regions (busy / light)
FL_DESIGN=unified FL_REGIME=solo FL_SEED=7 python federated_solo.py fl       # solo regime
FL_EVAL_SEED=11 python eval_fixed.py                                         # fixed-time reference (+ grouping data)
FL_DESIGN=unified FL_SEED=7 FL_EVAL_SEED=11 MS_MODES=independent,fl python eval_multiseed.py
python summarize_multiseed.py
```
Defaults: 5 federated rounds x 4 episodes + 1 final local-only round, blend 0.5 (node moves halfway to its region's average), same
seeds/budget for `independent` and `fl` so they differ only by aggregation. Unified runs start every node from one common initial model.
`FL_GROUPING=load` needs `ms_results/fixed_e*.json` (run `eval_fixed.py` on the new map first); nodes with fixed-time avg queue > 20 m are "busy".

## Results on the 8-junction map (avg wait, seconds; pooled; error bars from paired differences over eval seeds)
- 20 eval seeds (error ~2-4 s): fixed-time 213.2 | canonical 195.8 | unified independent (joint) 198.1 | unified global FL 199.6 | unified grouped FL 199.1.
  All DQN variants beat fixed-time by ~6-8%; unified = canonical within noise; **FL (global or grouped) = independent within noise**.
- Solo regime (2 seeds, 20 eval seeds): unified independent 197.9 vs canonical 195.8 (equal); solo FL 203.0 (slightly worse).
- Only repeatable FL effect: small queue gain at #3more; independent training already gets most of it. Global FL slightly lowers trips (~1%).
- **Evaluation noise is large**: 5 eval seeds gave +/-7-8 s error and produced several false "gains"/"gaps". Use >=20 eval seeds and paired
  comparisons. Junction 343322679 is chaotic (queue 143-415 m for the same model); no controller is distinguishable from fixed-time there.

## Ideas not yet tested
Data efficiency (short training budgets, FL vs independent), new-node transfer (train a regional model on N-1 nodes, fine-tune the last),
varied-demand training, FL on a larger map with many busy adjacent junctions, 2-slot neighbour inflow, per-junction overrides for unified.

## Conventions / gotchas
- Match the existing style: heavy explanatory docstrings describing *why* and recorded failed experiments; module-level constants.
- `torch.load(..., weights_only=True)`; `apply_phase_merge()` and any custom program must be re-applied after **every** `traci.start()`.
- `apply_action()` owns the sim clock; multi-agent scripts use their own shared-clock state machine (`Runner` objects).
- Evaluation is epsilon=0 and reproducible per eval seed; training-episode metrics are not.
- Raw traffic data never crosses a node boundary: only shared-layer state_dicts go to the aggregator.
- Commit attribution: end commits with the `Co-Authored-By` trailer given by the harness; only commit/push when the user asks.
