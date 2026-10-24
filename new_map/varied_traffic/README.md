# Varied-traffic training (74-junction map, 0.35 demand)

DQN (independent) vs DQN+FL, both trained on **4 different traffic seeds**
instead of the single seed-42 demand used by the earlier runs in `new_map/`.

## Setup

| | |
|---|---|
| Network | unified DQN v2 (`FL_DESIGN=unified UNIFIED_VERSION=v2`), 74 junctions, joint simulation |
| Episodes | 24 = 5 rounds x 4 episodes + 1 final local-only round of 4 |
| Training traffic | 4 blocks of 6 episodes: traffic seed **42 -> 7 -> 123 -> 2024** (`FL_TRAIN_VARY=1`) |
| Per traffic seed | SUMO `--seed` = that seed, and that seed's own random 35% subset of `combined.rou.xml` (`thin_demand.random_subset`, ~3,830-3,890 vehicles) |
| FL | FedAvg, load-grouped regions (busy / light, `FL_GROUPING=load`), blend 0.5, step weighting |
| Run seeds | 42 and 7 (`FL_SEED`: network init + exploration only; both runs see the same traffic schedule) |
| Evaluation | greedy, `demand_0.35.rou.xml` (not a training demand), eval seeds 11, 22, 33, 44, 55 |

## Results (mean over 2 run seeds x 5 eval seeds)

| Metric | Fixed time | DQN | DQN+FL | DQN+FL vs fixed |
|---|---|---|---|---|
| Wait, all vehicles (s) | 505.8 | 257.3 | **241.5** | -52.3% |
| Wait, finished trips (s) | 261.0 | 125.2 | **117.3** | -55.0% |
| Travel time (s) | 640.5 | 533.7 | **527.8** | -17.6% |
| Time loss (s) | 358.1 | 231.9 | **225.0** | -37.2% |
| Teleports | 266 | 151 | **142** | -46.5% |
| Trips finished | 2361 | 2846 | **2868** | +21.5% |
| Total junction queue (m) | 1322 | 558 | **554** | -58% |

DQN+FL vs DQN, paired by run seed (wait, all vehicles):

| Run seed | DQN | DQN+FL | Change |
|---|---|---|---|
| 42 | 248.3 | 248.9 | +0.2% (tie, FL better in 3/5 eval seeds) |
| 7 | 266.3 | 234.1 | -12.1% (FL better in 5/5 eval seeds) |

Compared with the earlier single-seed runs (44 episodes on seed-42 traffic):
DQN+FL improved (wait 249.2 -> 241.5 s) with fewer episodes; DQN did not
change clearly (254.3 -> 257.3 s). Both runs changed episode count and
traffic variety at once, so the gain can't be attributed to either alone.

Caveat: only 2 run seeds - FL >= DQN in both, but the FL advantage rests on
one seed. More run seeds are needed before claiming it. Note that
`summary_all_models.txt` section (2) pairs each run against the *pooled* mean
of the other model, not the same run seed, so its "neg in 2/2 seeds" overstates
the FL vs DQN result; use the table above.

## Contents

| Folder | |
|---|---|
| `code/` | snapshot of every `new_map/*.py` + `combined.sumocfg` used for these runs |
| `models/` | `dqn_policy_{fl,fl_preagg,independent}_u2[_g]_v[_s7]_<junction>.pt` (74 per model; no suffix = run seed 42), `regional_model_region_main_{busy,light}_u2_g_v[_s7].pt` |
| `logs/` | training logs; `logs/eval/` per-evaluation logs |
| `results/` | per-run JSON (`t<run seed>_e<eval seed>_<model>.json`), fixed-time JSON, `summary_all_models.txt` |

`fl_preagg` (models just before the last aggregation) were saved but not evaluated.

## Reproduce (from `new_map/`, after `python setup_map.py`)

```sh
export FL_DESIGN=unified UNIFIED_VERSION=v2 FL_TRAIN_VARY=1
FL_SEED=42 FL_GROUPING=load python federated_regional.py train fl
FL_SEED=42                  python federated_regional.py train independent
# evaluate (per eval seed e; add FL_GROUPING=load for fl)
FL_SEED=42 FL_EVAL_SEED=11 MS_MODES=fl FL_GROUPING=load python eval_multiseed.py
python summarize_multiseed.py
```

The training demands (`demand_train_s{42,7,123,2024}.rou.xml`) are generated
automatically on first use and are not stored here. Evaluation scripts load
models from `new_map/` itself, so copy `models/*` back there to re-evaluate.
