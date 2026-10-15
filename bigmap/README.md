# bigmap: unified regional FL on a 20-junction map

The unified DQN + regional federated learning code from `cl_neighbor/`, run on
a larger map merged from 9 overlapping OSM tiles (Indiranagar / Domlur / HAL
area, Bangalore).

## Setup

Run everything from this folder. Unpack the map and demand once:

```
python setup_map.py
```

## Map and demand

- 9 OSM tiles merged with netconvert (same options as the original `map.net.xml`) into one network.
- **20 controllable junctions**, with 2-7 green phases each.
- Demand: 6,316 random passenger trips over 1 hour (`combined.rou.xml`). The full density (18,948 trips) gridlocked, so it was thinned to 1/3. Under fixed-time control, 61% of the trips finish.

## Changes from `cl_neighbor/`

- `unified_dqn.py`: `MAX_PHASES` raised from 5 to 7 (one junction has 7 phases).
- `eval_fl_compare.py`: adds a one-pass queue parser and `wait_all` (avg wait over **all** vehicles, including those unfinished at 3600 s).
- `eval_multiseed.py`, `eval_fixed.py`, `summarize_multiseed.py`: use the above. The busy junctions in the summary are picked from the fixed-time results.
- `run_baseline.py` (new): the fixed-time run used to calibrate the demand.

## Commands

```
FL_DESIGN=unified python federated_regional.py train fl
FL_DESIGN=unified python federated_regional.py train independent
FL_EVAL_SEED=11 python eval_fixed.py
FL_DESIGN=unified FL_EVAL_SEED=11 MS_MODES=independent,fl python eval_multiseed.py
python summarize_multiseed.py
```

Repeat the eval commands for eval seeds 11, 22, 33, 44 and 55. Training takes
about 2.5 h for 5 rounds x 4 episodes + 1 final local round.

## Results (training seed 42, 5 eval seeds)

| | Fixed-time | Independent | FL |
|---|---|---|---|
| Avg wait, all vehicles (s) | 524.6 | 294.7 | 286.7 |
| Trips finished | 3,774 | 4,615 | 4,628 |
| Teleports | 397 | 171 | 160 |

- Both DQN variants cut the average wait by about 45% vs fixed-time and finish about 22% more trips.
- FL vs independent is within noise: FL is better on 3 of 5 eval seeds. It needs more training seeds to tell.
- The trained checkpoints (`dqn_policy_fl_u2_*`, `dqn_policy_independent_u2_*`) are included. Full numbers are in `summary_bigmap.txt`.

## Next steps

1. More training seeds (`FL_SEED=7`, `123`, `2024`) to settle FL vs independent.
2. Two regions with `FL_GROUPING=load` (busy / light). The fixed-time results it needs are already in `ms_results/`.
3. Solo regime (`federated_solo.py`) on this map.
