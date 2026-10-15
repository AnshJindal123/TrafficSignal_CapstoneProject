# bigmap: unified regional FL on the merged 9-tile map

The `unified_fl` code (`unified_dqn.py` v2, `federated_regional.py`, joint
regime) run on a **bigger map** built from 9 overlapping OSM tiles. Each
script is a copy of `cl_neighbor/` from the `unified_fl` branch, and every
change from that copy is listed below. Run everything **from this folder**.
The scripts read `combined.sumocfg` / `combined.net.xml` by relative path, so
here those names hold the merged map, not the old 8-junction one.

**First, after cloning:** `python setup_map.py`. The map (33 MB) and demand
(5 MB) are committed gzipped to keep the repo small, and this unpacks them to
`combined.net.xml` / `combined.rou.xml` (both git-ignored).

Kept to the essentials:
- **Scripts:** only those the unified FL pipeline uses. The old 8-junction pipeline (`train_all.py`, `eval_one.py`, charts, ...) stays in `cl_neighbor/` only.
- **Checkpoints:** the final `fl_u2` and `independent_u2` checkpoints plus the regional model. The `fl_preagg_u2` checkpoints were not committed, but their eval results are in `ms_results/` and `summary_bigmap.txt`.
- **Map source:** the 9 source `.osm` tiles (84 MB) are not committed. The netconvert command below rebuilds the net from them.

## The map

| | old 8-junction map | merged map (this folder) |
|---|---|---|
| source | 4 separate OSM maps, joined as disconnected islands | 9 overlapping OSM tiles of one contiguous area, lat 12.952-12.996, lon 77.621-77.678 (Indiranagar / Domlur / HAL) |
| passenger road | 247 lane-km | 674 lane-km |
| controllable junctions (>=2 green phases) | 8 | **20** |
| phase counts | 2, 3, 5 | 2 (x7), 3 (x7), 4 (x4), 6 (x1), 7 (x1) |
| junctions with a neighbour signal within 700 m | - | 15 of 20 |

The map overlaps the old map, and 5 old junctions reappear:
- **Same ID:** `1213632636`, `GS_cluster_12692472176...`, `cluster_10291078963...#4more` and `cluster_11594569878...`.
- **Renamed by the join:** old `#3more` is now `cluster_11801662689..._#6more`, with 6 phases and no phase merge, and old `481750173` is now `cluster_481750173_6440019886`.
- The old per-junction overrides (`PHASE_MERGE_TLS`, `RICHER_STATE_TLS`, `IMBALANCE_OVERRIDES`, `REWARD_WEIGHTS_OVERRIDES`) are keyed by the old IDs, so here they apply to no junction. Every junction runs on defaults.

Built with the same netconvert options as the project's original `map.net.xml`,
plus `--no-turnarounds` as in `combined.net.xml`:
```
netconvert --osm-files "map.osm,map (1).osm,...,map (8).osm" -o combined.net.xml --proj.utm \
  --geometry.remove --roundabouts.guess --tls.discard-simple --tls.join --tls.guess-signals \
  --ramps.guess --junctions.join --no-turnarounds
```

## Demand (calibrated, see run_baseline.py)

`randomTrips.py -e 3600 -p 0.19 --seed 42 --vclass passenger --validate`
produced 18,948 trips, the old map's density scaled by road length. It
gridlocked the network, because random trips across a 9 km map are much
longer. So the demand was thinned by keeping every k-th vehicle. These are
the fixed-time baseline numbers (55-65 s program on all 20 junctions, every
scheduled vehicle counted):

| demand | vehicles | finished | teleports | avg wait (all vehicles) |
|---|---|---|---|---|
| full | 18,948 | 18% | 7,267 | 1,174 s (gridlock) |
| 1/2 | 9,474 | 44% | 1,285 | 755 s |
| **1/3 (used)** | **6,316** | **61%** | **382** | **520 s** |
| old 8-junction map, for reference | 7,086 | 69% | 1,203 | 489 s |

`combined.rou.xml(.gz)` is the 1/3 demand. To regenerate it, run randomTrips
with the command above, then keep every 3rd `<vehicle>` (index % 3 == 0).

## Code changes vs `unified_fl/cl_neighbor`

- `unified_dqn.py`: `MAX_PHASES` raised from 5 to 7, because one junction here has 7 green phases. This only changes the shape of `phase_embed`, which is local and never aggregated. The shared encoder and head are unchanged.
- `eval_fl_compare.py`:
  - New `junction_queues()` reads the queue xml once for all junctions. The old `junction_queue()` re-parsed the file per junction, which doesn't scale to 20 junctions; it is still there.
  - `FAIR_ARGS` and extra fields in `trip_metrics()`: SUMO's tripinfo only lists **finished** trips by default. A controller that clears more hard-case vehicles therefore looks worse on `wait`. The original fields are kept as they were, and the new `vehicles`, `wait_all` and `delay_all` fields count every scheduled vehicle (finished, still on the road, or never departed).
- `eval_multiseed.py` and `eval_fixed.py`: pass `FAIR_ARGS` to SUMO and use `junction_queues()`.
- `summarize_multiseed.py`:
  - Prints `wait_all` and `delay_all` when they are present.
  - The busy junctions are now the 5 with the longest fixed-time queue, instead of the 4 hard-coded old-map junctions.
  - Adds the `fixed` vs `independent_u2` / `fl_u2` pairs.
- New `run_baseline.py`: the fixed-time demand-calibration run.

## What was run

```
python setup_map.py                                                # once
FL_DESIGN=unified python federated_regional.py train fl            # train_fl_u2.log
FL_DESIGN=unified python federated_regional.py train independent   # train_independent_u2.log
FL_EVAL_SEED=<11|22|33|44|55> python eval_fixed.py
FL_DESIGN=unified FL_EVAL_SEED=<11|22|33|44|55> MS_MODES=independent,fl,fl_preagg python eval_multiseed.py
# (drop fl_preagg from MS_MODES unless you retrain - those checkpoints aren't committed)
python summarize_multiseed.py > summary_bigmap.txt
```
Defaults throughout: training seed 42, 5 federated rounds x 4 episodes + 1
final local round, blend 0.5, one region of all 20 nodes, step weighting.

## Results

Training seed 42 only, evaluated on 5 SUMO eval seeds (11, 22, 33, 44, 55)
with epsilon=0 and all 20 junctions live. Means over the 5 eval seeds; the
full table is in `summary_bigmap.txt`.

| | fixed-time | unified independent | unified FL | FL before last aggregation |
|---|---|---|---|---|
| avg wait, all 6,316 vehicles (s) | 524.6 | 294.7 | **286.7** | 312.3 |
| avg wait, finished trips only (s) | 297.6 | 185.0 | **175.1** | 190.3 |
| time loss, finished (s) | 384.4 | 279.3 | **268.0** | 283.4 |
| trips finished | 3,774 | 4,615 | **4,628** | 4,532 |
| teleports | 397 | 171 | **160** | 190 |

- **DQN vs fixed time: big, clear win.** Both unified variants cut the all-vehicle wait by about 44-45%, finish about 22% more trips (+840) and halve teleports. On the 8-junction map the gain was about 6-8% (`cl_neighbor/CLAUDE.md`), so this map has far more room for learned control.
- **FL vs independent: within noise.** Across the 5 eval seeds FL's mean is slightly better: -7.9 s all-vehicle wait (-2.7%), +13 trips, -10 teleports. But per eval seed, FL minus independent wait_all is -1.6, +2.7, +25.3, -15.1 and -50.9 s, so FL is better on 3 seeds and worse on 2. With one training seed, this is not a demonstrated FL gain. It matches the 8-junction finding (FL = independent within noise).
- **The final local round matters.** The models saved just before the last aggregation (`fl_preagg`) are worse than both, so averaging alone pulls each node away from its local optimum. The final local-only round recovers it.
- **Busy junctions, average queue (m), fixed / indep / FL:**
  - `cluster_11801662689..._#6more` (old #3more, now 6 phases): 1035 / 311 / 287
  - `#4more`: 233 / 13 / 13
  - `cluster_11187974841..._#8more`: 192 / 123 / 122
  - `cluster_11155831177..._#8more`: 141 / 32 / 32
  - `GS_cluster_11003327957...`: 132 / 84 / 92

### Suggested next steps
1. **More training seeds** (`FL_SEED=7`, `123`, `2024`; about 2.5 h per mode when fl and independent run in parallel). Then summarize_multiseed's paired differences get real standard errors, and this decides whether FL's small edge is real.
2. **Two regions** (`FL_GROUPING=load`): `ms_results/fixed_e*.json` is already here, and this map has about 8 busy and 12 light nodes.
3. **Solo regime** (`federated_solo.py`) on this map.
4. **Per-junction overrides for the new IDs**: none apply here, as noted above. In particular, old #3more's phase merge doesn't carry over to its new 6-phase ID.
