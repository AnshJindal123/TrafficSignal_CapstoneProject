# Phase 3 Progress Note — New Junction Integration

## What this covers
Integrating the new map (`map.osm`) into the existing FRL-STL pipeline (SUMO + TraCI + DQN,
same as Phase 2's Silk Board / Majestic / Hosa Road / Byapanahalli runs), and getting a real
before/after comparison on it.

## 1. Map conversion
`map.osm` covers a ~0.6 km × 1.2 km area (lat 12.9813–12.9868, lon 77.6318–77.6427) with
9 OSM `traffic_signals` nodes. After `netconvert` with junction-joining enabled, these collapse
into **3 signalised junctions** in the SUMO network. The busiest one —
`cluster_11594569878_11909580134_3352035441_6901551440` (6 controlled lanes, the same
4-phase G/y/G/y structure as your Silk Board TLS) — was used as the main junction, matching
the single-junction methodology from Chapter 8–9.

```
netconvert --osm-files map.osm -o map.net.xml \
    --geometry.remove --roundabouts.guess --ramps.guess \
    --junctions.join --tls.guess-signals --tls.discard-simple --tls.join
```

## 2. Traffic demand
`randomTrips.py` with a period of 4s over the 3600s window gives ~900 vehicles — light-to-moderate
demand appropriate for this smaller junction (your original `-p 1.2` setting, sized for Silk Board,
gridlocked this smaller network and produced meaningless numbers — teleports everywhere,
network-wide mean speed of 0.08 m/s. Worth remembering: `-p` needs to be re-tuned per map, not
reused as-is).

## 3. A real bug found and fixed
Running your existing `dqn_traffic.py` logic unchanged on this junction, the DQN agent performed
**far worse** than the fixed-timer baseline (avg waiting time roughly 20x worse). Root cause:

```python
# original choose_action()
if dir1 > dir2 * 1.5:
    return 0
if dir2 > dir1 * 1.5:
    return 2
```

This "fairness override" runs *before* the `MIN_GREEN` check and compares queue **ratios** only.
On a light junction, queues sit at 1–3 vehicles per lane, where tiny noise (2 cars vs. 1 car) already
satisfies the 1.5x ratio. So the override was firing almost every decision step, forcing constant
phase switches and burning most of the cycle in yellow transitions instead of actually serving
traffic. This didn't show up at Silk Board because its queues are large enough that ratios are
stable — it's a robustness gap that only appears at lower traffic volumes.

**Fix:** require both the ratio *and* an absolute minimum imbalance (3 vehicles) before overriding,
and check `MIN_GREEN` first unless the imbalance is real:

```python
MIN_IMBALANCE = 3 / 20.0  # normalized, matches get_state()'s /20 scaling

significant_imbalance = (
    (dir1 > dir2 * 1.5 and (dir1 - dir2) >= MIN_IMBALANCE) or
    (dir2 > dir1 * 1.5 and (dir2 - dir1) >= MIN_IMBALANCE)
)

if phase_time < MIN_GREEN and not significant_imbalance:
    return current_phase
```

## 4. Multi-episode training
Single-episode training (as in `dqn_traffic.py`) only lets epsilon decay from 1.0 to ~0.05 by the very
end of the hour, so the agent is exploring almost the entire episode and never gets to actually use
what it learned. Restructured the script to run 8 episodes back-to-back, carrying the policy network
and epsilon across them, and only recording trip/summary/queue output on the final (mostly-exploiting)
episode. This is a natural fit for Phase 3's stated scope ("training and testing the model under
various different traffic scenarios").

## 5. Results (before = fixed-timer, after = trained DQN agent, final episode)

| Metric              | Baseline | DQN     | Improvement |
|----------------------|---------:|--------:|------------:|
| Avg waiting time (s) |     9.96 |    8.22 |     17.47%  |
| Avg travel time (s)  |    75.03 |   72.93 |      2.80%  |
| Avg time loss (s)    |    29.11 |   27.03 |      7.15%  |
| Junction queue       |     5.42 |    3.73 |     31.18%  |
| Mean speed (m/s)     |     5.63 |    7.87 |     39.79%  |
| Throughput (veh)     |      883 |     883 |      0.00%  |

Modest, consistent gains across the board, in the same range as your best Phase 2 junctions
(Byapanahalli, Majestic) — and unlike Hosa Road, no metric regressed. All 883 vehicles completed
their trips in both runs, so there's no gridlock skewing the numbers.

## Files
- `map.net.xml` — converted SUMO network
- `junction.sumocfg` — SUMO config
- `routes.rou.xml` / `trips.trips.xml` — generated demand
- `dqn_traffic.py` — fixed, multi-episode training script
- `logs.py` — before/after metrics comparison (same structure as your Phase 2 `logs.py`)
- `before_*.xml`, `after_*.xml` — raw SUMO outputs from this run
- `dqn_policy.pt` — trained policy weights
- `improvement_chart.png` — results chart, same style as Figure 9.1–9.4

## Next steps (for the rest of Phase 3)
- Run this same fixed script against your existing 4 maps to confirm the fairness-override bug
  wasn't silently costing performance there too, especially Hosa Road (which showed regressions).
- Wire this junction into the RegionalController/GlobalController layer from Chapter 6 once a
  second junction is added, to start exercising the FedAvg aggregation path instead of a single
  standalone agent.
- Consider increasing `NUM_EPISODES` further and logging reward-per-episode to a CSV so you can
  plot a learning curve — useful evidence for the viva that the agent is actually converging, not
  just landing on a lucky final episode.
