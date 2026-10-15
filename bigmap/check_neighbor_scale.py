"""
Verification stage 2 (see plan): prints raw, UNNORMALIZED
traci.lane.getLastStepVehicleNumber() sums for every discovered
neighbor-outflow lane group, sampled across a full-hour run with every
junction on the realistic fixed-time baseline (matching what a
training episode's bystanders actually look like). Used to sanity-check
NEIGHBOR_OUTFLOW_SCALE=20.0 before it's baked into any checkpoint - if
these sums cluster far from the 0-20 range the existing queue features
use, the divisor should be adjusted first.
"""
import traci

import dqn_traffic_combined as m
import discover_neighbor_topology as nt

config = m.discover_controllable_tls()
topology = nt.discover_neighbor_outflows(config)

pairs = [(tls_id, neighbor, lanes)
         for tls_id, neighbors in topology.items()
         for neighbor, lanes in neighbors]

traci.start(["sumo", "-c", "combined.sumocfg", "--no-step-log",
             "--duration-log.disable", "--seed", str(m.SEED)])
for tls_id in config:
    m.apply_phase_merge(tls_id)
    info = config[tls_id]
    m.install_realistic_program(tls_id, info["green_phases"], info["lane_groups"])

samples = {p[:2]: [] for p in pairs}
step = 0
while traci.simulation.getTime() < 3600:
    traci.simulationStep()
    step += 1
    if step % 5 == 0:
        for tls_id, neighbor, lanes in pairs:
            total = sum(traci.lane.getLastStepVehicleNumber(l) for l in lanes)
            samples[(tls_id, neighbor)].append(total)
traci.close()

print(f"{'From -> To':<90}{'min':>6}{'mean':>8}{'max':>6}")
for (tls_id, neighbor), vals in samples.items():
    label = f"{tls_id} -> {neighbor}"
    mn, mx = min(vals), max(vals)
    mean = sum(vals) / len(vals)
    print(f"{label:<90}{mn:>6}{mean:>8.2f}{mx:>6}")
