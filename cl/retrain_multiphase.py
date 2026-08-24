"""One-off: retrain only the two multi-phase (3- and 6-action) junctions
after adding the fairness bound in dqn_traffic_combined.py. The 2-action
junctions are provably unaffected by that change (fairness_bound reduces
to exactly the old MAX_GREEN force-switch for n_actions=2), so they don't
need to be retrained.
"""
import dqn_traffic_combined as m

TARGETS = [
    "cluster_10291078963_10291078964_11497411790_11497411791_#4more",
    "cluster_11801662689_12136946746_12136946747_12136946748_#3more",
]

config = m.discover_controllable_tls()
for tls_id in TARGETS:
    info = config[tls_id]
    m.train_one_junction(tls_id, info["green_phases"], info["lane_groups"])
