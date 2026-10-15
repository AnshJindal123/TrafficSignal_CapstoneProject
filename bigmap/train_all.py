"""
Retrains the 4 junctions that actually RECEIVE a neighbor-inflow feature
under the corrected topology direction (discover_neighbor_topology.py's
discover_neighbor_inflows(), not discover_neighbor_outflows()): #4more,
#3more, GS_cluster, cluster_11594569878... - each of these has at least
one upstream neighbor whose outgoing lanes reach it within
NEIGHBOR_MAX_DISTANCE_M, so their state_size (input layer) grows versus
../cl_retrain and needs a fresh checkpoint.

CORRECTION (2026-09-01): the first version of this feature attached each
junction's own outgoing-lane traffic to ITS OWN state - self-referential
noise, not "data received from a neighbor." Fixed by inverting the
mapping (discover_neighbor_inflows): a discovered edge A -> B now feeds
its scalar to B (the actual downstream receiver), not A. Consequence:
1213632636 - which has an outgoing edge to a neighbor but nothing
feeding INTO it - correctly drops OUT of this list entirely; its
../cl_retrain canonical checkpoint (unaugmented state, since it now has
zero neighbors) is copied forward unchanged instead of being retrained,
same as 343322679/442107275/481750173 (which have zero neighbors either
direction).

Trained directly against the current, final baseline: 55-65s compulsory
flat green / 5s yellow fixed-time bystanders, global (20,60) green
bounds, tightened IMBALANCE_OVERRIDES for 442107275/481750173 (inherited
from dqn_traffic_combined.py, unaffected by this feature). Episode/
weight schedule reused unchanged from ../cl_retrain/train_all.py.

Note: ../cl_retrain's current #3more canonical is actually the
'vsdqn_'-promoted joint-training checkpoint (trained against real
frozen DQN neighbors), not a plain bystander-trained one. This run
produces a fresh bystander-trained #3more baseline for the new (now
correctly-directed) state shape - the vsdqn_-style joint round is
reproduced afterward in train_3more_vs_dqn_neighbors.py once every
other junction's neighbor-shaped checkpoint exists.

Written directly to canonical filenames (tag="").
"""
import shutil

import dqn_traffic_combined as m
import discover_neighbor_topology as nt

config = m.discover_controllable_tls()
neighbor_topology = nt.discover_neighbor_inflows(config)

# 1213632636 has zero upstream neighbors under the corrected topology -
# same architecture as ../cl_retrain, so reuse its checkpoint/results
# directly instead of retraining.
for kind in ["tripinfo", "summary", "queue"]:
    shutil.copy(f"../cl_retrain/seq_after_{kind}_1213632636.xml", f"seq_after_{kind}_1213632636.xml")
shutil.copy("../cl_retrain/dqn_policy_seq_1213632636.pt", "dqn_policy_seq_1213632636.pt")
print("copied 1213632636's ../cl_retrain checkpoint/results forward unchanged (zero upstream neighbors)")

# (tls_id, episodes, reward_weights)
TARGETS = [
    ("GS_cluster_12692472176_12692472177_12692472178_12692472179", 45, (0.01, 0.001, 1)),
    ("cluster_10291078963_10291078964_11497411790_11497411791_#4more", 30, (0.01, 0.001, 1)),
    ("cluster_11594569878_11909580134_3352035441_6901551440", 30, (0.01, 0.001, 1)),
    ("cluster_11801662689_12136946746_12136946747_12136946748_#3more", 60, (0.005, 0.002, 1)),
]

for tls_id, episodes, weights in TARGETS:
    info = config[tls_id]
    specs = neighbor_topology[tls_id]
    print(f"\n### tag='' episodes={episodes} weights={weights} "
          f"green_bounds={m.green_bounds_for(tls_id)} "
          f"imbalance={m.imbalance_for(tls_id)} "
          f"neighbors={[n for n, _ in specs]} on {tls_id} ###\n")
    m.train_one_junction(tls_id, info["green_phases"], info["lane_groups"],
                          num_episodes=episodes, reward_weights=weights, tag="",
                          bystander_config=config, neighbor_specs=specs)

print("\n=== train_all.py finished all runs ===")
