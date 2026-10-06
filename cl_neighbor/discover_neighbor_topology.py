"""
Topology discovery: for each controllable junction, find its nearest
neighboring controllable junction(s) within NEIGHBOR_MAX_DISTANCE_M,
and the specific outgoing lanes at this junction whose traffic heads
toward that neighbor - the "vehicles leaving this signal toward the
next neighbour" data channel this feature is built around.

Two distinct data sources, deliberately kept separate:
  1. traci (live simulation) - to find each tls_id's OUTGOING lanes via
     getControlledLinks(), which discover_controllable_tls() in
     dqn_traffic_combined.py doesn't compute (it only tracks INCOMING
     lane_groups, used for the queue/wait/speed state features).
  2. combined.net.xml (static, parsed once) - to trace those outgoing
     edges forward through the network graph to the nearest other
     controllable junction. Only one direct one-hop connection exists
     among the 8 controllable junctions (1213632636 -> cluster_
     11594569878_11909580134_3352035441_6901551440); every other pair
     requires tracing through intermediate non-signalized junctions.

Junction node IDs in the XML do NOT always equal the tls_id string
(verified false for GS_cluster_12692472176_12692472177_12692472178_
12692472179, whose real junction node has no "GS_" prefix) - so this
module never looks up junction nodes directly, only follows
<connection tl="..."> attributes, which DO reliably match tls_id.

Distance (not hop count) is the cutoff unit: hop count to the same real
neighbor varied 1-10 in probing, while genuinely "close" pairs
clustered under ~700m regardless of hop count. NEIGHBOR_MAX_HOPS is
purely a runtime safety cap on the search, not the primary gate.

Known approximation, accepted as pragmatic (same spirit as this
codebase's existing getLastStepVehicleNumber-as-proxy design choices):
a lane's traffic may fan out at an intermediate priority junction
toward multiple eventual destinations, only one of which is the
tracked neighbor - the resulting outflow scalar is an upper-bound
proxy on "traffic actually heading to this neighbor," not an exact
per-vehicle route isolation.
"""
import heapq
import xml.etree.ElementTree as ET

import traci

import dqn_traffic_combined as m

NEIGHBOR_MAX_DISTANCE_M = 700.0
NEIGHBOR_MAX_HOPS = 20


def _outgoing_lanes_by_tls(config):
    """{tls_id: set(outgoing_lane_ids)}, one shared traci session (same
    DISCOVER_CMD/apply_phase_merge pattern as discover_controllable_tls())."""
    traci.start(m.DISCOVER_CMD)
    result = {}
    for tls_id, info in config.items():
        m.apply_phase_merge(tls_id)
        logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        phases = logic.phases
        links = traci.trafficlight.getControlledLinks(tls_id)
        out_lanes = set()
        for gp in info["green_phases"]:
            state = phases[gp].state
            for i, ch in enumerate(state):
                if ch not in ('G', 'g'):
                    continue
                for entry in links[i]:
                    out_lanes.add(entry[1])  # entry = (inLane, outLane, viaLane)
        result[tls_id] = out_lanes
    traci.close()
    return result


def _parse_net_graph(net_file):
    """edge_len[edge_id] = max lane length on that (non-internal) edge.
    adj[from_edge] = [(to_edge, tl_or_None), ...] from every <connection>.
    lane_to_edge[lane_id] = owning edge_id."""
    root = ET.parse(net_file).getroot()
    edge_len = {}
    lane_to_edge = {}
    for edge in root.findall('edge'):
        if edge.get('function') == 'internal':
            continue
        eid = edge.get('id')
        lengths = [float(lane.get('length')) for lane in edge.findall('lane')
                   if lane.get('length') is not None]
        edge_len[eid] = max(lengths) if lengths else 0.0
        for lane in edge.findall('lane'):
            lane_to_edge[lane.get('id')] = eid

    adj = {}
    for conn in root.findall('connection'):
        from_edge, to_edge = conn.get('from'), conn.get('to')
        if from_edge is None or to_edge is None:
            continue
        adj.setdefault(from_edge, []).append((to_edge, conn.get('tl') or None))

    return edge_len, adj, lane_to_edge


def _nearest_neighbors_from_edge(start_edge, edge_len, adj, tls_ids, exclude_tls):
    """Dijkstra-style search (priority queue by cumulative distance, not
    hop-count BFS) from start_edge. Returns {neighbor_tls: distance} for
    every controllable tls reached within the cutoff - expansion along a
    branch stops the moment it hits a controllable junction, since only
    the NEAREST one in each direction is the real neighbor."""
    found = {}
    visited = set()
    heap = [(0.0, start_edge, 0)]
    while heap:
        dist_to_start, edge, hops = heapq.heappop(heap)
        if edge in visited:
            continue
        visited.add(edge)
        if hops > NEIGHBOR_MAX_HOPS:
            continue
        dist_to_end = dist_to_start + edge_len.get(edge, 0.0)
        if dist_to_end > NEIGHBOR_MAX_DISTANCE_M:
            continue
        for to_edge, tl in adj.get(edge, []):
            if tl is not None and tl in tls_ids and tl != exclude_tls:
                if tl not in found or dist_to_end < found[tl]:
                    found[tl] = dist_to_end
                continue  # don't trace past a controllable junction
            if to_edge not in visited:
                heapq.heappush(heap, (dist_to_end, to_edge, hops + 1))
    return found


def discover_neighbor_outflows(config, net_file="combined.net.xml"):
    """Returns {tls_id: [(neighbor_tls_id, source_lane_ids), ...]}, sorted
    by neighbor_tls_id, one entry per neighbor found within
    NEIGHBOR_MAX_DISTANCE_M (possibly []). source_lane_ids are tls_id's
    own outgoing lanes (sorted) whose traced path reaches that neighbor.
    """
    tls_ids = set(config.keys())
    out_lanes_by_tls = _outgoing_lanes_by_tls(config)
    edge_len, adj, lane_to_edge = _parse_net_graph(net_file)

    result = {}
    for tls_id in config:
        neighbor_lanes = {}
        for lane in out_lanes_by_tls[tls_id]:
            edge = lane_to_edge.get(lane)
            if edge is None:
                continue
            found = _nearest_neighbors_from_edge(edge, edge_len, adj, tls_ids, tls_id)
            for neighbor in found:
                neighbor_lanes.setdefault(neighbor, set()).add(lane)
        result[tls_id] = [(neighbor, sorted(lanes))
                           for neighbor, lanes in sorted(neighbor_lanes.items())]
    return result


def discover_neighbor_inflows(config, net_file="combined.net.xml"):
    """Returns {tls_id: [(upstream_tls_id, upstream_lanes), ...]} - the
    direction actually needed for the neighbor-outflow FEATURE (as
    opposed to discover_neighbor_outflows()'s raw topology, which
    describes each junction's own outgoing lanes and where they lead).

    The feature is meant to give a junction early warning about traffic
    a neighbor just released toward it - "vehicles leaving the [upstream]
    signal and moving towards the next neighbour, so the next signal is
    better prepared." That means for a discovered edge A -> B (A's
    outgoing lanes reach B), the resulting scalar belongs on B's state,
    not A's - A doesn't learn anything new about a neighbor from
    watching its own outgoing lanes; B does. This inverts
    discover_neighbor_outflows()'s {A: [(B, A's lanes)]} mapping into
    {B: [(A, A's lanes)]} so each consumer is the actual downstream
    receiver.

    Consequence worth noting explicitly: a junction with outgoing edges
    to a neighbor (present in discover_neighbor_outflows()) does not
    necessarily have anything feeding INTO it - e.g. 1213632636 sends
    traffic onward but nothing within NEIGHBOR_MAX_DISTANCE_M sends
    traffic to it, so under this inflow mapping it correctly gets zero
    neighbors, not the one it had under the (wrong) outflow mapping.
    """
    outflows = discover_neighbor_outflows(config, net_file)
    inflows = {tls_id: [] for tls_id in config}
    for upstream_tls, neighbors in outflows.items():
        for downstream_tls, lanes in neighbors:
            inflows[downstream_tls].append((upstream_tls, lanes))
    for tls_id in inflows:
        inflows[tls_id].sort(key=lambda x: x[0])
    return inflows


if __name__ == "__main__":
    config = m.discover_controllable_tls()
    topology = discover_neighbor_outflows(config)
    print("=== outflow topology (raw, A's own outgoing lanes -> B) ===")
    for tls_id, neighbors in topology.items():
        if not neighbors:
            print(f"{tls_id}: no neighbors within {NEIGHBOR_MAX_DISTANCE_M}m")
            continue
        for neighbor, lanes in neighbors:
            print(f"{tls_id} -> {neighbor}: {len(lanes)} lane(s) {lanes}")

    print("\n=== inflow topology (what the feature actually uses: B receives A's lanes) ===")
    inflows = discover_neighbor_inflows(config)
    for tls_id, neighbors in inflows.items():
        if not neighbors:
            print(f"{tls_id}: receives from no one within {NEIGHBOR_MAX_DISTANCE_M}m")
            continue
        for upstream, lanes in neighbors:
            print(f"{tls_id} <- {upstream}: {len(lanes)} lane(s) {lanes}")
