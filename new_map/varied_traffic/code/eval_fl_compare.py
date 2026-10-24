"""
Evaluates the 'independent', 'fl' (and, if present, 'fl_preagg') checkpoints produced by
federated_regional.py under identical conditions (all 8 DQNs live at once,
epsilon=0, same seed/demand) and prints a DQN-vs-Federated-DQN comparison.
No fixed-time before_*.xml baseline is needed.

Usage: python eval_fl_compare.py
Outputs: fl_eval_{tripinfo,summary,queue}_{mode}.xml
"""
import os
import xml.etree.ElementTree as ET

import torch

import dqn_traffic_combined as m
import discover_neighbor_topology as nt
import federated_regional as fr

# "fl_preagg" = FL models from just before the last aggregation (if present).
MODES = ["independent", "fl"] + (
    ["fl_preagg"] if os.path.exists(fr.ckpt_name("fl_preagg", "1213632636")) else [])
net = ET.parse("combined.net.xml").getroot()


def controlled_lanes(tls_id):
    return {f"{c.get('from')}_{c.get('fromLane')}"
            for c in net.findall("connection") if c.get("tl") == tls_id}


# SUMO only writes tripinfo for FINISHED trips by default, so "wait" etc. below
# average finished trips only - that penalises a controller that clears more
# hard-case vehicles (found on the FED_AVG branch, see eval_fair.py there).
# With FAIR_ARGS SUMO also writes the vehicles still on the road / never
# departed at the end; trip_metrics keeps the original finished-only fields
# unchanged and adds *_all fields that count every scheduled vehicle.
FAIR_ARGS = ["--tripinfo-output.write-unfinished", "true",
             "--tripinfo-output.write-undeparted", "true"]


def trip_metrics(path):
    n = w = d = loss = 0
    total = w_all = delay_all = 0
    for t in ET.parse(path).getroot().findall("tripinfo"):
        total += 1
        w_all += float(t.get("waitingTime", 0))
        delay_all += float(t.get("waitingTime", 0)) + float(t.get("departDelay", 0))
        if float(t.get("arrival", -1)) < 0:      # unfinished / undeparted (FAIR_ARGS only)
            continue
        n += 1
        w += float(t.get("waitingTime", 0))
        d += float(t.get("duration", 0))
        loss += float(t.get("timeLoss", 0))
    return {"trips": n, "wait": w / n, "travel": d / n, "loss": loss / n,
            "vehicles": total, "wait_all": w_all / total, "delay_all": delay_all / total}


def summary_metrics(path):
    last = ET.parse(path).getroot().findall("step")[-1]
    return {"teleports": int(last.get("teleports")), "running": int(last.get("running"))}


def junction_queues(path, tls_ids):
    """{tls_id: avg queue (m)} in ONE streaming pass over the queue xml.
    junction_queue() below re-parses the whole file per junction, which is
    fine for 8 junctions on the small map but not for 20+ on the merged map."""
    lane_owner = {}
    for t in tls_ids:
        for lane in controlled_lanes(t):
            lane_owner.setdefault(lane, []).append(t)
    totals = {t: 0.0 for t in tls_ids}
    steps = 0
    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag != "data":
            continue
        steps += 1
        for lane in elem.iter("lane"):
            for t in lane_owner.get(lane.get("id"), ()):
                totals[t] += float(lane.get("queueing_length_experimental", 0))
        elem.clear()
    return {t: v / steps if steps else 0.0 for t, v in totals.items()}


def junction_queue(path, lanes):
    total = steps = 0
    for ts in ET.parse(path).getroot().findall("data"):
        steps += 1
        for lane in ts.findall(".//lane"):
            if lane.get("id") in lanes:
                total += float(lane.get("queueing_length_experimental", 0))
    return total / steps if steps else 0.0


def evaluate(mode, config, topology):
    runners = fr.build_runners(config, topology, 1)
    for tls_id, r in runners.items():
        r.agent.policy_net.load_state_dict(
            torch.load(fr.ckpt_name(mode, tls_id), weights_only=True))
        r.agent.epsilon = 0.0
    fr.run_episode(runners, config, learn=False, sumo_extra=[
        "--tripinfo-output", f"fl_eval_tripinfo_{mode}{fr.SEED_SUFFIX}.xml",
        "--summary", f"fl_eval_summary_{mode}{fr.SEED_SUFFIX}.xml",
        "--queue-output", f"fl_eval_queue_{mode}{fr.SEED_SUFFIX}.xml"])


if __name__ == "__main__":
    config = m.discover_controllable_tls()
    topology = nt.discover_neighbor_inflows(config)
    for mode in MODES:
        print(f"evaluating {mode} ...", flush=True)
        evaluate(mode, config, topology)

    res = {md: {**trip_metrics(f"fl_eval_tripinfo_{md}{fr.SEED_SUFFIX}.xml"),
                **summary_metrics(f"fl_eval_summary_{md}{fr.SEED_SUFFIX}.xml")} for md in MODES}
    others = [md for md in MODES if md != "independent"]
    print(f"\n{'Network-wide metric':<22}{'Independent':>13}" + "".join(f"{md:>13}{'gain %':>9}" for md in others))
    for key, lower_better in [("wait", True), ("travel", True), ("loss", True),
                              ("teleports", True), ("trips", False)]:
        a = res["independent"][key]
        row = f"{key:<22}{a:>13.2f}"
        for md in others:
            b = res[md][key]
            gain = ((a - b) if lower_better else (b - a)) / a * 100 if a else 0.0
            row += f"{b:>13.2f}{gain:>8.2f}%"
        print(row)

    print(f"\n{'Junction avg queue (m)':<66}{'Indep':>8}" + "".join(f"{md:>11}{'gain %':>9}" for md in others))
    for tls_id in config:
        lanes = controlled_lanes(tls_id)
        a = junction_queue(f"fl_eval_queue_independent{fr.SEED_SUFFIX}.xml", lanes)
        row = f"{tls_id:<66}{a:>8.2f}"
        for md in others:
            b = junction_queue(f"fl_eval_queue_{md}{fr.SEED_SUFFIX}.xml", lanes)
            row += f"{b:>11.2f}{((a - b) / a * 100 if a else 0):>8.2f}%"
        print(row)
