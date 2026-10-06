"""
Fair re-evaluation: every vehicle scheduled in the hour is counted, not
just the ones that finished.

The project's headline wait/travel/loss % (compare_federated.py,
chart_all8dqn.py, pick_best_general.py) average ONLY vehicles that finished
their trip - SUMO's tripinfo omits the rest by default. Runs that clear more
traffic finish more hard-case vehicles, which raises their average, so the
old metric penalizes throughput (and flatters the baseline, which leaves out
the most stuck vehicles). Here SUMO also writes tripinfo for vehicles still
on the road at 3600s (write-unfinished) and for vehicles that never got onto
the road (write-undeparted).

Runs (all 8 junctions simultaneous, same seed/loop as eval_all_simultaneous.py):
  baseline   realistic 55-65s fixed-time program on all 8
  milestone  canonical per-junction models (dqn_policy_seq_*.pt)
  fedavg     FL FedAvg global model at all 8 (dqn_policy_fed_global.pt)
  mix        FedAvg global at 7 + canonical milestone model at #3more

  python eval_fair.py <run>      one run -> fair_<run>_{tripinfo,summary}.xml
  python eval_fair.py compare    table -> fair_comparison.txt

Built-in check: including unfinished vehicles only adds rows, it doesn't
change the simulation - so `compare` also reports finished-only numbers,
which must match the old results (milestone: 5373 trips, 183.5s avg wait).
"""
import sys
import xml.etree.ElementTree as ET

import traci

import dqn_traffic_combined as m
import discover_neighbor_topology as nt
import eval_federated as ef

RUNS = ["baseline", "milestone", "fedavg", "mix"]
LABELS = {
    "baseline": "Baseline (fixed-time)",
    "milestone": "Milestone (per-junction DQN)",
    "fedavg": "FL: FedAvg global",
    "mix": "FedAvg + milestone #3more",
}
FAIR_ARGS = ("--tripinfo-output.write-unfinished", "true",
             "--tripinfo-output.write-undeparted", "true")


def run(name):
    config = m.discover_controllable_tls()
    topology = nt.discover_neighbor_inflows(config)
    out = f"fair_{name}_{{}}"
    if name == "baseline":
        traci.start(["sumo", "-c", "combined.sumocfg",
                     "--tripinfo-output", out.format("tripinfo") + ".xml",
                     "--summary", out.format("summary") + ".xml",
                     "--no-step-log", "--duration-log.disable", "--seed", str(m.SEED), *FAIR_ARGS])
        for tls_id, info in config.items():
            m.install_realistic_program(tls_id, info["green_phases"], info["lane_groups"])
        while traci.simulation.getTime() < 3600:
            traci.simulationStep()
        traci.close()
    else:
        canonical = {
            "milestone": tuple(config),
            "fedavg": (),
            "mix": tuple(t for t in config if "#3more" in t),
        }[name]
        ef.eval_sim(config, topology, "global_", canonical_tls=canonical,
                    out=out, extra_args=FAIR_ARGS, queue=False)
    print(f"=== eval_fair.py {name} finished ===")


def stats(path):
    trips = ET.parse(path).getroot().findall("tripinfo")
    n = len(trips)
    finished = [t for t in trips if float(t.get("arrival", -1)) >= 0]
    undeparted = [t for t in trips if float(t.get("depart", -1)) < 0]

    def avg(items, key):
        return sum(float(t.get(key, 0)) for t in items) / len(items) if items else 0.0

    return {
        "total": n,
        "finished": len(finished),
        "on_road": n - len(finished) - len(undeparted),
        "undeparted": len(undeparted),
        "wait_all": avg(trips, "waitingTime"),
        "delay_all": sum(float(t.get("waitingTime", 0)) + float(t.get("departDelay", 0))
                         for t in trips) / n,
        "loss_all": avg([t for t in trips if float(t.get("depart", -1)) >= 0], "timeLoss"),
        "wait_finished": avg(finished, "waitingTime"),
    }


def compare():
    s = {r: stats(f"fair_{r}_tripinfo.xml") for r in RUNS}
    b = s["baseline"]

    def pct(before, after):
        return (before - after) / before * 100 if before else 0.0

    lines = ["=== FAIR COMPARISON - every vehicle scheduled in the hour counted ===",
             "  wait  = avg time stopped in traffic, all vehicles (finished + still on road + never entered)",
             "  delay = wait + time spent waiting to enter the road (departDelay) - total time lost queueing",
             "  loss  = avg timeLoss vs free-flow, all vehicles that entered",
             "",
             f"{'Run':<31}{'vehicles':>9}{'finished':>10}{'on road':>9}{'never in':>10}"
             f"{'wait s':>9}{'%':>8}{'delay s':>10}{'%':>8}{'loss s':>9}{'%':>8}"]
    for r in RUNS:
        x = s[r]
        lines.append(f"{LABELS[r]:<31}{x['total']:>9}{x['finished']:>10}{x['on_road']:>9}{x['undeparted']:>10}"
                     f"{x['wait_all']:>9.1f}{pct(b['wait_all'], x['wait_all']):>7.1f}%"
                     f"{x['delay_all']:>10.1f}{pct(b['delay_all'], x['delay_all']):>7.1f}%"
                     f"{x['loss_all']:>9.1f}{pct(b['loss_all'], x['loss_all']):>7.1f}%")
    lines += ["", "=== SANITY CHECK - finished trips only (must equal the old evaluation) ==="]
    for r in RUNS:
        x = s[r]
        lines.append(f"{LABELS[r]:<31}{x['finished']:>6} finished, avg wait {x['wait_finished']:.1f}s")
    out = "\n".join(lines)
    print(out)
    with open("fair_comparison.txt", "w") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare()
    elif sys.argv[1] in RUNS:
        run(sys.argv[1])
    else:
        sys.exit(f"usage: python eval_fair.py {{{'|'.join(RUNS)}|compare}}")
