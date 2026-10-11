"""
Read-only diagnostic for ONE junction (default 343322679): logs every
decision it makes in a full all-8-live evaluation episode (epsilon=0, same
shared-clock driver as eval_multiseed / headroom_check), labelling WHICH
mechanism made it (starvation override / min-green hold / max-green forced /
imbalance switch / DQN) and what the DQN would have chosen, so canonical and
unified models can be compared decision by decision.

Usage:
  python diag_343.py canonical <eval_seed>
  python diag_343.py unified_solo <train_seed> <eval_seed>     # independent_u2_solo ckpts
  python diag_343.py unified_solo_fl <train_seed> <eval_seed>  # fl_u2_solo ckpts
Writes diag/<label>_e<eval_seed>.json. Nothing is trained or modified.
"""
import json
import os
import sys

KIND = sys.argv[1]
if KIND == "canonical":
    TRAIN_SEED, EVAL_SEED = 0, int(sys.argv[2])
else:
    TRAIN_SEED, EVAL_SEED = int(sys.argv[2]), int(sys.argv[3])
    os.environ["FL_DESIGN"] = "unified"
    os.environ["UNIFIED_VERSION"] = "v2"
    os.environ["FL_REGIME"] = "solo"
    os.environ["FL_SEED"] = str(TRAIN_SEED)
os.environ["FL_EVAL_SEED"] = str(EVAL_SEED)

import numpy as np
import torch
import traci

import dqn_traffic_combined as m
import discover_neighbor_topology as nt
import federated_regional as fr

TARGET = os.environ.get("DIAG_TLS", "343322679")
LABEL = {"canonical": "canonical", "unified_solo": f"solo_ind_t{TRAIN_SEED}",
         "unified_solo_fl": f"solo_fl_t{TRAIN_SEED}"}[KIND]


def branch_of(agent, state):
    """Mirror of SingleJunctionAgent.choose_action's control flow, read-only,
    returning which mechanism decides. (epsilon = 0 so no random branch.)"""
    queues = agent._queues_from_state(state)
    cur = agent.green_phases.index(agent.current_phase)
    bound = agent.max_green * (agent.n_actions - 1)
    starved = [i for i in range(agent.n_actions)
               if i != cur and agent.wait_since_served[i] > bound and queues[i] > 0]
    if starved:
        return "starved"
    best = int(np.argmax(queues))
    imb = (best != cur and queues[best] > queues[cur] * agent.imbalance_ratio and
           (queues[best] - queues[cur]) >= agent.min_imbalance)
    if agent.phase_time < agent.min_green and not imb:
        return "min_green_hold"
    if agent.phase_time > agent.max_green:
        return "max_green_forced"
    if imb:
        return "imbalance"
    return "dqn"


def main():
    config = m.discover_controllable_tls()
    topo = nt.discover_neighbor_inflows(config)
    design = "canonical" if KIND == "canonical" else "unified"
    runners = fr.build_runners(config, topo, 1, design=design)
    mode = "canonical" if KIND == "canonical" else ("independent" if KIND == "unified_solo" else "fl")
    for tls_id, r in runners.items():
        r.agent.policy_net.load_state_dict(torch.load(fr.ckpt_name(mode, tls_id), weights_only=True))
        r.agent.epsilon = 0.0

    tgt = runners[TARGET].agent
    lanes = traci.trafficlight  # placeholder to keep import used
    log, timeline = [], []

    traci.start(["sumo", "-c", "combined.sumocfg", "--no-step-log", "--duration-log.disable",
                 "--seed", str(EVAL_SEED)])
    for tls_id in config:
        m.apply_phase_merge(tls_id)
    fr.reset_episode_state(runners, config)
    ctrl_lanes = traci.trafficlight.getControlledLanes(TARGET)

    while traci.simulation.getTime() < 3600:
        traci.simulationStep()
        t = traci.simulation.getTime()
        if t % 60 == 0:
            timeline.append({"t": t,
                             "queue": sum(traci.lane.getLastStepHaltingNumber(l) for l in set(ctrl_lanes)),
                             "wait": sum(traci.lane.getWaitingTime(l) for l in set(ctrl_lanes)),
                             "veh": sum(traci.lane.getLastStepVehicleNumber(l) for l in set(ctrl_lanes))})
        for tls_id, r in runners.items():
            a = r.agent
            if r.transition_remaining > 0:
                r.transition_remaining -= 1
                if r.transition_remaining == 0:
                    traci.trafficlight.setPhase(tls_id, r.pending_action)
                    a.current_phase = r.pending_action
                    a.phase_time = 0
                continue
            a.phase_time += 1
            idx = a.green_phases.index(a.current_phase)
            for i in range(a.n_actions):
                a.wait_since_served[i] = 0 if i == idx else a.wait_since_served[i] + 1
            if a.phase_time % fr.DECISION_INTERVAL != 0:
                continue
            state = a.get_state()
            if tls_id == TARGET:
                branch = branch_of(a, state)
                with torch.no_grad():
                    q = a.policy_net(torch.tensor(state, dtype=torch.float32)).tolist()
                dqn_pick = int(np.argmax(q))
            action = a.choose_action(state)
            if tls_id == TARGET:
                log.append({"t": t, "cur": a.green_phases.index(a.current_phase), "phase_time": a.phase_time,
                            "queues": [float(x) * 20 for x in a._queues_from_state(state)],
                            "branch": branch, "dqn_pick": dqn_pick, "q": q,
                            "chosen": a.green_phases.index(action)})
            if action != a.current_phase:
                traci.trafficlight.setPhase(tls_id, traci.trafficlight.getPhase(tls_id) + 1)
                r.pending_action, r.transition_remaining = action, fr.YELLOW_STEPS
            else:
                traci.trafficlight.setPhase(tls_id, a.current_phase)
    traci.close()
    os.makedirs("diag", exist_ok=True)
    json.dump({"label": LABEL, "eval_seed": EVAL_SEED, "log": log, "timeline": timeline},
              open(f"diag/{LABEL}_e{EVAL_SEED}.json", "w"))
    print("done", LABEL, EVAL_SEED, len(log), "decisions")


if __name__ == "__main__":
    main()
