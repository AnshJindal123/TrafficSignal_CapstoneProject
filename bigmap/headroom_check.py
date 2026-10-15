"""
Phase 0 headroom check: how many of each junction's decisions does the DQN
actually make, versus the hand-written rules in choose_action (starvation
fairness override, min_green hold, max_green forced switch, significant-
imbalance switch)?

choose_action only calls policy_net on its LAST branch (epsilon=0 here), so a
wrapper around policy_net that counts calls = exact count of DQN-decided
decisions. For those, also record whether the DQN's pick differs from (a)
"stay in the current phase" and (b) "go to the phase with the longest queue" -
if the DQN only ever agrees with those trivial rules, there is nothing for
federation to improve at that junction.

Same shared-clock driver as eval_all_simultaneous.py (all 8 live, epsilon=0).
Usage: python headroom_check.py [ckpt_mode]   # default: canonical dqn_policy_seq_*.pt
       python headroom_check.py independent   # dqn_policy_independent_*.pt (hidden 64)
Writes headroom_<mode>.json
"""
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import traci

import dqn_traffic_combined as m
import discover_neighbor_topology as nt

MODE = sys.argv[1] if len(sys.argv) > 1 else "canonical"
DECISION_INTERVAL = 5
YELLOW_STEPS = 5


class Recorder(nn.Module):
    def __init__(self, net, agent):
        super().__init__()
        self.net, self.agent = net, agent
        self.calls = []

    def forward(self, x):
        out = self.net(x)
        a = self.agent
        cur = a.green_phases.index(a.current_phase)
        queues = x[:a.n_actions] if not a.use_richer_state else x[[i * 4 for i in range(a.n_actions)]]
        self.calls.append((int(torch.argmax(out)), cur, int(torch.argmax(queues)),
                           bool(float(queues.max()) > 0)))
        return out


def main():
    config = m.discover_controllable_tls()
    topo = nt.discover_neighbor_inflows(config)
    runners = {}
    for tls_id, info in config.items():
        safe = tls_id.replace("/", "_")
        if MODE == "canonical":
            agent = m.SingleJunctionAgent(tls_id, info["green_phases"], info["lane_groups"],
                                          neighbor_specs=topo[tls_id])
            ckpt = f"dqn_policy_seq_{safe}.pt"
        else:
            agent = m.SingleJunctionAgent(tls_id, info["green_phases"], info["lane_groups"],
                                          neighbor_specs=topo[tls_id], hidden_size=64)
            ckpt = f"dqn_policy_{MODE}_{safe}.pt"
        agent.policy_net.load_state_dict(torch.load(ckpt, weights_only=True))
        agent.epsilon = 0.0
        rec = Recorder(agent.policy_net, agent)
        agent.policy_net = rec
        runners[tls_id] = {"agent": agent, "rec": rec, "pending": None, "trans": 0, "decisions": 0}

    traci.start(["sumo", "-c", "combined.sumocfg", "--no-step-log", "--duration-log.disable",
                 "--seed", str(m.SEED)])
    for tls_id in config:
        m.apply_phase_merge(tls_id)

    while traci.simulation.getTime() < 3600:
        traci.simulationStep()
        for tls_id, r in runners.items():
            a = r["agent"]
            if r["trans"] > 0:
                r["trans"] -= 1
                if r["trans"] == 0:
                    traci.trafficlight.setPhase(tls_id, r["pending"])
                    a.current_phase = r["pending"]
                    a.phase_time = 0
                continue
            a.phase_time += 1
            idx = a.green_phases.index(a.current_phase)
            for i in range(a.n_actions):
                a.wait_since_served[i] = 0 if i == idx else a.wait_since_served[i] + 1
            if a.phase_time % DECISION_INTERVAL != 0:
                continue
            r["decisions"] += 1
            action = a.choose_action(a.get_state())
            if action != a.current_phase:
                traci.trafficlight.setPhase(tls_id, traci.trafficlight.getPhase(tls_id) + 1)
                r["pending"], r["trans"] = action, YELLOW_STEPS
            else:
                traci.trafficlight.setPhase(tls_id, a.current_phase)
    traci.close()

    out = {}
    print(f"{'junction':<24}{'actions':>8}{'decisions':>10}{'DQN-decided':>12}{'share':>8}"
          f"{'DQN!=stay':>10}{'DQN!=maxQ':>10}{'DQN switch,q>0':>15}")
    for tls_id, r in runners.items():
        calls, n = r["rec"].calls, r["decisions"]
        k = len(calls)
        diff_stay = sum(1 for d, c, q, _ in calls if d != c)
        diff_maxq = sum(1 for d, c, q, anyq in calls if anyq and d != q)
        sw_busy = sum(1 for d, c, q, anyq in calls if d != c and anyq)
        out[tls_id] = {"actions": r["agent"].n_actions, "decisions": n, "dqn_decided": k,
                       "dqn_differs_from_stay": diff_stay, "dqn_differs_from_maxq": diff_maxq,
                       "dqn_switch_with_queue": sw_busy}
        print(f"{tls_id[:23]:<24}{r['agent'].n_actions:>8}{n:>10}{k:>12}{k / max(1, n):>8.1%}"
              f"{diff_stay / max(1, k):>10.1%}{diff_maxq / max(1, k):>10.1%}{sw_busy / max(1, k):>15.1%}")
    json.dump(out, open(f"headroom_{MODE}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
