"""
Evaluates federated checkpoints with the SAME methodology as the rest of
this project (frozen weights, epsilon=0, fixed SUMO seed), so results are
directly comparable to the canonical milestone:

  python eval_federated.py solo [tag]
      One fresh SUMO run per junction, that junction under its federated
      checkpoint, the other 7 on realistic fixed-time bystanders - mirrors
      eval_one.py. Writes seq_after_{tripinfo,summary,queue}_fed{tag}_{tls}.xml.

  python eval_federated.py sim [tag]
      All 8 junctions under their federated checkpoints at once on one
      shared clock - mirrors eval_all_simultaneous.py (same yellow/decision
      state machine, same per-step phase re-assert). Writes
      seq_after_{tripinfo,summary,queue}_fed{tag}_all8.xml.

  python eval_federated.py sim [tag] --canonical "#3more"
      Same as sim, but the listed junctions (comma-separated substrings of
      tls_id) run their canonical milestone model (dqn_policy_seq_*.pt)
      instead - isolates how much those junctions cost the federated model.
      Writes seq_after_{tripinfo,summary,queue}_fed{tag}mix_all8.xml.

tag picks which checkpoints load: dqn_policy_fed_{tag}{tls}.pt (the
personalized per-junction models). Pass tag=global_ to evaluate the
single shared global model at every junction instead - loads
dqn_policy_fed_{tag_without_global_}global.pt for all 8.

Compare with: python compare_federated.py [tag]
"""
import sys

import torch
import traci

import dqn_traffic_combined as m
import discover_neighbor_topology as nt
import federated_dqn as fd

DECISION_INTERVAL = 5
YELLOW_STEPS = 5


def load_weights(tls_id, tag):
    if tag.endswith("global_"):
        base = tag[:-len("global_")]
        return torch.load(f"dqn_policy_fed_{base}global.pt", weights_only=True)
    safe = tls_id.replace("/", "_")
    return torch.load(f"dqn_policy_fed_{tag}{safe}.pt", weights_only=True)


def build_agent(tls_id, info, specs, tag, canonical=False):
    if canonical:
        # The milestone's own per-junction model, built exactly as
        # eval_all_simultaneous.py builds it.
        agent = m.SingleJunctionAgent(tls_id, info["green_phases"], info["lane_groups"],
                                      neighbor_specs=specs)
        safe = tls_id.replace("/", "_")
        agent.policy_net.load_state_dict(torch.load(f"dqn_policy_seq_{safe}.pt", weights_only=True))
        agent.epsilon = 0.0
        return agent
    agent = fd.FederatedJunctionAgent(tls_id, info["green_phases"], info["lane_groups"],
                                      neighbor_specs=specs)
    agent.policy_net.load_state_dict(load_weights(tls_id, tag))
    agent.epsilon = 0.0
    return agent


def eval_solo(config, topology, tag):
    for tls_id, info in config.items():
        safe = tls_id.replace("/", "_")
        agent = build_agent(tls_id, info, topology[tls_id], tag)
        out = f"fed{tag}_{safe}"
        traci.start(["sumo", "-c", "combined.sumocfg",
                     "--tripinfo-output", f"seq_after_tripinfo_{out}.xml",
                     "--summary", f"seq_after_summary_{out}.xml",
                     "--queue-output", f"seq_after_queue_{out}.xml",
                     "--no-step-log", "--duration-log.disable", "--seed", str(m.SEED)])
        m.apply_phase_merge(tls_id)
        m.install_realistic_bystanders(config, tls_id)
        while traci.simulation.getTime() < 3600:
            state = agent.get_state()
            agent.apply_action(agent.choose_action(state))
        traci.close()
        print(f"  solo eval done: {tls_id}")


class _Runner:
    def __init__(self, agent):
        self.agent = agent
        self.pending_action = None
        self.transition_remaining = 0


def eval_sim(config, topology, tag, canonical_tls=(), out=None, extra_args=(), queue=True):
    runners = {t: _Runner(build_agent(t, info, topology[t], tag, canonical=t in canonical_tls))
               for t, info in config.items()}
    if out is None:
        out = f"seq_after_{{}}_fed{tag}mix_all8" if canonical_tls else f"seq_after_{{}}_fed{tag}_all8"
    for t in canonical_tls:
        print(f"  using canonical milestone model at {t}")
    cmd = ["sumo", "-c", "combined.sumocfg",
           "--tripinfo-output", out.format("tripinfo") + ".xml",
           "--summary", out.format("summary") + ".xml",
           "--no-step-log", "--duration-log.disable", "--seed", str(m.SEED), *extra_args]
    if queue:
        cmd += ["--queue-output", out.format("queue") + ".xml"]
    traci.start(cmd)
    for tls_id in config:
        m.apply_phase_merge(tls_id)

    while traci.simulation.getTime() < 3600:
        traci.simulationStep()
        for tls_id, r in runners.items():
            agent = r.agent
            if r.transition_remaining > 0:
                r.transition_remaining -= 1
                if r.transition_remaining == 0:
                    traci.trafficlight.setPhase(tls_id, r.pending_action)
                    agent.current_phase = r.pending_action
                    agent.phase_time = 0
                continue

            agent.phase_time += 1
            action_idx = agent.green_phases.index(agent.current_phase)
            for i in range(agent.n_actions):
                agent.wait_since_served[i] = 0 if i == action_idx else agent.wait_since_served[i] + 1

            if agent.phase_time % DECISION_INTERVAL != 0:
                continue

            action = agent.choose_action(agent.get_state())
            if action != agent.current_phase:
                traci.trafficlight.setPhase(tls_id, traci.trafficlight.getPhase(tls_id) + 1)
                r.pending_action = action
                r.transition_remaining = YELLOW_STEPS
            else:
                # Re-assert every decision so SUMO's own program timer can't
                # drift the live phase away from agent.current_phase (same
                # fix as eval_all_simultaneous.py).
                traci.trafficlight.setPhase(tls_id, agent.current_phase)
    traci.close()
    print("  simultaneous eval done")


if __name__ == "__main__":
    args = sys.argv[1:]
    canonical_keys = []
    if "--canonical" in args:
        i = args.index("--canonical")
        canonical_keys = args[i + 1].split(",")
        args = args[:i] + args[i + 2:]
    mode = args[0]
    tag = args[1] if len(args) > 1 else ""
    config = m.discover_controllable_tls()
    topology = nt.discover_neighbor_inflows(config)
    canonical_tls = tuple(t for t in config if any(k in t for k in canonical_keys))
    if canonical_keys and len(canonical_tls) != len(canonical_keys):
        sys.exit(f"--canonical {canonical_keys} matched {canonical_tls}, expected one junction per key")
    if canonical_tls and mode != "sim":
        sys.exit("--canonical is only supported in sim mode")
    if mode == "solo":
        eval_solo(config, topology, tag)
    elif mode == "sim":
        eval_sim(config, topology, tag, canonical_tls)
    else:
        sys.exit(f"unknown mode {mode!r}, expected 'solo' or 'sim'")
    print(f"=== eval_federated.py {mode} finished ===")
