"""
FedAvg training over all 8 controllable junctions, using the padded
shared architecture from federated_dqn.py.

Each round:
  1. Server broadcasts the current global policy weights to all 8 clients.
  2. Every client trains LOCALLY for --local-episodes episodes on its own
     junction (own reward, own replay memory, own Adam state, realistic
     fixed-time bystanders - identical setup to train_all.py). Clients run
     in parallel worker processes, one SUMO instance each on its own port.
     Only WEIGHTS ever leave a client - replay memory/raw traffic data stay
     with that client across rounds (carried in client_states, which the
     server only ferries back to the same client, never mixes).
  3. Server averages the 8 updated weight sets (plain FedAvg, equal
     weights - every client does the same number of local episodes) into
     the new global model.

After --rounds rounds: each client runs --personalize-episodes more local
episodes starting from the final global weights, WITHOUT further
averaging (standard FL personalization - one averaged policy usually
underfits any single junction's specific traffic pattern).

Outputs (all under a distinct 'fed_' prefix, never touching the canonical
dqn_policy_seq_*.pt milestone checkpoints):
  dqn_policy_fed_{tag}global.pt          final averaged global model
  dqn_policy_fed_{tag}{safe_tls}.pt      per-junction personalized model
  fed_{tag}checkpoint.pt                 resume state, rewritten after
                                         every completed round - rerunning
                                         the same command after a crash or
                                         shutdown resumes from the last
                                         completed round automatically.

Usage:
  python train_federated.py                       # full run (defaults below)
  python train_federated.py --rounds 1 --local-episodes 1 \
      --personalize-episodes 1 --tag smoke_       # quick smoke test
  python train_federated.py --fedper --tag per_   # FedPer: share hidden layers only,
                                                  # each junction keeps its own output layer
  python train_federated.py --grouped --tag grp_  # hidden layers shared by all; output
                                                  # layer shared within same-phase-count groups
"""
import argparse
import os
import random
import sys
import time
from collections import deque
from multiprocessing import Pool

import numpy as np
import torch

import dqn_traffic_combined as m
import discover_neighbor_topology as nt
import federated_dqn as fd

BASE_PORT = 8900


def _client_work(args):
    """Runs in a worker process. Rebuilds this client's agent from the
    global weights + its own persisted client_state, trains n_episodes
    locally, returns updated weights + client_state."""
    (idx, tls_id, info, specs, global_weights, client_state,
     n_episodes, eps_decay, seed, bystander_config) = args
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = fd.FederatedJunctionAgent(tls_id, info["green_phases"], info["lane_groups"],
                                      neighbor_specs=specs, epsilon_decay=eps_decay)
    agent.policy_net.load_state_dict(global_weights)
    agent.target_net.load_state_dict(global_weights)
    if client_state is not None:
        agent.memory = deque(client_state["memory"], maxlen=m.memory_size)
        agent.optimizer.load_state_dict(client_state["optimizer"])
        agent.epsilon = client_state["epsilon"]

    rewards = []
    for _ in range(n_episodes):
        rewards.append(fd.local_train_episode(agent, bystander_config, port=BASE_PORT + idx))

    new_client_state = {
        "memory": list(agent.memory),
        "optimizer": agent.optimizer.state_dict(),
        "epsilon": agent.epsilon,
    }
    return tls_id, agent.policy_net.state_dict(), new_client_state, rewards


def _short(tls_id):
    for key in ("#4more", "#3more", "GS_cluster", "cluster_11594569878"):
        if key in tls_id:
            return key
    return tls_id


# m.DQN's output layer (fc.0 / fc.2 are the two hidden layers). Under
# --fedper these stay local to each junction: output slot i means a
# different physical phase at every junction, so averaging it across
# clients mixes unrelated decisions (see #3more under plain FedAvg).
LOCAL_KEYS = ("fc.4.weight", "fc.4.bias")


def _average(state_dicts):
    averaged = {}
    for key in state_dicts[0]:
        averaged[key] = torch.stack([sd[key].float() for sd in state_dicts], dim=0).mean(dim=0)
    return averaged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-episodes", type=int, default=3)
    parser.add_argument("--personalize-episodes", type=int, default=5)
    parser.add_argument("--tag", default="")
    parser.add_argument("--fedper", action="store_true",
                        help="average only the hidden layers; each junction keeps its own output "
                             "layer for the whole run (no global model, no personalization pass)")
    parser.add_argument("--grouped", action="store_true",
                        help="average hidden layers across all junctions, but average the output "
                             "layer only among junctions with the same number of phases")
    args = parser.parse_args()
    if args.fedper and args.grouped:
        parser.error("--fedper and --grouped are mutually exclusive")
    local_heads = args.fedper or args.grouped

    config = m.discover_controllable_tls()
    topology = nt.discover_neighbor_inflows(config)
    tls_ids = list(config.keys())
    # --grouped: output slot i means "phase i of an N-phase junction", which is
    # only comparable between junctions with the same N. On this map that
    # puts the six 2-phase junctions in one group, #4more (3) and #3more (5)
    # alone - so those two effectively keep a FedPer-style local head.
    groups = {}
    for t in tls_ids:
        groups.setdefault(len(config[t]["green_phases"]), []).append(t)

    total_local = args.rounds * args.local_episodes
    eps_decay = m.scaled_epsilon_decay(total_local, explore_fraction=0.8)

    ckpt_path = f"fed_{args.tag}checkpoint.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        start_round = ckpt["round_completed"] + 1
        global_weights = ckpt["global"]
        client_states = ckpt["client_states"]
        heads = ckpt.get("heads")
        print(f"Resuming from {ckpt_path}: {ckpt['round_completed'] + 1}/{args.rounds} rounds already done")
    else:
        random.seed(m.SEED)
        np.random.seed(m.SEED)
        torch.manual_seed(m.SEED)
        init_net = m.DQN(fd.MAX_ACTIONS, fd.STATE_SIZE, fd.HIDDEN_SIZE)
        start_round = 0
        global_weights = init_net.state_dict()
        client_states = {t: None for t in tls_ids}
        heads = ({t: {k: global_weights[k].clone() for k in LOCAL_KEYS} for t in tls_ids}
                 if local_heads else None)

    def client_weights(t):
        # FedPer / grouped: shared (averaged) hidden layers + this junction's own
        # (FedPer) or its group's (grouped) output layer.
        return {**global_weights, **heads[t]} if local_heads else global_weights

    if args.fedper:
        mode = "FedPer (hidden layers shared, output layer local)"
    elif args.grouped:
        mode = ("Grouped heads (hidden layers shared, output layer shared within phase-count groups: "
                + ", ".join(f"{n}-phase x{len(ts)}" for n, ts in sorted(groups.items())) + ")")
    else:
        mode = "FedAvg"
    print(f"\n### {mode}: rounds={args.rounds} local_episodes={args.local_episodes} "
          f"personalize_episodes={0 if local_heads else args.personalize_episodes} clients={len(tls_ids)} "
          f"state_size={fd.STATE_SIZE} max_actions={fd.MAX_ACTIONS} "
          f"eps_decay={eps_decay:.6f} tag='{args.tag}' ###\n")
    sys.stdout.flush()

    with Pool(processes=len(tls_ids)) as pool:
        for rnd in range(start_round, args.rounds):
            t0 = time.time()
            jobs = [(i, t, config[t], topology[t], client_weights(t), client_states[t],
                     args.local_episodes, eps_decay, m.SEED + 1000 * (rnd + 1) + i, config)
                    for i, t in enumerate(tls_ids)]
            results = pool.map(_client_work, jobs)

            new_weights = []
            for tls_id, weights, cstate, rewards in results:
                client_states[tls_id] = cstate
                new_weights.append(weights)
                if local_heads:
                    heads[tls_id] = {k: weights[k] for k in LOCAL_KEYS}
            if args.grouped:
                for members in groups.values():
                    group_head = _average([heads[t] for t in members])
                    for t in members:
                        heads[t] = group_head
            # With local heads, the averaged output layer in global_weights is
            # never used - client_weights() overrides it per junction.
            global_weights = _average(new_weights)

            torch.save({"round_completed": rnd, "global": global_weights,
                        "client_states": client_states, "heads": heads}, ckpt_path)

            summary = "  ".join(f"{_short(t)}:{r[-1]:.0f}" for t, _, _, r in results)
            eps = results[0][2]["epsilon"]
            print(f"Round {rnd + 1}/{args.rounds} done in {(time.time() - t0) / 60:.1f} min | "
                  f"epsilon={eps:.3f} | last-episode reward per client: {summary}")
            sys.stdout.flush()

        if local_heads:
            for t in tls_ids:
                torch.save(client_weights(t), f"dqn_policy_fed_{args.tag}{t.replace('/', '_')}.pt")
            print(f"\nSaved per-junction models -> dqn_policy_fed_{args.tag}<tls>.pt")
        else:
            torch.save(global_weights, f"dqn_policy_fed_{args.tag}global.pt")
            print(f"\nSaved global model -> dqn_policy_fed_{args.tag}global.pt")

        if not local_heads and args.personalize_episodes > 0:
            t0 = time.time()
            jobs = [(i, t, config[t], topology[t], global_weights, client_states[t],
                     args.personalize_episodes, eps_decay, m.SEED + 999999 + i, config)
                    for i, t in enumerate(tls_ids)]
            results = pool.map(_client_work, jobs)
            for tls_id, weights, _, _ in results:
                safe = tls_id.replace("/", "_")
                torch.save(weights, f"dqn_policy_fed_{args.tag}{safe}.pt")
            print(f"Personalization ({args.personalize_episodes} episodes/client) done in "
                  f"{(time.time() - t0) / 60:.1f} min -> dqn_policy_fed_{args.tag}<tls>.pt")

    print("\n=== train_federated.py finished ===")


if __name__ == "__main__":
    main()
