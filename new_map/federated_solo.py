"""
Regional federated learning, "solo" training regime: each intersection node
trains ALONE against the realistic fixed-time program on its 7 neighbors
(install_realistic_bystanders) in its OWN simulation - the way the canonical
dqn_policy_seq_*.pt checkpoints were produced (train_one_junction) - instead
of all 8 learning together in one shared simulation (federated_regional.py,
"joint" regime). Same unified network, rounds, blend, aggregation and
evaluation otherwise, so independent-vs-fl AND solo-vs-joint are comparable.

Architecture (matches README_REGIONAL_FL.md): every node is its own OS process
holding its agent, replay buffer, optimizer and epsilon. The coordinator
process talks to nodes over pipes and only ever receives/sends MODEL PARAMETERS
(the shared state_dict entries) - never states, reward history or traffic data.

Per round: nodes train `episodes_per_round` local episodes IN PARALLEL ->
send shared params -> RegionalAggregator (federated_regional) -> blended back.
After `rounds` rounds one extra local-only round runs (no aggregation).
independent mode = identical, minus the aggregation step.

Usage (env: FL_DESIGN=unified UNIFIED_VERSION=v2 FL_REGIME=solo FL_SEED=<n>):
    python federated_solo.py <fl|independent> [rounds] [episodes_per_round]
Writes dqn_policy_{mode}{tag}_{tls}.pt (+ fl_preagg snapshot, regional model).
"""
import multiprocessing as mp
import random
import sys
import time

import numpy as np
import torch
import traci

import dqn_traffic_combined as m
import discover_neighbor_topology as nt
import federated_regional as fr


def run_local_episode(agent, tls_id, config):
    """One 3600 s episode of local DQN training - the body of
    dqn_traffic_combined.train_one_junction's episode loop, unchanged."""
    agent.current_phase = agent.green_phases[0]
    agent.phase_time = 0
    agent.wait_since_served = [0] * agent.n_actions
    agent.prev_queue = 0
    agent.prev_queues = [0] * agent.n_actions
    # 8 node processes start SUMO at nearly the same moment; a start-up race
    # ("Could not connect") killed a whole seed once, so retry instead of dying.
    for attempt in range(6):
        try:
            traci.start(["sumo", "-c", "combined.sumocfg", "--no-step-log",
                         "--duration-log.disable", "--seed", str(fr.TRAIN_SEED)])
            break
        except traci.exceptions.FatalTraCIError:
            if attempt == 5:
                raise
            try:
                traci.close()
            except Exception:
                pass
            time.sleep(5 + 5 * random.random())
    m.apply_phase_merge(tls_id)
    m.install_realistic_bystanders(config, tls_id)
    steps = 0
    r_sum = r_abs = 0.0
    while traci.simulation.getTime() < 3600:
        state = agent.get_state()
        action = agent.choose_action(state)
        agent.apply_action(action)
        next_state = agent.get_state()
        reward = agent.get_reward()
        agent.memory.append((state, action, reward, next_state))
        agent.train()
        agent.maybe_update_target(steps)
        agent.decay_epsilon()
        steps += 1
        r_sum += reward
        r_abs += abs(reward)
    traci.close()
    return steps, r_sum, r_abs


def node_main(idx, tls_id, info, specs, config, conn, total_episodes):
    """Intersection node: owns its agent; talks to the coordinator only via
    parameter messages."""
    seed = fr.TRAIN_SEED * 1000 + idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = fr.build_agent(tls_id, info, specs, m.scaled_epsilon_decay(total_episodes))
    while True:
        msg = conn.recv()
        cmd = msg[0]
        if cmd == "get":
            conn.send(fr.get_shared_params(agent))
        elif cmd == "set":
            fr.set_shared_params(agent, msg[1], msg[2])
            conn.send("ok")
        elif cmd == "train":
            steps = 0
            r_sum = r_abs = 0.0
            for _ in range(msg[1]):
                s, a, b = run_local_episode(agent, tls_id, config)
                steps += s
                r_sum += a
                r_abs += b
            conn.send({"params": fr.get_shared_params(agent), "steps": steps,
                       "avg_reward": r_sum / max(1, steps), "avg_abs_reward": r_abs / max(1, steps),
                       "epsilon": agent.epsilon})
        elif cmd == "save":
            torch.save(agent.policy_net.state_dict(), fr.ckpt_name(msg[1], tls_id))
            conn.send("ok")
        elif cmd == "quit":
            return


def main(mode, rounds, episodes_per_round):
    assert mode in ("fl", "independent") and fr.REGIME == "solo"
    config = m.discover_controllable_tls()
    topology = nt.discover_neighbor_inflows(config)
    total_eps = (rounds + 1) * episodes_per_round
    ctx = mp.get_context("spawn")
    nodes = {}
    for idx, (tls_id, info) in enumerate(config.items()):
        parent, child = ctx.Pipe()
        p = ctx.Process(target=node_main, args=(idx, tls_id, info, topology[tls_id], config,
                                                child, total_eps), daemon=True)
        p.start()
        nodes[tls_id] = (p, parent)

    def ask_all(msg_for):
        for t, (_, c) in nodes.items():
            c.send(msg_for(t))
        return {t: c.recv() for t, (_, c) in nodes.items()}

    aggregator = fr.RegionalAggregator(fr.REGION_ID)
    if fr.DESIGN == "unified":   # same common initial model in both modes
        first = next(iter(nodes))
        nodes[first][1].send(("get",))
        init = nodes[first][1].recv()
        aggregator = fr.RegionalAggregator(fr.REGION_ID, tuple(init.keys()))
        ask_all(lambda t: ("set", init, 1.0))
    print(f"### solo mode={mode} region={fr.REGION_ID} nodes={len(nodes)} rounds={rounds} "
          f"+1 final local round, episodes/round={episodes_per_round} blend={fr.BLEND_ALPHA} "
          f"seed={fr.TRAIN_SEED} design={fr.DESIGN} weighting={fr.WEIGHTING} ###", flush=True)

    for rnd in range(rounds + 1):
        final_local = (rnd == rounds)
        res = ask_all(lambda t: ("train", episodes_per_round))
        note = "final local-only round, no aggregation" if final_local else "no aggregation"
        if mode == "fl" and not final_local:
            if rnd == rounds - 1:
                ask_all(lambda t: ("save", "fl_preagg"))
            updates = {}
            for t, r in res.items():
                scale = 1.0 if fr.WEIGHTING == "steps" else min(
                    max(r["avg_abs_reward"], fr.REWARD_SCALE_CLIP[0]), fr.REWARD_SCALE_CLIP[1])
                updates[t] = (r["params"], r["steps"] / scale)
            regional = aggregator.aggregate(updates)
            ask_all(lambda t: ("set", regional, fr.BLEND_ALPHA))
            note = "aggregated"
        print(f"== round {rnd} complete ({note}) | eps "
              f"{np.mean([r['epsilon'] for r in res.values()]):.3f} | avg reward "
              + " ".join(f"{t.split('_')[0][:6]}:{r['avg_reward']:.3f}" for t, r in res.items()),
              flush=True)

    ask_all(lambda t: ("save", mode))
    if mode == "fl":
        torch.save(aggregator.get_regional_model(),
                   f"regional_model_{fr.REGION_ID}{fr.SEED_SUFFIX}.pt")
    for _, c in nodes.values():
        c.send(("quit",))
    for p, _ in nodes.values():
        p.join(timeout=30)
    print(f"=== federated_solo.py finished mode={mode} ===", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: FL_REGIME=solo python federated_solo.py <fl|independent> [rounds] [episodes_per_round]")
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else fr.ROUNDS,
         int(sys.argv[3]) if len(sys.argv) > 3 else fr.EPISODES_PER_ROUND)
