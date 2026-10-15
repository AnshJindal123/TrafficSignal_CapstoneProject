"""
Diagnostic (not part of the training/eval pipeline): records how long
the DQN policy actually holds each phase before switching at 442107275,
frozen weights, one episode - checking whether MIN_GREEN=20 is the
binding constraint behind the bursty release pattern found in
diagnose_switching.py (avg 48s between switches, but is that a tight
distribution around 20s with occasional long holds, or something else?).
"""
import sys

import torch
import traci

import dqn_traffic_combined as m

TLS_ID = sys.argv[1] if len(sys.argv) > 1 else "442107275"

if __name__ == "__main__":
    config = m.discover_controllable_tls()
    info = config[TLS_ID]
    agent = m.SingleJunctionAgent(TLS_ID, info["green_phases"], info["lane_groups"])
    ckpt = torch.load(f"dqn_policy_seq_{TLS_ID}.pt", weights_only=True)
    agent.policy_net.load_state_dict(ckpt)
    agent.epsilon = 0.0

    traci.start(["sumo", "-c", "combined.sumocfg", "--no-step-log",
                 "--duration-log.disable", "--seed", str(m.SEED)])
    m.apply_phase_merge(TLS_ID)
    m.install_realistic_bystanders(config, TLS_ID)

    dwell_times = []
    last_switch_time = 0.0
    last_phase = None
    while traci.simulation.getTime() < 3600:
        now = traci.simulation.getTime()
        state = agent.get_state()
        action = agent.choose_action(state)
        if last_phase is not None and action != last_phase:
            dwell_times.append(now - last_switch_time)
            last_switch_time = now
        last_phase = action
        agent.apply_action(action)
    traci.close()

    dwell_times.sort()
    n = len(dwell_times)
    at_min_green = sum(1 for d in dwell_times if d <= m.MIN_GREEN + 1)
    print(f"\n{TLS_ID}: {n} dwell periods")
    print(f"  min={dwell_times[0]:.1f}s max={dwell_times[-1]:.1f}s "
          f"median={dwell_times[n//2]:.1f}s mean={sum(dwell_times)/n:.1f}s")
    print(f"  at/near MIN_GREEN({m.MIN_GREEN}s, +/-1s): {at_min_green}/{n} ({100*at_min_green/n:.0f}%)")
    print(f"  distribution (10 buckets):")
    lo, hi = dwell_times[0], dwell_times[-1]
    width = (hi - lo) / 10 if hi > lo else 1
    for i in range(10):
        b_lo, b_hi = lo + i*width, lo + (i+1)*width
        count = sum(1 for d in dwell_times if b_lo <= d < b_hi or (i == 9 and d == b_hi))
        print(f"    [{b_lo:6.1f}, {b_hi:6.1f}): {'#' * count} ({count})")

    print("\n=== diagnose_dwell.py finished ===")
