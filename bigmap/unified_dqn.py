"""
Unified-interface DQN: ONE network architecture (identical parameter shapes)
for every junction, whatever its phase count, neighbor count or state mode.
Built as an alternative to the per-junction shapes in dqn_traffic_combined.py
so the whole network - not just one layer - can be federated.

Why the old design can't be federated: each junction's input layer is sized
by its own phase/neighbor count and its output layer by its action count, so
only the hidden->hidden layer has a common shape, and its units mean
different things at different junctions.

Design (DeepSets-style, per-phase scoring):
  state  = [phase_0 feats | phase_1 feats | ... | context]     (length n*F + C)
  phase feats (F=5): queue/20, wait/300, speed/15, queue_trend/20, is_current
  context   (C=2): phase_time/MAX_GREEN, sum of neighbor-inflow features
  net: a shared encoder embeds each phase from [its feats, context]; each
       phase's Q-value is a shared head on [its embedding, mean embedding of
       all phases]. Parameters never depend on n, so any n works unchanged,
       and "long queue, long wait -> serve it" is one rule with one set of
       weights at every junction.

Information parity: a junction that today only sees queues + phase (the
"thin" state, 6 of 8 junctions - richer state was measured to HURT them, see
RICHER_STATE_TLS in dqn_traffic_combined.py) gets wait/speed/trend/phase_time
zero-filled here, so it sees no more than it does today.
UNIFIED_FEATURES=all gives every junction every feature instead.

Versions (UNIFIED_VERSION env, default v2; labels/filenames carry the tag):
  v1 (tag _u): the 5 per-phase features above only. Measured (4 seeds x 5 eval
     seeds): 7.7% worse avg wait than the canonical per-junction checkpoints,
     almost all of it at #3more (queue 82 m vs 33 m) - the phase scorer cannot
     tell WHICH phase is which, and #3more has 5 phases of very different
     geometry (3 single-lane approaches, phases 0/2 merged).
  v2 (tag _u2): adds phase identity back without breaking sharing -
     * 3 more per-phase features: lane count/4, cycle position i/4, time since
       this phase was last served/300 (clipped at 2);
     * a small per-node learned phase-identity embedding (phase_embed): LOCAL,
       never aggregated, so each node keeps its own idea of "which phase is
       which" while every other parameter stays shared.

Reuses SingleJunctionAgent entirely (heuristics in choose_action via the
_queues_from_state hook, reward, replay, Double-DQN train(), epsilon) - only
the state encoding and the network differ.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import dqn_traffic_combined as m

VERSION = os.environ.get("UNIFIED_VERSION", "v2")
assert VERSION in ("v1", "v2")
TAG = "_u" if VERSION == "v1" else "_u2"
PHASE_FEATS = 5 if VERSION == "v1" else 8
CTX_FEATS = 2
# Merged 9-tile map (bigmap/): largest controllable junction has 7 green
# phases (joinedS_11150087489_...), so the original 5 would crash there.
# Changes phase_embed's shape only - shared enc/head shapes are unaffected.
MAX_PHASES = 7
EMB_DIM = 0 if VERSION == "v1" else 4
LOCAL_KEY_PREFIXES = ("phase_embed",)   # per-node parameters, never aggregated
UNIFIED_HIDDEN = 64
UNIFIED_FEATURES = os.environ.get("UNIFIED_FEATURES", "match")   # match | all


class UnifiedDQN(nn.Module):
    def __init__(self, hidden=UNIFIED_HIDDEN):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(PHASE_FEATS + CTX_FEATS + EMB_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        if EMB_DIM:
            self.phase_embed = nn.Embedding(MAX_PHASES, EMB_DIM)
            nn.init.normal_(self.phase_embed.weight, std=0.1)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, x):
        n = (x.shape[-1] - CTX_FEATS) // PHASE_FEATS
        phases = x[..., :n * PHASE_FEATS].reshape(*x.shape[:-1], n, PHASE_FEATS)
        ctx = x[..., n * PHASE_FEATS:].unsqueeze(-2).expand(*x.shape[:-1], n, CTX_FEATS)
        parts = [phases, ctx]
        if EMB_DIM:
            emb = self.phase_embed(torch.arange(n)).expand(*x.shape[:-1], n, EMB_DIM)
            parts.append(emb)
        h = self.enc(torch.cat(parts, dim=-1))                      # (..., n, H)
        pooled = h.mean(dim=-2, keepdim=True).expand_as(h)
        return self.head(torch.cat([h, pooled], dim=-1)).squeeze(-1)  # (..., n)


class UnifiedJunctionAgent(m.SingleJunctionAgent):
    def __init__(self, *args, unified_hidden=UNIFIED_HIDDEN, **kwargs):
        super().__init__(*args, **kwargs)
        self.rich = (UNIFIED_FEATURES == "all") or (self.tls_id in m.RICHER_STATE_TLS)
        self.state_size = self.n_actions * PHASE_FEATS + CTX_FEATS
        self.policy_net = UnifiedDQN(unified_hidden)
        self.target_net = UnifiedDQN(unified_hidden)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=m.lr)

    def _queues_from_state(self, state):
        return [state[i * PHASE_FEATS] for i in range(self.n_actions)]

    def get_state(self):
        import traci
        state = []
        for i, group in enumerate(self.lane_groups):
            queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in group)
            if self.rich:
                wait = sum(traci.lane.getWaitingTime(l) for l in group)
                speeds = [traci.lane.getLastStepMeanSpeed(l) for l in group]
                speed = sum(speeds) / len(speeds) if speeds else 0.0
                trend = self.prev_queues[i] - queue
            else:
                wait = speed = trend = 0.0
            self.prev_queues[i] = queue
            is_cur = 1.0 if self.green_phases[i] == self.current_phase else 0.0
            state.extend([queue / 20.0, wait / 300.0, speed / 15.0, trend / 20.0, is_cur])
            if VERSION == "v2":
                state.extend([len(group) / 4.0, i / 4.0,
                              min(self.wait_since_served[i] / 300.0, 2.0)])
        state.append(self.phase_time / m.MAX_GREEN if self.rich else 0.0)
        state.append(float(sum(self._neighbor_outflow_features())))
        return np.array(state, dtype=np.float32)


if __name__ == "__main__":
    # Self-test (no SUMO sim): identical parameter shapes for any phase count,
    # batch/unbatched agreement, and permutation-equivariance of Q over phases.
    torch.manual_seed(0)
    net = UnifiedDQN()
    shapes = {k: tuple(v.shape) for k, v in net.state_dict().items()}
    for n in (2, 3, 5):
        x = torch.rand(7, n * PHASE_FEATS + CTX_FEATS)
        q = net(x)
        assert q.shape == (7, n), q.shape
        assert torch.allclose(net(x[0]), q[0], atol=1e-6)
        perm = torch.randperm(n)
        xp = torch.cat([x[:, :n * PHASE_FEATS].reshape(7, n, PHASE_FEATS)[:, perm].reshape(7, -1),
                        x[:, n * PHASE_FEATS:]], dim=1)
        if not EMB_DIM:
            assert torch.allclose(net(xp), q[:, perm], atol=1e-5), "not permutation-equivariant"
    print("version", VERSION, "| param shapes (same for every n):", shapes)
    print("n_params:", sum(p.numel() for p in net.parameters()))
    print("self-test ok")
