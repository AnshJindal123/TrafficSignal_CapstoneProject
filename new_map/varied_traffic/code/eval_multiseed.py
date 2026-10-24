"""
Evaluation-noise study (Step 1): re-evaluates ONE training seed's existing
checkpoints (independent / fl / fl_preagg, greedy epsilon=0) on ONE SUMO
evaluation seed, so the same models can be tested on many traffic
realisations. Nothing is retrained.

Usage: FL_SEED=<train seed> FL_EVAL_SEED=<eval seed> python eval_multiseed.py
Writes ms_results/t<train>_e<eval>_<mode>.json (resumable: existing files are
skipped); the large SUMO xml outputs go to ms_eval_tmp/ and are deleted after
parsing. Summarise with: python summarize_multiseed.py
"""
import json
import os

import torch

import dqn_traffic_combined as m
import discover_neighbor_topology as nt
import federated_regional as fr
import eval_fl_compare as ef

OUT_TMP = "ms_eval_tmp"
OUT_RES = "ms_results"


def main():
    os.makedirs(OUT_TMP, exist_ok=True)
    os.makedirs(OUT_RES, exist_ok=True)
    config = m.discover_controllable_tls()
    topology = nt.discover_neighbor_inflows(config)
    modes = os.environ.get("MS_MODES", ",".join(ef.MODES)).split(",")
    for mode in modes:
        canonical = (mode == "canonical")
        label = mode if canonical else mode + fr.DESIGN_TAG   # e.g. fl, fl_u
        tag = (f"canonical_e{fr.EVAL_SEED}" if canonical
               else f"t{fr.TRAIN_SEED}_e{fr.EVAL_SEED}_{label}")
        res_path = f"{OUT_RES}/{tag}.json"
        if os.path.exists(res_path):
            continue
        runners = fr.build_runners(config, topology, 1, design="canonical" if canonical else None)
        for tls_id, r in runners.items():
            r.agent.policy_net.load_state_dict(
                torch.load(fr.ckpt_name(mode, tls_id), weights_only=True))
            r.agent.epsilon = 0.0
        paths = {k: f"{OUT_TMP}/{k}_{tag}.xml" for k in ("tripinfo", "summary", "queue")}
        fr.run_episode(runners, config, learn=False, sumo_extra=[
            "--tripinfo-output", paths["tripinfo"], "--summary", paths["summary"],
            "--queue-output", paths["queue"], *ef.FAIR_ARGS])
        out = {"train_seed": 0 if canonical else fr.TRAIN_SEED, "eval_seed": fr.EVAL_SEED, "mode": label,
               **ef.trip_metrics(paths["tripinfo"]), **ef.summary_metrics(paths["summary"]),
               "queue": ef.junction_queues(paths["queue"], list(config))}
        with open(res_path, "w") as f:
            json.dump(out, f)
        for p in paths.values():
            os.remove(p)
        print(f"done {tag}", flush=True)


if __name__ == "__main__":
    main()
