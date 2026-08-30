"""
Diagnostic: does obs_capacity cause honest nodes to cluster toward the Byzantine node?

Supervisor hypothesis: advertising inflated capacity lures legitimate nodes toward
the Byzantine node, inducing a suboptimal network topology and lowering throughput.
Current results show improved throughput — this script checks whether the RL policy
actually responds to the capacity lie at all, by comparing node-to-node distances
under obs_capacity vs no_byzantine.

Metric: mean distance of honest nodes to node[-1] (the Byzantine node slot).
If the capacity lie is an effective attractor, this distance should be lower under
obs_capacity than under no_byzantine across the episode.

Uses evaluate_from_config to guarantee the obs space matches the trained model.
Hooks into the environment at the class level to record node positions each step.

Usage (run from resilient-backbone/ root):
    python src/Utils/check_capacity_trajectory.py \\
        --model "out/models/29_04/**" \\
        --eps   50
"""

import argparse
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

sys.path.insert(0, str(Path("src").resolve()))

from unittest.mock import MagicMock
sys.modules.setdefault("tkinter", MagicMock())

import MANET.Environments as envs
from evaluate_MANETAgents import evaluate_from_config

# ---------------------------------------------------------------------------
# Global position recorder — filled by the monkeypatched step
# ---------------------------------------------------------------------------

_records: list[dict] = []   # {"ep_step": int, "hon": [...], "ref": [...]}
_original_step = envs.Basic_MANETEnv.step


def _recording_step(self, action):
    result = _original_step(self, action)
    hon = [np.array(n.pos) for n in self.nodes if not getattr(n, "is_byzantine", False)]
    ref = [np.array(n.pos) for n in self.nodes if     getattr(n, "is_byzantine", False)]
    # For no_byzantine condition use last node as reference slot
    if not ref:
        ref = [np.array(self.nodes[-1].pos)]
    mean_dist = np.mean([
        np.min([np.linalg.norm(h - r) for r in ref])
        for h in hon
    ]) if hon and ref else np.nan
    _records.append({"ep_step": self.ep_step, "dist": mean_dist})
    return result


envs.Basic_MANETEnv.step = _recording_step

# ---------------------------------------------------------------------------
# Config builder (mirrors run_byzantine_sweep.py BASE_EVAL_PARAMS)
# ---------------------------------------------------------------------------

_BASE: dict = {
    "allow_early_ep_finish": False,
    "attackerModel": "Static_ClusterJammers",
    "deterministic": True,
    "ep_length": 500,
    "jammer_ep_steps": 0,
    "jammersSpawnNextToUsers": False,
    "n_eval_eps": 30,
    "networkConnectedAtStart": False,
    "numbOfJammers": 0,
    "numbOfUsers": [2, 21],
    "obs_type": "real",
    "render_mode": None,
    "routing_func": "sequential_max_flow",
    "seed": 22,
    "state_loss_rate": None,
    "step_size": 2,
}


def _build(model_pattern: str, n_byz: int, capacity_factor: float,
           n_eps: int, exp_name: str) -> dict:
    params = dict(_BASE)
    params["n_eval_eps"] = n_eps
    params["numbOfByzantineNodes"] = n_byz
    if n_byz > 0:
        params["byzantine_attack_type"] = "obs_capacity"
        params["byzantine_obs_capacity_factor"] = capacity_factor
    return {
        "mode": "evaluate",
        "agent_patterns": [model_pattern],
        "baselines_with_cs": [],
        "env_constr": "Dynamic_MANETEnv",
        "eval_params": params,
        "eval_type": "grouped_eval",
        "log_dir": f"out/evaluations/_cap_check/{exp_name}",
        "log_results": False,
        "log_state": False,
        "log_eval": False,
    }


def _run_condition(config: dict) -> np.ndarray:
    """Run evaluate_from_config and return recorded distances as (n_eps, ep_len)."""
    global _records
    _records = []

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        tmp = f.name
    try:
        evaluate_from_config(tmp)
    finally:
        Path(tmp).unlink(missing_ok=True)

    # Group by episode (ep_step resets to 0 each episode)
    eps: dict[int, list] = defaultdict(list)
    ep_idx = 0
    prev_step = -1
    for r in _records:
        if r["ep_step"] <= prev_step:
            ep_idx += 1
        eps[ep_idx].append(r["dist"])
        prev_step = r["ep_step"]

    ep_len = config["eval_params"]["ep_length"]
    arr = np.full((len(eps), ep_len), np.nan)
    for i, dists in eps.items():
        arr[i, : len(dists)] = dists
    return arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help='Glob for model .zip, e.g. "out/models/29_04/**"')
    parser.add_argument("--eps", type=int, default=50)
    parser.add_argument("--capacity-factor", type=float, default=5.0)
    parser.add_argument("--output", default="out/visualizations/capacity_trajectory")
    args = parser.parse_args()

    print("Running no_byzantine …")
    cfg_base = _build(args.model, n_byz=0, capacity_factor=1.0,
                      n_eps=args.eps, exp_name="no_byzantine")
    dists_base = _run_condition(cfg_base)

    print("Running obs_capacity …")
    cfg_cap = _build(args.model, n_byz=1, capacity_factor=args.capacity_factor,
                     n_eps=args.eps, exp_name="obs_capacity")
    dists_cap = _run_condition(cfg_cap)

    # -----------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    ep_len = cfg_base["eval_params"]["ep_length"]
    steps = np.arange(1, ep_len + 1)
    palette = sns.color_palette("Set1", 2)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for dists, label, color in [
        (dists_base, "No Byzantine (reference node honest)", palette[0]),
        (dists_cap,  f"obs_capacity ×{args.capacity_factor:.0f}", palette[1]),
    ]:
        n = dists.shape[0]
        mean = np.nanmean(dists, axis=0)
        se   = np.nanstd(dists, axis=0) / np.sqrt(n)
        ax.plot(steps, mean, color=color, label=label, linewidth=1.5)
        ax.fill_between(steps, mean - 1.96 * se, mean + 1.96 * se,
                        color=color, alpha=0.15)

    ax.set_xlabel("Episode step")
    ax.set_ylabel("Mean distance to Byzantine node slot")
    ax.set_title("Does obs_capacity attract honest nodes?")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "capacity_trajectory.png", dpi=200)
    fig.savefig(out_dir / "capacity_trajectory.svg")
    plt.close(fig)

    final_base = np.nanmean(dists_base[:, -50:])
    final_cap  = np.nanmean(dists_cap[:,  -50:])
    diff_pct   = 100 * (final_base - final_cap) / final_base
    print("\nMean distance to Byzantine slot (last 50 steps):")
    print(f"  no_byzantine : {final_base:.4f}")
    print(f"  obs_capacity : {final_cap:.4f}")
    print(f"  Δ            : {diff_pct:+.1f}%  "
          f"({'closer → attractor active' if diff_pct > 0 else 'no attractor effect'})")
    print(f"\nPlots saved to {out_dir}/")


if __name__ == "__main__":
    main()
