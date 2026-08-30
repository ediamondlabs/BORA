"""
Compute stabilized mean throughput and 95% CI for all evaluation conditions.

For each experiment directory, finds every condition subdir, locates the
PADRE agent CSV (excludes Gredy_Trffc), filters to throughput rows, and
computes the per-episode mean over the last STABILIZATION_STEPS steps.

Usage (run from resilient-backbone/):
    python src/Utils/compute_throughput_stats.py

Output: one row per (experiment, condition) sorted by experiment name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STABILIZATION_STEPS = 500   # last N steps of each episode used for the mean
SKIP_AGENTS = {"Gredy_Trffc"}  # baseline agents to skip

BASE = Path("out/evaluations")

# Experiments to report — edit this list or set to None to scan all subdirs.
EXPERIMENTS: list[str] | None = [
    # Heuristic harm sweep
    "29_04_jammer_2byz_v3_mag0.0",
    "29_04_jammer_2byz_v3_mag0.5",
    "29_04_jammer_2byz_v3_mag1.0",
    "29_04_jammer_2byz_v3_mag2.0",
    "29_04_jammer_2byz_v3_mag4.0",
    # Temporal replay sweep
    "29_04_obs_replay_500_2byz_mag0.0",
    "29_04_obs_replay_500_2byz_mag0.5",
    "29_04_obs_replay_500_2byz_mag1.0",
    "29_04_obs_replay_500_2byz_mag2.0",
    "29_04_obs_replay_500_2byz_mag4.0",
    # N=7 direction lie sweep
    "18_05_N7_2byz_re_mag0.0",
    "18_05_N7_2byz_re_mag0.5",
    "18_05_N7_2byz_re_mag1.0",
    "18_05_N7_2byz_re_mag2.0",
    "18_05_N7_2byz_re_mag4.0",
    # N_byz scaling
    "29_04_user_dir_nbyz_byz1",
    "29_04_user_dir_nbyz_byz2",
    "29_04_user_dir_nbyz_byz3",
    "29_04_user_dir_nbyz_byz4",
    # 1-byz direction lie
    "29_04_1byz_dirlie_2000ep",
    # BABEL evaluations
    "29_04_learned_500ep",
    "29_04_learned_local",
    "29_04_learned_1byz_local_500ep",
    "29_04_learned_coordinated_500ep",
    # Adversarial training
    "adv_honest_2byz_heuristic",
    "adv_honest_2byz_learned",
    # Gossip1 defense
    "29_04_resiliency_gossip1_2byz",
]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _agent_csv(cond_dir: Path) -> Path | None:
    """Return the first non-baseline agent CSV found under cond_dir."""
    for csv in sorted(cond_dir.rglob("*.csv")):
        if csv.stem not in SKIP_AGENTS and "evaluation_config" not in csv.name:
            return csv
    return None


def stabilized_stats(csv_path: Path) -> tuple[float, float, int] | None:
    """
    Return (mean, ci_95, n_episodes) of per-episode stabilized throughput.
    Returns None if the file cannot be parsed.
    """
    df = pd.read_csv(csv_path)
    if "value_name" in df.columns:
        df = df[df["value_name"] == "throughput"]
    if df.empty:
        return None

    max_step = df["ep_step"].max()
    cutoff = max_step - STABILIZATION_STEPS
    stable = df[df["ep_step"] > cutoff]
    if stable.empty:
        return None

    per_ep = stable.groupby("episode")["value"].mean()
    n = len(per_ep)
    mean = float(per_ep.mean())
    ci = float(1.96 * per_ep.std() / np.sqrt(n)) if n > 1 else 0.0
    return mean, ci, n


def compute_experiment(exp_dir: Path) -> list[dict]:
    rows = []
    if not exp_dir.is_dir():
        return rows

    for cond_dir in sorted(exp_dir.iterdir()):
        if not cond_dir.is_dir():
            continue
        csv = _agent_csv(cond_dir)
        if csv is None:
            continue
        result = stabilized_stats(csv)
        if result is None:
            continue
        mean, ci, n = result
        rows.append({
            "experiment": exp_dir.name,
            "condition":  cond_dir.name,
            "mean_Mbps":  round(mean, 3),
            "ci_95_Mbps": round(ci, 3),
            "n_episodes": n,
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not BASE.exists():
        print(f"ERROR: {BASE} not found. Run from resilient-backbone/.", file=sys.stderr)
        sys.exit(1)

    exp_names = EXPERIMENTS if EXPERIMENTS is not None else sorted(
        p.name for p in BASE.iterdir() if p.is_dir()
    )

    all_rows: list[dict] = []
    for name in exp_names:
        exp_dir = BASE / name
        if not exp_dir.exists():
            print(f"  WARNING: {name} not found, skipping", file=sys.stderr)
            continue
        rows = compute_experiment(exp_dir)
        all_rows.extend(rows)
        print(f"  {name}: {len(rows)} condition(s)", file=sys.stderr)

    if not all_rows:
        print("No data found.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(all_rows)

    # Pretty-print to stdout
    col_w = max(df["experiment"].str.len().max(), 12)
    cond_w = max(df["condition"].str.len().max(), 12)
    header = f"{'experiment':<{col_w}}  {'condition':<{cond_w}}  {'mean [Mbit/s]':>14}  {'±95% CI':>9}  {'n_ep':>6}"
    print(header)
    print("-" * len(header))
    prev_exp = None
    for _, row in df.iterrows():
        if prev_exp and row["experiment"] != prev_exp:
            print()
        print(
            f"{row['experiment']:<{col_w}}  "
            f"{row['condition']:<{cond_w}}  "
            f"{row['mean_Mbps']:>14.3f}  "
            f"{row['ci_95_Mbps']:>9.3f}  "
            f"{row['n_episodes']:>6}"
        )
        prev_exp = row["experiment"]

    # Also write a CSV for easy import
    out_csv = Path("out/throughput_stats.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
