"""
Regenerate 0_Thesis/figures/data/steps_learned_local.csv.

Sources point at the 4000-episode evals so the plotted curves reproduce the
shipped steps_learned_local.csv exactly (verified 2026-06-24: all 8 plotted
columns regenerate with maxΔ=0.000). Legend semantics match
steps_learned_local.tex:
  NoByzantine   : out/evaluations/29_04_nobyz_4000ep/learned_obs_user_dir          (clean baseline, 16.78)
  LearnedLocal  : out/evaluations/29_04_2byz_local_4000ep/learned_obs_user_dir     (Dir. Learned rho=0)
  Oracle        : out/evaluations/29_04_2byz_oracle_4000ep/learned_obs_user_dir    (Dir. Learned rho=inf)
  Coordinated   : out/evaluations/29_04_learned_coordinated/learned_obs_user_dir   (Dir. Learned Coordinated)
  Full          : out/evaluations/29_04_2byz_full_obs_60M_4000ep/learned_obs_full  (BORA rho=1, f=2)
  FullFloat     : out/evaluations/oracle_fullvec_2byz_N5_2000ep/learned_obs_full   (BORA rho=inf, f=2)
  FullGossipF1  : out/evaluations/29_04_1byz_full_obs_30M_4000ep/learned_obs_full  (BORA rho=1, f=1)
  FullFloatF1   : out/evaluations/oracle_fullvec_1byz_N5_2000ep/learned_obs_full   (BORA rho=inf, f=1)

(The old UserDirLie column was dropped 2026-06-24: it was a heuristic
direction-lie reference on the realistic-observation baseline (12.79), never
plotted by steps_learned_local.tex, and its shipped values matched no on-disk
eval. The heuristic dir-lie lives in the steps_jammer/harm figures instead.)

Run from resilient-backbone/:
    python src/Utils/visualize/byzantine/gen_steps_learned_local.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EVAL_BASE = Path("out/evaluations")

SOURCES: dict[str, Path] = {
    "NoByzantine":  EVAL_BASE / "29_04_nobyz_4000ep"            / "learned_obs_user_dir",
    "LearnedLocal": EVAL_BASE / "29_04_2byz_local_4000ep"       / "learned_obs_user_dir",
    "Oracle":       EVAL_BASE / "29_04_2byz_oracle_4000ep"      / "learned_obs_user_dir",
    "Coordinated":  EVAL_BASE / "29_04_learned_coordinated"     / "learned_obs_user_dir",
    "Full":         EVAL_BASE / "29_04_2byz_full_obs_60M_4000ep" / "learned_obs_full",
    "FullFloat":    EVAL_BASE / "oracle_fullvec_2byz_N5_2000ep"   / "learned_obs_full",
    "FullGossipF1": EVAL_BASE / "29_04_1byz_full_obs_30M_4000ep"  / "learned_obs_full",
    "FullFloatF1":  EVAL_BASE / "oracle_fullvec_1byz_N5_2000ep"   / "learned_obs_full",
}

OUTPUT_CSV = Path("../0_Thesis/figures/data/steps_learned_local.csv")

# Step range to include in output (match existing x axis: 10..2000 step 10,
# but some learned CSVs start at 0 — we clip to the common range).
X_MIN = 10
X_MAX = 2000
X_STEP = 10

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_csvs(cond_dir: Path) -> pd.DataFrame:
    """Recursively load RL-agent CSVs under cond_dir; return concatenated DataFrame.

    Skips greedy-baseline files (Gredy_Trffc.csv, GreedyMaximizer.csv) which are
    written alongside the RL agent CSV by evaluate_MANETAgents but not needed here.
    """
    _SKIP = {"Gredy_Trffc.csv", "GreedyMaximizer.csv"}
    csvs = [p for p in sorted(cond_dir.rglob("*.csv")) if p.name not in _SKIP]
    if not csvs:
        raise FileNotFoundError(f"No RL-agent CSVs found under {cond_dir}")
    dfs = []
    for p in csvs:
        df = pd.read_csv(p)
        if "value" not in df.columns and "throughput" in df.columns:
            df = df.rename(columns={"throughput": "value"})
        # Drop non-throughput rows (e.g. detection_error) that contaminate means.
        if "value_name" in df.columns:
            df = df[df["value_name"] == "throughput"]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _per_step_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-step mean and 95 % CI (across episodes) as a DataFrame.

    Returns columns: ep_step, mean, up, low.
    """
    # Average within each (episode, step) pair in case of duplicate rows.
    per_ep = (
        df.groupby(["ep_step", "episode"])["value"]
        .mean()
        .reset_index(name="v")
    )
    agg = per_ep.groupby("ep_step")["v"].agg(["mean", "std", "count"])
    agg["ci"] = 1.96 * agg["std"] / np.sqrt(agg["count"])
    agg["up"]  = agg["mean"] + agg["ci"]
    agg["low"] = agg["mean"] - agg["ci"]
    return agg[["mean", "up", "low"]].reset_index()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    x_vals = np.arange(X_MIN, X_MAX + 1, X_STEP)

    frames: dict[str, pd.DataFrame] = {}
    for label, src_dir in SOURCES.items():
        print(f"Loading {label} from {src_dir} …", end=" ", flush=True)
        raw = _load_csvs(src_dir)
        stats = _per_step_stats(raw)
        # Reindex to the common x grid; steps outside range get NaN.
        stats = stats.set_index("ep_step").reindex(x_vals)
        frames[label] = stats
        n_ep = raw["episode"].nunique()
        print(f"({n_ep} episodes, {raw['ep_step'].nunique()} unique steps)")

    # Build merged output DataFrame.
    out = pd.DataFrame({"x": x_vals})
    for label, stats in frames.items():
        out[f"mean{label}"] = stats["mean"].values
        out[f"up{label}"]   = stats["up"].values
        out[f"low{label}"]  = stats["low"].values

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False, float_format="%.8f")
    print(f"\nWrote {len(out)} rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
