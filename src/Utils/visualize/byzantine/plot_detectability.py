"""
Plot the obs_user_dir attacker tradeoff: harm vs. detectability vs. magnitude.

Reads a magnitude sweep (same experiments as compare_magnitude_sweep.py) and
produces a dual-axis plot:
  - Left y-axis:  throughput degradation Δτ [%]
  - Right y-axis: mean detection error ‖offset‖₂ per Byzantine node
  - x-axis:       magnitude

The detection error is read from the "detection_error" value_name column in the
evaluation CSVs (written by evaluate_MANETAgents.py when obs_user_dir is active).

A high harm / low detection-error ratio indicates the "stealth" regime where the
attack is maximally disruptive but hardest to flag.

Usage (run from the resilient-backbone/ root):
    python src/Utils/visualize/byzantine/plot_detectability.py \\
        --base       out/evaluations \\
        --names      sweep_m0.0 sweep_m0.5 sweep_m1.0 sweep_m2.0 sweep_m4.0 \\
        --magnitudes 0.0 0.5 1.0 2.0 4.0

Optional arguments:
    --base          Root directory containing per-experiment subdirectories.
    --names         Ordered list of experiment names, one per magnitude.
    --magnitudes    Matching ordered list of magnitude float values.
    --condition     Attack condition to analyse (default: obs_user_dir).
    --baseline-idx  Index into --names used as throughput denominator. Default: 0.
    --output        Output directory. Default: out/visualizations/detectability
    --no-pgf        Skip PGFPlots export.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STABILIZATION_STEPS = 500
THROUGHPUT_SCALE = 1.0
BASELINE_AGENTS: list[str] = ["Gredy_Trffc"]
AGENT_OR_BASE_DICT: dict = {"Agent": "PADRE", "Baseline": "GTM"}
_BASELINE_SUBDIR = "no_byzantine"

_DISPLAY_LABELS: dict[str, str] = {
    "no_byzantine":    "No Byzantine",
    "obs_user_dir":    "User Dir Lie",
    "obs_noise":       "Obs. Noise",
    "obs_capacity":    "Capacity Lie",
    "obs_demand":      "Demand Lie",
}

_FIG_W = 4.5
_FIG_H = 3.0
_DPI = 450
USE_LATEX = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_agent_dfs(experiment_dir: Path) -> dict[str, dict[str, pd.DataFrame]]:
    result: dict[str, dict[str, pd.DataFrame]] = {}
    if not experiment_dir.exists():
        print(f"  [skip] not found: {experiment_dir}")
        return result
    for cond_dir in sorted(experiment_dir.iterdir()):
        if not cond_dir.is_dir():
            continue
        label = _DISPLAY_LABELS.get(cond_dir.name, cond_dir.name)
        agents: dict[str, pd.DataFrame] = {}
        for csv_path in sorted(cond_dir.rglob("*.csv")):
            df = pd.read_csv(csv_path)
            if not {"ep_step", "episode", "value_name", "value"}.issubset(df.columns):
                continue
            mask = df["value_name"] == "throughput"
            df.loc[mask, "value"] *= THROUGHPUT_SCALE
            agents[csv_path.stem] = df
        if agents:
            result[label] = agents
    return result


def _stabilized_per_episode(df: pd.DataFrame, value_name: str = "throughput") -> np.ndarray:
    tp = df[df["value_name"] == value_name]
    if tp.empty:
        return np.array([])
    max_step = tp["ep_step"].max()
    late = tp[tp["ep_step"] > max_step - STABILIZATION_STEPS]
    return late.groupby("episode")["value"].mean().values


def _deg_mean_ci(base_mean: float, attack_episodes: np.ndarray) -> tuple[float, float]:
    if base_mean == 0 or np.isnan(base_mean):
        return 0.0, 0.0
    per_ep = (base_mean - attack_episodes) / base_mean * 100
    mean = float(per_ep.mean())
    ci = float(1.96 * per_ep.std() / np.sqrt(len(per_ep))) if len(per_ep) > 1 else 0.0
    return mean, ci


# ---------------------------------------------------------------------------
# Build tradeoff data
# ---------------------------------------------------------------------------

def build_tradeoff(
    base_dir: Path,
    names: list[str],
    magnitudes: list[float],
    condition: str,
    baseline_idx: int,
) -> dict[str, dict]:
    """Return {agent: {"magnitudes", "harms", "harm_cis", "det_errors", "det_cis"}}"""
    experiments = []
    for name in names:
        print(f"  Loading {name} …")
        experiments.append(_load_agent_dfs(base_dir / name))

    baseline_label = _DISPLAY_LABELS.get(_BASELINE_SUBDIR, _BASELINE_SUBDIR)
    attack_label = _DISPLAY_LABELS.get(condition, condition)

    all_agents = sorted({
        agent
        for exp in experiments
        for cond_data in exp.values()
        for agent in cond_data
    })

    baseline_cond_data = experiments[baseline_idx].get(baseline_label, {})

    tradeoff: dict[str, dict] = {}
    for agent in all_agents:
        if agent not in baseline_cond_data:
            continue
        base_mean = _stabilized_per_episode(baseline_cond_data[agent]).mean()
        if base_mean == 0 or np.isnan(base_mean):
            continue

        mags, harms, harm_cis, det_errors, det_cis = [], [], [], [], []
        for mag, exp in zip(magnitudes, experiments):
            cond_data = exp.get(attack_label, {})
            if agent not in cond_data:
                continue
            df = cond_data[agent]

            # Throughput degradation.
            tp_vals = _stabilized_per_episode(df, "throughput")
            if len(tp_vals) == 0:
                continue
            harm, harm_ci = _deg_mean_ci(base_mean, tp_vals)

            # Detection error (logged by evaluator when obs_user_dir is active).
            det_vals = _stabilized_per_episode(df, "detection_error")
            if len(det_vals) > 0:
                det_mean = float(det_vals.mean())
                det_ci = float(
                    1.96 * det_vals.std() / np.sqrt(len(det_vals))
                    if len(det_vals) > 1 else 0.0
                )
            else:
                # Fall back to analytical value: detection_error = magnitude
                # (since offset = [mag, 0] by default, ‖offset‖₂ = mag).
                det_mean = float(mag)
                det_ci = 0.0

            mags.append(mag)
            harms.append(harm)
            harm_cis.append(harm_ci)
            det_errors.append(det_mean)
            det_cis.append(det_ci)

        if mags:
            tradeoff[agent] = {
                "magnitudes": mags,
                "harms": harms,
                "harm_cis": harm_cis,
                "det_errors": det_errors,
                "det_cis": det_cis,
            }

    return tradeoff


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save(output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{name}.png", dpi=_DPI, bbox_inches="tight")
    plt.savefig(output_dir / f"{name}.svg", bbox_inches="tight")
    print(f"  Saved: {name}.png / .svg")


def plot_tradeoff(
    tradeoff: dict[str, dict],
    condition: str,
    output_dir: Path,
    no_pgf: bool,
) -> None:
    """Dual-axis: harm % (left) and detection_error (right) vs. magnitude."""
    if USE_LATEX:
        plt.rcParams["text.usetex"] = True

    sns.set_theme(style="whitegrid", context="paper")
    harm_color = sns.color_palette("Set1")[0]
    det_color = sns.color_palette("Set1")[1]

    for agent, data in tradeoff.items():
        mags = np.array(data["magnitudes"])
        harms = np.array(data["harms"])
        harm_cis = np.array(data["harm_cis"])
        det_errors = np.array(data["det_errors"])
        det_cis = np.array(data["det_cis"])

        is_baseline = agent in BASELINE_AGENTS
        agent_label = (
            AGENT_OR_BASE_DICT.get("Baseline", agent)
            if is_baseline
            else AGENT_OR_BASE_DICT.get("Agent", agent)
        )
        linestyle = "--" if is_baseline else "-"

        fig, ax1 = plt.subplots(figsize=(_FIG_W, _FIG_H), constrained_layout=True)
        ax2 = ax1.twinx()

        # Harm curve on left axis.
        ax1.plot(mags, harms, color=harm_color, linewidth=2,
                 linestyle=linestyle, marker="o", markersize=5,
                 label=r"$\Delta\tau$ (harm)")
        ax1.fill_between(mags, harms - harm_cis, harms + harm_cis,
                         color=harm_color, alpha=0.15)
        ax1.axhline(0, color="black", linewidth=0.6, linestyle=":")
        ax1.set_xlabel("magnitude", fontsize=9)
        ax1.set_ylabel(r"$\Delta\tau\,[\%]$", color=harm_color, fontsize=9)
        ax1.tick_params(axis="y", labelcolor=harm_color, labelsize=8)

        # Detection error on right axis.
        ax2.plot(mags, det_errors, color=det_color, linewidth=2,
                 linestyle=linestyle, marker="s", markersize=5,
                 label=r"detection error $\|\epsilon\|_2$")
        ax2.fill_between(mags, det_errors - det_cis, det_errors + det_cis,
                         color=det_color, alpha=0.15)
        ax2.set_ylabel(r"detection error $\|\epsilon\|_2$", color=det_color, fontsize=9)
        ax2.tick_params(axis="y", labelcolor=det_color, labelsize=8)

        # Combined legend.
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   fontsize=8, loc="upper left")

        ax1.set_title(
            f"{_DISPLAY_LABELS.get(condition, condition)} ({agent_label})",
            fontsize=9,
        )

        _save(output_dir, f"detectability_{agent}")
        plt.close()

        if not no_pgf:
            _export_pgf(data, agent, condition, output_dir)

    if USE_LATEX:
        plt.rcParams["text.usetex"] = False


def _export_pgf(
    data: dict,
    agent: str,
    condition: str,
    output_dir: Path,
) -> None:
    pgf_dir = output_dir / "pgfplots"
    pgf_dir.mkdir(parents=True, exist_ok=True)
    csv_name = f"detectability_{agent}.csv"

    df = pd.DataFrame({
        "x":         data["magnitudes"],
        "harm":      data["harms"],
        "harm_up":   [h + c for h, c in zip(data["harms"], data["harm_cis"])],
        "harm_lo":   [h - c for h, c in zip(data["harms"], data["harm_cis"])],
        "det":       data["det_errors"],
        "det_up":    [d + c for d, c in zip(data["det_errors"], data["det_cis"])],
        "det_lo":    [d - c for d, c in zip(data["det_errors"], data["det_cis"])],
    })
    df.to_csv(pgf_dir / csv_name, index=False)

    lines = [
        f"% Detectability tradeoff — {condition} — {agent}",
        r"% Required preamble: \usepackage{pgfplots,pgfplotstable}",
        r"%   \usepgfplotslibrary{fillbetween}  \pgfplotsset{compat=1.18}",
        "",
        r"\begin{tikzpicture}",
        r"    \begin{axis}[",
        r"        grid=major, grid style={dashed, gray!30},",
        r"        width=\linewidth,",
        r"        axis y line*=left,",
        r"        xlabel={\small magnitude},",
        r"        ylabel={\small $\Delta\tau\,[\%]$},",
        r"        tick label style={font=\small}, label style={font=\small},",
        r"    ]",
        f"    \\addplot [name path=harmup, draw=none, forget plot] table [x=x, y=harm_up, col sep=comma] {{{csv_name}}};",
        f"    \\addplot [name path=harmlo, draw=none, forget plot] table [x=x, y=harm_lo, col sep=comma] {{{csv_name}}};",
        r"    \addplot [red!30, opacity=0.3, forget plot] fill between [of=harmup and harmlo];",
        f"    \\addplot [red, thick, mark=*, mark size=1.5pt] table [x=x, y=harm, col sep=comma] {{{csv_name}}};",
        r"    \addlegendentry{harm $\Delta\tau$}",
        r"    \end{axis}",
        r"    \begin{axis}[",
        r"        axis y line*=right, axis x line=none,",
        r"        ylabel={\small detection error $\|\epsilon\|_2$},",
        r"        tick label style={font=\small}, label style={font=\small},",
        r"    ]",
        f"    \\addplot [name path=detup, draw=none, forget plot] table [x=x, y=det_up, col sep=comma] {{{csv_name}}};",
        f"    \\addplot [name path=detlo, draw=none, forget plot] table [x=x, y=det_lo, col sep=comma] {{{csv_name}}};",
        r"    \addplot [blue!30, opacity=0.3, forget plot] fill between [of=detup and detlo];",
        f"    \\addplot [blue, thick, mark=square*, mark size=1.5pt] table [x=x, y=det, col sep=comma] {{{csv_name}}};",
        r"    \addlegendentry{detection error}",
        r"    \end{axis}",
        r"\end{tikzpicture}",
    ]
    (pgf_dir / f"detectability_{agent}.tex").write_text("\n".join(lines))
    print(f"  PGFPlots: detectability_{agent}.csv / .tex")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot obs_user_dir harm vs. detectability vs. magnitude."
    )
    parser.add_argument(
        "--base", default="out/evaluations",
        help="Root directory containing per-experiment subdirectories.",
    )
    parser.add_argument(
        "--names", nargs="+", required=True,
        help="Ordered list of experiment names, one per magnitude.",
    )
    parser.add_argument(
        "--magnitudes", nargs="+", type=float, required=True,
        help="Matching ordered list of magnitude float values.",
    )
    parser.add_argument(
        "--condition", default="obs_user_dir",
        help="Attack condition to analyse (default: obs_user_dir).",
    )
    parser.add_argument(
        "--baseline-idx", type=int, default=0,
        help="Index into --names used as throughput baseline (default: 0).",
    )
    parser.add_argument(
        "--output", default="out/visualizations/detectability",
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--no-pgf", action="store_true",
        help="Skip PGFPlots CSV/tex export.",
    )
    args = parser.parse_args()

    if len(args.names) != len(args.magnitudes):
        parser.error("--names and --magnitudes must have the same number of entries.")

    output_dir = Path(args.output)
    base_dir = Path(args.base)

    print(f"\nDetectability analysis — {len(args.names)} experiments")
    print(f"Magnitudes: {args.magnitudes}\n")

    tradeoff = build_tradeoff(
        base_dir=base_dir,
        names=args.names,
        magnitudes=args.magnitudes,
        condition=args.condition,
        baseline_idx=args.baseline_idx,
    )

    if not tradeoff:
        print("No usable data found.")
        raise SystemExit(1)

    plot_tradeoff(tradeoff, args.condition, output_dir, args.no_pgf)
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
