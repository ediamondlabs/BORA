"""
Polar plot of obs_user_dir throughput degradation vs. offset angle.

Reads one evaluation experiment per offset angle and plots degradation (%)
on polar axes (angle = offset direction, radius = degradation %).

Usage (run from the resilient-backbone/ root):
    python src/Utils/visualize/byzantine/plot_direction_sensitivity.py \\
        --base    out/evaluations \\
        --names   sweep_a0 sweep_a45 sweep_a90 sweep_a135 \\
                  sweep_a180 sweep_a225 sweep_a270 sweep_a315 \\
        --angles  0 45 90 135 180 225 270 315

Optional arguments:
    --base          Root directory containing per-experiment subdirectories.
                    Default: out/evaluations
    --names         Ordered list of experiment names, one per angle.
    --angles        Matching ordered list of angles in degrees.
    --condition     Attack condition to plot (default: obs_user_dir).
    --baseline-idx  Index into --names used as the throughput denominator.
                    Default: 0 (first entry).
    --output        Output directory. Default: out/visualizations/angle_sweep
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
    "no_byzantine":               "No Byzantine",
    "obs_user_dir":               "User Dir Lie",
    "obs_capacity":               "Capacity Lie",
    "obs_demand":                 "Demand Lie",
    "obs_interference_lie":       "Interference Lie (hide)",
    "obs_interference_amplify":   "Interference Amplify",
    "obs_noise":                  "Obs. Noise",
    "obs_all":                    "All Obs. Attacks",
    "obs_capacity_deflate":       "Capacity Deflate",
    "obs_demand_deflate":         "Demand Deflate",
    "obs_replay":                 "Obs. Replay",
}

_FIG_W = 4.5
_FIG_H = 4.5
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


def _stabilized_per_episode(df: pd.DataFrame) -> np.ndarray:
    tp = df[df["value_name"] == "throughput"]
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
# Build angle curves
# ---------------------------------------------------------------------------

def build_angle_curves(
    base_dir: Path,
    names: list[str],
    angles: list[float],
    condition: str,
    baseline_idx: int,
) -> dict[str, dict]:
    """Return {agent: {"angles": [...], "means": [...], "cis": [...]}}"""
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

    baseline_exp = experiments[baseline_idx]
    baseline_cond_data = baseline_exp.get(baseline_label, {})

    curves: dict[str, dict] = {}
    for agent in all_agents:
        if agent not in baseline_cond_data:
            continue
        base_mean = _stabilized_per_episode(baseline_cond_data[agent]).mean()
        if base_mean == 0 or np.isnan(base_mean):
            continue

        angle_list, means, cis = [], [], []
        for angle, exp in zip(angles, experiments):
            cond_data = exp.get(attack_label, {})
            if agent not in cond_data:
                continue
            ep_vals = _stabilized_per_episode(cond_data[agent])
            if len(ep_vals) == 0:
                continue
            mean, ci = _deg_mean_ci(base_mean, ep_vals)
            angle_list.append(angle)
            means.append(mean)
            cis.append(ci)

        if angle_list:
            curves[agent] = {"angles": angle_list, "means": means, "cis": cis}

    return curves


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save(output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{name}.png", dpi=_DPI, bbox_inches="tight")
    plt.savefig(output_dir / f"{name}.svg", bbox_inches="tight")
    print(f"  Saved: {name}.png / .svg")


def plot_polar(
    curves: dict[str, dict],
    condition: str,
    output_dir: Path,
    no_pgf: bool,
) -> None:
    """Polar plot: radius = degradation %, angle = offset direction."""
    sns.set_theme(style="whitegrid", context="paper")

    if USE_LATEX:
        plt.rcParams["text.usetex"] = True

    palette = sns.color_palette("Set1", len(curves))

    for (agent, data), color in zip(curves.items(), palette):
        angles_deg = np.array(data["angles"])
        means = np.array(data["means"])
        cis = np.array(data["cis"])

        angles_rad = np.deg2rad(angles_deg)

        # Only close the polygon if all 8 expected angles (every 45°) are present.
        # Closing a partial curve produces a misleading chord across missing data.
        expected_step = 45.0
        spans_full_circle = (
            len(angles_deg) >= 2
            and abs((angles_deg[-1] - angles_deg[0] + 360) % 360
                    - (len(angles_deg) - 1) * expected_step) < 1.0
            and abs(angles_deg[-1] - angles_deg[0] + expected_step - 360) < 1.0
        )
        if spans_full_circle:
            angles_rad_plot = np.append(angles_rad, angles_rad[0])
            means_plot = np.append(means, means[0])
            ci_plot = np.append(cis, cis[0])
        else:
            angles_rad_plot = angles_rad
            means_plot = means
            ci_plot = cis

        is_baseline = agent in BASELINE_AGENTS
        agent_label = (
            AGENT_OR_BASE_DICT.get("Baseline", agent)
            if is_baseline
            else AGENT_OR_BASE_DICT.get("Agent", agent)
        )

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"},
                               figsize=(_FIG_W, _FIG_H), constrained_layout=True)

        ax.fill_between(
            angles_rad_plot,
            means_plot - ci_plot,
            means_plot + ci_plot,
            color=color,
            alpha=0.2,
        )
        ax.plot(
            angles_rad_plot,
            means_plot,
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            label=f"{_DISPLAY_LABELS.get(condition, condition)} ({agent_label})",
        )

        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"],
                           fontsize=8)
        # set_ylabel on polar axes labels the r-axis but renders at an awkward fixed
        # position; use set_rlabel_position + annotate instead for reliable placement.
        ax.set_rlabel_position(67.5)
        ax.annotate(
            r"$\Delta\tau\,[\%]$",
            xy=(np.deg2rad(67.5), ax.get_rmax()),
            xytext=(5, 5), textcoords="offset points",
            fontsize=8, color="grey",
        )
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)

        _save(output_dir, f"direction_sensitivity_{agent}")
        plt.close()

    if USE_LATEX:
        plt.rcParams["text.usetex"] = False

    # Also produce a rectangular bar chart for easier reading in papers.
    _plot_bar(curves, condition, output_dir, palette)

    if not no_pgf:
        _export_pgf(curves, condition, output_dir)


def _plot_bar(
    curves: dict[str, dict],
    condition: str,
    output_dir: Path,
    palette,
) -> None:
    """Bar chart version: degradation % per angle, one bar per angle."""
    for (agent, data), color in zip(curves.items(), palette):
        angles_deg = np.array(data["angles"])
        means = np.array(data["means"])
        cis = np.array(data["cis"])

        is_baseline = agent in BASELINE_AGENTS
        agent_label = (
            AGENT_OR_BASE_DICT.get("Baseline", agent)
            if is_baseline
            else AGENT_OR_BASE_DICT.get("Agent", agent)
        )

        fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
        sns.set_theme(style="whitegrid", context="paper")
        ax.bar(
            angles_deg, means,
            yerr=cis,
            color=color,
            alpha=0.8,
            capsize=4,
            width=30,
            label=f"{_DISPLAY_LABELS.get(condition, condition)} ({agent_label})",
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_xticks(angles_deg)
        ax.set_xticklabels([f"{a}°" for a in angles_deg], fontsize=8)
        ax.set_xlabel("Offset angle (degrees)", fontsize=9)
        ax.set_ylabel(r"$\Delta\tau\,[\%]$", fontsize=9)
        ax.legend(fontsize=8)

        _save(output_dir, f"direction_bar_{agent}")
        plt.close()


def _export_pgf(
    curves: dict[str, dict],
    condition: str,
    output_dir: Path,
) -> None:
    import re as _re

    def _san(s: str) -> str:
        return _re.sub(r"[^A-Za-z0-9]", "", s)

    pgf_dir = output_dir / "pgfplots"
    pgf_dir.mkdir(parents=True, exist_ok=True)

    for agent, data in curves.items():
        csv_name = f"direction_sensitivity_{agent}.csv"
        df = pd.DataFrame({
            "angle": data["angles"],
            "mean": data["means"],
            "up": [m + c for m, c in zip(data["means"], data["cis"])],
            "low": [m - c for m, c in zip(data["means"], data["cis"])],
        })
        df.to_csv(pgf_dir / csv_name, index=False)

        lines = [
            f"% Direction sensitivity — {condition} — {agent}",
            r"% Required preamble: \usepackage{pgfplots,pgfplotstable}",
            r"%   \usepgfplotslibrary{fillbetween}  \pgfplotsset{compat=1.18}",
            "",
            r"\begin{tikzpicture}",
            r"    \begin{axis}[",
            r"        grid=major, grid style={dashed, gray!30},",
            r"        width=\linewidth,",
            r"        xlabel={\small offset angle (degrees)},",
            r"        ylabel={\small $\Delta\tau\,[\%]$},",
            r"        xtick=data,",
            r"        tick label style={font=\small}, label style={font=\small},",
            r"    ]",
            f"    \\addplot [name path=up, draw=none, forget plot] table [x=angle, y=up, col sep=comma] {{{csv_name}}};",
            f"    \\addplot [name path=lo, draw=none, forget plot] table [x=angle, y=low, col sep=comma] {{{csv_name}}};",
            "    \\addplot [blue!30, opacity=0.3, forget plot] fill between [of=up and lo];",
            f"    \\addplot [blue, thick, mark=*, mark size=2pt] table [x=angle, y=mean, col sep=comma] {{{csv_name}}};",
            f"    \\addlegendentry{{{_DISPLAY_LABELS.get(condition, condition)}}}",
            r"    \end{axis}",
            r"\end{tikzpicture}",
        ]
        (pgf_dir / f"direction_sensitivity_{agent}.tex").write_text("\n".join(lines))
        print(f"  PGFPlots: direction_sensitivity_{agent}.csv / .tex")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Polar plot of obs_user_dir degradation vs. offset angle."
    )
    parser.add_argument(
        "--base", default="out/evaluations",
        help="Root directory containing per-experiment subdirectories.",
    )
    parser.add_argument(
        "--names", nargs="+", required=True,
        help="Ordered list of experiment names, one per angle.",
    )
    parser.add_argument(
        "--angles", nargs="+", type=float, required=True,
        help="Matching ordered list of offset angles in degrees.",
    )
    parser.add_argument(
        "--condition", default="obs_user_dir",
        help="Attack condition subdirectory to read (default: obs_user_dir).",
    )
    parser.add_argument(
        "--baseline-idx", type=int, default=0,
        help="Index into --names used as throughput baseline (default: 0).",
    )
    parser.add_argument(
        "--output", default="out/visualizations/angle_sweep",
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--no-pgf", action="store_true",
        help="Skip PGFPlots CSV/tex export.",
    )
    args = parser.parse_args()

    if len(args.names) != len(args.angles):
        parser.error("--names and --angles must have the same number of entries.")

    output_dir = Path(args.output)
    base_dir = Path(args.base)

    print(f"\nAngle sweep — {len(args.names)} experiments")
    print(f"Angles: {args.angles}\n")

    curves = build_angle_curves(
        base_dir=base_dir,
        names=args.names,
        angles=args.angles,
        condition=args.condition,
        baseline_idx=args.baseline_idx,
    )

    if not curves:
        print("No usable data found.")
        raise SystemExit(1)

    plot_polar(curves, args.condition, output_dir, args.no_pgf)
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
