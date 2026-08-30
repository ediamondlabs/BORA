"""
Plot throughput degradation vs. number of Byzantine nodes.

For a fixed attack magnitude (default: highest tested), reads one evaluation
experiment per N_byz value and plots degradation (%) against N_byz.

Usage (run from the resilient-backbone/ root):
    python src/Utils/visualize/byzantine/compare_nbyz_sweep.py \\
        --base    out/evaluations \\
        --names   sweep_nbyz1 sweep_nbyz2 sweep_nbyz3 sweep_nbyz4 \\
        --nbyz-values 1 2 3 4

Optional arguments:
    --base          Root directory containing per-experiment subdirectories.
                    Default: out/evaluations
    --names         Ordered list of experiment directory names (one per N_byz).
    --nbyz-values   Matching ordered list of N_byz int values.
    --baseline-idx  Index into --names of the experiment whose no_byzantine condition
                    is used as the throughput denominator. Default: 0 (first entry).
    --output        Output directory for plots.
                    Default: out/visualizations/nbyz_sweep
    --no-pgf        Skip PGFPlots export.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Constants (mirror compare_magnitude_sweep.py)
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
    "obs_harmful":                "Dir + Interference Lie",
    "obs_capacity_deflate":       "Capacity Deflate",
    "obs_demand_deflate":         "Demand Deflate",
    "obs_replay":                 "Obs. Replay",
}

_FIG_W = 3.5
_FIG_H = 2.6
_FIG_H_LEGEND = _FIG_H + 0.3
_DPI = 450
USE_LATEX = False
_PGFPLOTS_DOWNSAMPLE = 10


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

def _set_theme() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    sns.set_palette("Set1")


def _save(output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{name}.png", dpi=_DPI, bbox_inches="tight")
    plt.savefig(output_dir / f"{name}.svg", bbox_inches="tight")
    print(f"  Saved: {name}.png / .svg")


# ---------------------------------------------------------------------------
# Data loading (identical to compare_magnitude_sweep.py)
# ---------------------------------------------------------------------------

def _load_agent_dfs(experiment_dir: Path) -> dict[str, dict[str, pd.DataFrame]]:
    """Return {condition_label: {agent_stem: DataFrame}} for one experiment dir."""
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
# Build degradation-vs-N_byz curves
# ---------------------------------------------------------------------------

def build_nbyz_curves(
    base_dir: Path,
    names: list[str],
    nbyz_values: list[int],
    baseline_idx: int,
) -> dict[str, dict[str, dict]]:
    """Return {agent: {condition_label: {"nbyz": [...], "means": [...], "cis": [...]}}}"""
    experiments: list[dict[str, dict[str, pd.DataFrame]]] = []
    for name in names:
        print(f"  Loading {name} …")
        experiments.append(_load_agent_dfs(base_dir / name))

    baseline_label = _DISPLAY_LABELS.get(_BASELINE_SUBDIR, _BASELINE_SUBDIR)
    all_agents = sorted({
        agent
        for exp in experiments
        for cond_data in exp.values()
        for agent in cond_data
    })
    attack_conditions = sorted({
        cond
        for exp in experiments
        for cond in exp
        if cond != baseline_label
    })

    baseline_exp = experiments[baseline_idx]
    baseline_cond_data = baseline_exp.get(baseline_label, {})

    curves: dict[str, dict[str, dict]] = {}
    for agent in all_agents:
        if agent not in baseline_cond_data:
            continue
        base_mean = _stabilized_per_episode(baseline_cond_data[agent]).mean()
        if base_mean == 0 or np.isnan(base_mean):
            continue

        curves[agent] = {}
        for cond in attack_conditions:
            nbyz_list, means, cis = [], [], []
            for n, exp in zip(nbyz_values, experiments):
                if cond not in exp or agent not in exp[cond]:
                    continue
                ep_vals = _stabilized_per_episode(exp[cond][agent])
                if len(ep_vals) == 0:
                    continue
                mean, ci = _deg_mean_ci(base_mean, ep_vals)
                nbyz_list.append(n)
                means.append(mean)
                cis.append(ci)
            if nbyz_list:
                curves[agent][cond] = {"nbyz": nbyz_list, "means": means, "cis": cis}

    return curves


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_nbyz_curves(
    curves: dict[str, dict[str, dict]],
    output_dir: Path,
    no_pgf: bool,
) -> None:
    """One plot per agent: degradation (%) vs N_byz, one line per attack type."""
    if USE_LATEX:
        plt.rcParams["text.usetex"] = True
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath, amssymb}"

    for agent, cond_curves in curves.items():
        if not cond_curves:
            continue

        attack_labels = list(cond_curves.keys())
        palette = sns.color_palette("Set1", len(attack_labels))
        color_map = dict(zip(attack_labels, palette))
        marks = ["o", "s", "^", "D", "P", "X"]

        n_cond = len(attack_labels)
        fig_w = max(_FIG_W * 2, n_cond * 1.5)
        plt.figure(figsize=(fig_w, _FIG_H_LEGEND), constrained_layout=True)
        _set_theme()
        ax = plt.gca()

        is_baseline = agent in BASELINE_AGENTS
        agent_label = (
            AGENT_OR_BASE_DICT.get("Baseline", agent)
            if is_baseline
            else AGENT_OR_BASE_DICT.get("Agent", agent)
        )
        linestyle = "--" if is_baseline else "-"

        for i, (cond, data) in enumerate(cond_curves.items()):
            nbyz = np.array(data["nbyz"])
            means = np.array(data["means"])
            cis = np.array(data["cis"])
            color = color_map[cond]
            mark = marks[i % len(marks)]

            ax.plot(
                nbyz, means,
                linestyle=linestyle,
                linewidth=2,
                color=color,
                marker=mark,
                markersize=5,
                label=f"{cond} ({agent_label})",
            )
            ax.fill_between(nbyz, means - cis, means + cis, color=color, alpha=0.15)

        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.xlabel(r"$N_{\mathrm{byz}}$")
        plt.ylabel(r"$\Delta\tau \, [\%]$")
        ax.title.set_visible(False)

        handles, labels = ax.get_legend_handles_labels()
        plt.legend(
            handles, labels,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            labelspacing=0.5,
            columnspacing=0.5,
        )

        _save(output_dir, f"nbyz_curve_{agent}")
        plt.close()

        if not no_pgf:
            _export_pgf(cond_curves, agent, output_dir, color_map, marks)

    if USE_LATEX:
        plt.rcParams["text.usetex"] = False
        plt.rcParams["mathtext.default"] = "regular"


def _export_pgf(
    cond_curves: dict[str, dict],
    agent: str,
    output_dir: Path,
    color_map: dict,
    marks: list[str],
) -> None:
    import re as _re

    def _san(s: str) -> str:
        return _re.sub(r"[^A-Za-z0-9]", "", s)

    pgf_dir = output_dir / "pgfplots"
    pgf_dir.mkdir(parents=True, exist_ok=True)

    csv_name = f"nbyz_curve_{agent}.csv"
    all_x = sorted({n for data in cond_curves.values() for n in data["nbyz"]})
    csv_rows: dict = {"x": all_x}
    for cond, data in cond_curves.items():
        key = _san(cond)
        lookup_mean = dict(zip(data["nbyz"], data["means"]))
        lookup_ci   = dict(zip(data["nbyz"], data["cis"]))
        csv_rows[f"mean{key}"] = [lookup_mean.get(x, float("nan")) for x in all_x]
        csv_rows[f"up{key}"]   = [lookup_mean.get(x, float("nan")) + lookup_ci.get(x, 0) for x in all_x]
        csv_rows[f"low{key}"]  = [lookup_mean.get(x, float("nan")) - lookup_ci.get(x, 0) for x in all_x]

    pd.DataFrame(csv_rows).to_csv(pgf_dir / csv_name, index=False)

    palette = sns.color_palette("Set1", len(cond_curves))
    cond_colors = {
        label: "#{:02X}{:02X}{:02X}".format(int(r*255), int(g*255), int(b*255))
        for label, (r, g, b) in zip(cond_curves, palette)
    }
    mark_shapes = ["*", "square*", "triangle*", "diamond*", "pentagon*"]

    lines = [
        r"% N_byz scaling curve",
        r"% Required preamble: \usepackage{pgfplots,pgfplotstable,siunitx}",
        r"%   \usepgfplotslibrary{fillbetween}  \pgfplotsset{compat=1.18}",
        "",
    ]
    for label in cond_curves:
        key = _san(label)
        hex_col = cond_colors.get(label, "#000000")
        lines.append(f"\\definecolor{{color{key}}}{{HTML}}{{{hex_col[1:]}}}")
    lines.append("")

    lines += [
        r"\pgfplotsset{",
        r"    ci path/.style={draw=none},",
        r"    ci fill/.style={opacity=0.15},",
        r"    mean/.style={thick, mark size=2pt},",
    ]
    for i, label in enumerate(cond_curves):
        key = _san(label)
        mark = mark_shapes[i % len(mark_shapes)]
        lines.append(f"    {key}/.style={{color=color{key}, mark={mark}}},")
    lines += [r"}", ""]

    lines += [
        r"\begin{tikzpicture}",
        r"    \begin{axis}[",
        r"        grid=major, grid style={dashed, gray!30},",
        r"        width=\linewidth,",
        r"        xlabel={\small $N_{\mathrm{byz}}$},",
        r"        ylabel={\small $\Delta\tau\,[\%]$},",
        r"        xtick=data,",
        r"        legend style={at={(0.5,-0.20)}, anchor=north, legend columns=3},",
        r"        tick label style={font=\small}, label style={font=\small},",
        r"    ]",
        "",
    ]
    for label in cond_curves:
        key = _san(label)
        lines += [
            f"    % ===== {label} =====",
            f"    \\addplot [name path={key}up, ci path, forget plot] table [x=x, y=up{key}, col sep=comma] {{{csv_name}}};",
            f"    \\addplot [name path={key}lo, ci path, forget plot] table [x=x, y=low{key}, col sep=comma] {{{csv_name}}};",
            f"    \\addplot [{key}, ci fill, forget plot] fill between [of={key}up and {key}lo];",
            f"    \\addplot [{key}, mean] table [x=x, y=mean{key}, col sep=comma] {{{csv_name}}};",
            f"    \\addlegendentry{{{label}}}",
            "",
        ]
    lines += [r"    \end{axis}", r"\end{tikzpicture}"]
    (pgf_dir / f"nbyz_curve_{agent}.tex").write_text("\n".join(lines))
    print(f"  PGFPlots: nbyz_curve_{agent}.csv / .tex")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot throughput degradation vs. number of Byzantine nodes."
    )
    parser.add_argument(
        "--base", default="out/evaluations",
        help="Root directory containing per-experiment subdirectories.",
    )
    parser.add_argument(
        "--names", nargs="+", required=True,
        help="Ordered list of experiment names, one per N_byz value.",
    )
    parser.add_argument(
        "--nbyz-values", nargs="+", type=int, required=True,
        help="Matching ordered list of N_byz integer values.",
    )
    parser.add_argument(
        "--baseline-idx", type=int, default=0,
        help="Index into --names of the experiment used as throughput baseline (default: 0).",
    )
    parser.add_argument(
        "--output", default="out/visualizations/nbyz_sweep",
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--no-pgf", action="store_true",
        help="Skip PGFPlots CSV/tex export.",
    )
    args = parser.parse_args()

    if len(args.names) != len(args.nbyz_values):
        parser.error("--names and --nbyz-values must have the same number of entries.")

    output_dir = Path(args.output)
    base_dir = Path(args.base)

    print(f"\nN_byz scaling sweep — {len(args.names)} experiments")
    print(f"N_byz values: {args.nbyz_values}\n")

    curves = build_nbyz_curves(
        base_dir=base_dir,
        names=args.names,
        nbyz_values=args.nbyz_values,
        baseline_idx=args.baseline_idx,
    )

    if not curves:
        print("No usable data found. Run run_byzantine_sweep.py --n-byz first.")
        raise SystemExit(1)

    plot_nbyz_curves(curves, output_dir, args.no_pgf)
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
