"""
Plot throughput degradation vs. perturbation magnitude — the harm potential curve.

For each attack type, reads evaluation results from multiple experiments (one per
magnitude level) and plots degradation (%) against magnitude on a single set of
axes. PADRE and GTM baselines are overlaid using different line styles.

Also produces a throughput-over-steps plot (identical in style to the one in
compare_byzantine.py) for a reference magnitude, showing the absolute bit rate of
each attack type and the no-Byzantine baseline on the same axes.

This is the primary research plot for Phase 1 of the observation-manipulation
attack analysis.

Usage (run from the resilient-backbone/ root):
    python src/Utils/visualize/byzantine/compare_magnitude_sweep.py \\
        --base    out/evaluations \\
        --names   sweep_m0.0 sweep_m0.5 sweep_m1.0 sweep_m2.0 sweep_m4.0 \\
        --magnitudes 0.0 0.5 1.0 2.0 4.0

Optional arguments:
    --base          Root directory containing per-experiment subdirectories.
                    Default: out/evaluations
    --names         Ordered list of experiment directory names (one per magnitude).
    --magnitudes    Matching ordered list of magnitude float values.
    --baseline-idx  Index into --names of the no-attack experiment used as the
                    denominator for degradation. Default: 0 (first entry).
    --ref-idx       Index into --names of the experiment used for the steps plot.
                    Default: -1 (last / highest magnitude).
    --output        Output directory for plots. Default: out/visualizations/magnitude_sweep
    --no-pgf        Skip PGFPlots export.

The no_byzantine condition inside each experiment is used to verify that the
baseline throughput is consistent across magnitude levels.

Attack conditions are discovered automatically from subdirectories — any new
attack type written by run_byzantine_sweep.py will appear in the plot.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Constants (mirror compare_byzantine.py)
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
    "obs_interference_amplify":   "Interference Amplify (false alarm)",
    "obs_noise":                  "Obs. Noise",
    "obs_all":                    "All Obs. Attacks",
    "obs_harmful":                "Dir + Interference Lie",
    "obs_capacity_deflate":       "Capacity Deflate",
    "obs_demand_deflate":         "Demand Deflate",
    "obs_replay":                 "Obs. Replay",
    "byzantine_greyhole":         "Greyhole",
    "byzantine_sinkhole":         "Sinkhole",
    "byzantine_selective_jamming":"Selective Jam.",
    "byzantine_position_spoofing":"Pos. Spoofing",
    "byzantine_all_attacks":      "All Attacks",
}

# IEEE single-column width.
_FIG_W = 3.5
_FIG_H = 2.6
_FIG_H_LEGEND = _FIG_H + 0.3
_DPI = 450
USE_LATEX = False

# Keep every Nth step in the PGFPlots CSV to avoid TeX list-capacity overflow
# when compiling with pdflatex.  200 points per curve is more than enough for
# smooth rendering.  Set to 1 to disable downsampling (requires lualatex).
_PGFPLOTS_DOWNSAMPLE = 10


# ---------------------------------------------------------------------------
# Styling helpers (mirror compare_byzantine.py)
# ---------------------------------------------------------------------------

def _set_theme() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    sns.set_palette("Set1")


def _apply_scientific_notation(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.yaxis.get_offset_text().set_fontsize(10)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


def _save(output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{name}.png", dpi=_DPI, bbox_inches="tight")
    plt.savefig(output_dir / f"{name}.svg", bbox_inches="tight")
    print(f"  Saved: {name}.png / .svg")


# ---------------------------------------------------------------------------
# Data loading
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
    """Mean throughput in the last STABILIZATION_STEPS steps, one value per episode."""
    tp = df[df["value_name"] == "throughput"]
    max_step = tp["ep_step"].max()
    late = tp[tp["ep_step"] > max_step - STABILIZATION_STEPS]
    return late.groupby("episode")["value"].mean().values


def _deg_mean_ci(
    base_mean: float, attack_episodes: np.ndarray
) -> tuple[float, float]:
    """Return (mean degradation %, 95% CI half-width) across episodes."""
    if base_mean == 0 or np.isnan(base_mean):
        return 0.0, 0.0
    per_ep = (base_mean - attack_episodes) / base_mean * 100
    mean = float(per_ep.mean())
    ci = float(1.96 * per_ep.std() / np.sqrt(len(per_ep))) if len(per_ep) > 1 else 0.0
    return mean, ci


# ---------------------------------------------------------------------------
# Build the harm curve data
# ---------------------------------------------------------------------------

def build_harm_curves(
    base_dir: Path,
    names: list[str],
    magnitudes: list[float],
    baseline_idx: int,
) -> dict[str, dict[str, list]]:
    """
    Returns {agent: {condition_label: {"magnitudes": [...], "means": [...], "cis": [...]}}}
    """
    # Load all experiments.
    experiments: list[dict[str, dict[str, pd.DataFrame]]] = []
    for name in names:
        print(f"  Loading {name} …")
        experiments.append(_load_agent_dfs(base_dir / name))

    # Discover all agents and attack conditions (excluding the no-attack baseline).
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
            mags, means, cis = [], [], []
            for mag, exp in zip(magnitudes, experiments):
                if cond not in exp or agent not in exp[cond]:
                    continue
                ep_vals = _stabilized_per_episode(exp[cond][agent])
                if len(ep_vals) == 0:
                    continue
                mean, ci = _deg_mean_ci(base_mean, ep_vals)
                mags.append(mag)
                means.append(mean)
                cis.append(ci)
            if mags:
                curves[agent][cond] = {"magnitudes": mags, "means": means, "cis": cis}

    return curves


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_harm_curves(
    curves: dict[str, dict[str, dict]],
    output_dir: Path,
    no_pgf: bool,
) -> None:
    """One plot per agent: degradation (%) vs magnitude, one line per attack type."""
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
            mags = np.array(data["magnitudes"])
            means = np.array(data["means"])
            cis = np.array(data["cis"])
            color = color_map[cond]
            mark = marks[i % len(marks)]

            ax.plot(
                mags, means,
                linestyle=linestyle,
                linewidth=2,
                color=color,
                marker=mark,
                markersize=4,
                label=f"{cond} ({agent_label})",
            )
            ax.fill_between(mags, means - cis, means + cis, color=color, alpha=0.15)

        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        plt.xlabel(r"magnitude")
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

        _save(output_dir, f"harm_curve_{agent}")
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
    """Write harm_curve_{agent}.csv and harm_curve_{agent}.tex for LaTeX."""
    import re as _re

    def _san(s: str) -> str:
        return _re.sub(r"[^A-Za-z0-9]", "", s)

    pgf_dir = output_dir / "pgfplots"
    pgf_dir.mkdir(parents=True, exist_ok=True)

    csv_name = f"harm_curve_{agent}.csv"
    # Align all conditions on a common x axis (union of all magnitudes).
    all_x = sorted({m for data in cond_curves.values() for m in data["magnitudes"]})
    csv_rows: dict = {"x": all_x}
    for cond, data in cond_curves.items():
        key = _san(cond)
        lookup_mean = dict(zip(data["magnitudes"], data["means"]))
        lookup_ci   = dict(zip(data["magnitudes"], data["cis"]))
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
        r"% ============================================================",
        r"% Required preamble: \usepackage{pgfplots,pgfplotstable,siunitx}",
        r"%   \usepgfplotslibrary{fillbetween}  \pgfplotsset{compat=1.18}",
        r"% ============================================================",
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
        r"    mean/.style={thick, mark size=1.5pt},",
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
        r"        xlabel={\small magnitude},",
        r"        ylabel={\small $\Delta\tau\,[\%]$},",
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
    (pgf_dir / f"harm_curve_{agent}.tex").write_text("\n".join(lines))
    print(f"  PGFPlots: harm_curve_{agent}.csv / .tex")


# ---------------------------------------------------------------------------
# Steps plot — throughput vs episode step at a reference magnitude
# (mirrors compare_byzantine.py Plot 1)
# ---------------------------------------------------------------------------

def _step_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with columns mean, up, low indexed by ep_step."""
    tp = df[df["value_name"] == "throughput"]
    grouped = tp.groupby("ep_step")["value"]
    mean = grouped.mean()
    ci = 1.96 * grouped.std() / np.sqrt(grouped.count())
    return pd.DataFrame({"mean": mean, "up": mean + ci, "low": mean - ci})


def plot_steps(
    exp_data: dict[str, dict[str, pd.DataFrame]],
    magnitude: float,
    output_dir: Path,
    no_pgf: bool,
) -> None:
    """Throughput vs episode step for all conditions at a single reference magnitude.

    All attack types and the no-Byzantine baseline are shown on the same axes so
    the absolute bit rate difference is immediately visible.
    """
    conditions_present = list(exp_data.keys())
    if not conditions_present:
        print("  [skip steps plot] no data loaded for reference magnitude.")
        return

    all_agents = sorted({a for cond in exp_data.values() for a in cond})
    rl_agents = [a for a in all_agents if a not in BASELINE_AGENTS]
    baseline_agents_present = [a for a in all_agents if a in BASELINE_AGENTS]

    palette = sns.color_palette("Set1", len(conditions_present))
    color_map = dict(zip(conditions_present, palette))

    if USE_LATEX:
        plt.rcParams["text.usetex"] = True
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath, amssymb}"

    for agent in rl_agents:
        n_cond = len(conditions_present)
        fig_w = max(_FIG_W * 2, n_cond * 2.5)
        plt.figure(figsize=(fig_w, _FIG_H_LEGEND), constrained_layout=True)
        _set_theme()
        ax = plt.gca()

        # Build long-format DataFrame for this agent across all conditions.
        frames = []
        for label in conditions_present:
            if agent not in exp_data.get(label, {}):
                continue
            df = exp_data[label][agent]
            tp = df[df["value_name"] == "throughput"].copy()
            tp = tp.assign(condition=label)
            frames.append(tp)

        df_rl = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not df_rl.empty:
            sns.lineplot(
                data=df_rl,
                x="ep_step",
                y="value",
                hue="condition",
                errorbar=("ci", 95),
                markers=True,
                dashes=False,
                markevery=200,
                linewidth=2,
                palette=color_map,
                ax=ax,
            )

        # Baseline agents — dashed lines, same colour palette.
        for bname in baseline_agents_present:
            bframes = []
            for label in conditions_present:
                if bname not in exp_data.get(label, {}):
                    continue
                df = exp_data[label][bname]
                tp = df[df["value_name"] == "throughput"].copy()
                tp = tp.assign(condition=label)
                bframes.append(tp)
            if not bframes:
                continue
            df_b = pd.concat(bframes, ignore_index=True)
            for cond, grp in df_b.groupby("condition"):
                color = color_map.get(cond, "grey")
                step_mean = grp.groupby("ep_step")["value"].mean()
                ax.plot(
                    step_mean.index,
                    step_mean.values,
                    linestyle="--",
                    linewidth=1.5,
                    color=color,
                    alpha=0.8,
                    label=f"{cond} ({AGENT_OR_BASE_DICT.get('Baseline', bname)})",
                )

        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.xaxis.get_offset_text().set_fontsize(10)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.yaxis.get_offset_text().set_fontsize(10)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.xlabel(r"$t$")
        plt.ylabel(r"$\tau \, \text{[Mbit/s]}$")
        ax.set_title(f"magnitude = {magnitude:.3g}", fontsize=8)

        handles, labels_leg = ax.get_legend_handles_labels()
        plt.legend(
            handles, labels_leg,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            labelspacing=0.5,
            columnspacing=0.5,
        )

        tag = f"{magnitude:.3g}".replace(".", "_")
        _save(output_dir, f"steps_{agent}_mag{tag}")
        plt.close()

        if not no_pgf:
            _export_steps_pgf(exp_data, agent, magnitude, output_dir)

    if USE_LATEX:
        plt.rcParams["text.usetex"] = False
        plt.rcParams["mathtext.default"] = "regular"


def _export_steps_pgf(
    exp_data: dict[str, dict[str, pd.DataFrame]],
    agent: str,
    magnitude: float,
    output_dir: Path,
) -> None:
    """Write steps_{agent}_mag{m}.csv and .tex for the steps plot."""
    import re as _re

    def _san(s: str) -> str:
        return _re.sub(r"[^A-Za-z0-9]", "", s)

    conditions_present = list(exp_data.keys())
    series: dict[str, pd.DataFrame] = {}
    for label in conditions_present:
        if agent in exp_data.get(label, {}):
            series[label] = _step_stats(exp_data[label][agent])

    if not series:
        return

    tag = f"{magnitude:.3g}".replace(".", "_")
    pgf_dir = output_dir / "pgfplots"
    pgf_dir.mkdir(parents=True, exist_ok=True)
    csv_name = f"steps_{agent}_mag{tag}.csv"

    all_steps = sorted({s for df in series.values() for s in df.index})
    last_step = all_steps[-1] if all_steps else None
    if _PGFPLOTS_DOWNSAMPLE > 1:
        all_steps = [s for i, s in enumerate(all_steps) if i % _PGFPLOTS_DOWNSAMPLE == 0]
        if last_step is not None and last_step not in all_steps:
            all_steps.append(last_step)
    csv_rows: dict = {"x": all_steps}
    for label, df in series.items():
        key = _san(label)
        df_r = df.reindex(all_steps)
        csv_rows[f"mean{key}"] = df_r["mean"].values
        csv_rows[f"up{key}"]   = df_r["up"].values
        csv_rows[f"low{key}"]  = df_r["low"].values
    pd.DataFrame(csv_rows).to_csv(pgf_dir / csv_name, index=False)

    palette = sns.color_palette("Set1", len(conditions_present))
    cond_colors = {
        label: "#{:02X}{:02X}{:02X}".format(int(r*255), int(g*255), int(b*255))
        for label, (r, g, b) in zip(conditions_present, palette)
    }
    marks = ["*", "square*", "triangle*", "diamond*", "pentagon*"]

    lines = [
        r"% Steps plot — magnitude " + str(magnitude),
        r"% Required preamble: \usepackage{pgfplots,pgfplotstable,siunitx}",
        r"%   \usepgfplotslibrary{fillbetween}  \pgfplotsset{compat=1.18}",
        "",
    ]
    for label in series:
        key = _san(label)
        hex_col = cond_colors.get(label, "#000000")
        lines.append(f"\\definecolor{{color{key}}}{{HTML}}{{{hex_col[1:]}}}")
    lines.append("")

    lines += [r"\pgfplotsset{",
              r"    ci path/.style={draw=none}, ci fill/.style={opacity=0.15},",
              r"    mean/.style={thick, mark repeat=200, mark size=1.5pt},"]
    for i, label in enumerate(series):
        key = _san(label)
        mark = marks[i % len(marks)]
        lines.append(f"    {key}/.style={{color=color{key}, mark={mark}}},")
    lines += [r"}", ""]

    lines += [r"\begin{tikzpicture}", r"    \begin{axis}[",
              r"        grid=major, grid style={dashed, gray!30},",
              r"        width=\linewidth,",
              r"        xlabel={\small $t$}, ylabel={\small $\tau$ [\si{\mega\bit\per\second}]},",
              r"        legend style={at={(0.5,-0.20)}, anchor=north, legend columns=3},",
              r"        tick label style={font=\small}, label style={font=\small},",
              r"    ]", ""]
    for label in series:
        key = _san(label)
        lines += [
            f"    % ===== {label} =====",
            f"    \\addplot [name path={key}up, ci path, forget plot] table [x=x, y=up{key}, col sep=comma] {{{csv_name}}};",
            f"    \\addplot [name path={key}lo, ci path, forget plot] table [x=x, y=low{key}, col sep=comma] {{{csv_name}}};",
            f"    \\addplot [{key}, ci fill, forget plot] fill between [of={key}up and {key}lo];",
            f"    \\addplot [{key}, mean] table [x=x, y=mean{key}, col sep=comma] {{{csv_name}}};",
            f"    \\addlegendentry{{{label}}}", "",
        ]
    lines += [r"    \end{axis}", r"\end{tikzpicture}"]
    (pgf_dir / f"steps_{agent}_mag{tag}.tex").write_text("\n".join(lines))
    print(f"  PGFPlots: steps_{agent}_mag{tag}.csv / .tex")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot throughput degradation vs. perturbation magnitude."
    )
    parser.add_argument(
        "--base", default="out/evaluations",
        help="Root directory containing per-experiment subdirectories.",
    )
    parser.add_argument(
        "--names", nargs="+", required=True,
        help="Ordered list of experiment directory names, one per magnitude.",
    )
    parser.add_argument(
        "--magnitudes", nargs="+", type=float, required=True,
        help="Matching ordered list of magnitude float values.",
    )
    parser.add_argument(
        "--baseline-idx", type=int, default=0,
        help="Index into --names of the no-attack experiment (default: 0).",
    )
    parser.add_argument(
        "--output", default="out/visualizations/magnitude_sweep",
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--ref-idx", type=int, default=-1,
        help=(
            "Index into --names of the experiment used for the steps plot "
            "(default: -1, last / highest magnitude)."
        ),
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

    print(f"\nMagnitude sweep — {len(args.names)} experiments")
    print(f"Magnitudes: {args.magnitudes}\n")

    curves = build_harm_curves(
        base_dir=base_dir,
        names=args.names,
        magnitudes=args.magnitudes,
        baseline_idx=args.baseline_idx,
    )

    if not curves:
        print("No usable data found. Run run_byzantine_sweep.py --magnitude first.")
        raise SystemExit(1)

    plot_harm_curves(curves, output_dir, args.no_pgf)

    # Steps plot at the reference magnitude.
    ref_idx = args.ref_idx % len(args.names)
    ref_name = args.names[ref_idx]
    ref_magnitude = args.magnitudes[ref_idx]
    print(f"\nGenerating steps plot for reference magnitude {ref_magnitude:.3g} ({ref_name}) …")
    ref_data = _load_agent_dfs(base_dir / ref_name)
    plot_steps(ref_data, ref_magnitude, output_dir, args.no_pgf)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
