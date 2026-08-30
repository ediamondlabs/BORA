"""
Compare evaluation results across Byzantine attack conditions.

Loads CSVs from each condition's output directory and produces:
  1. Mean throughput over episode steps (one curve per condition, per RL agent)
     Baseline agents (e.g. GreedyMaximizer) are overlaid with a different linestyle.
  2. Throughput degradation (%) relative to the no-Byzantine baseline,
     with 95% confidence interval error bars.
  3. PGFPlots-ready CSV + .tex files under OUTPUT_DIR/pgfplots/.

Conditions are discovered automatically from subdirectories of the experiment
output directory — no manual editing required when new attack types are added.

Can be run standalone or called automatically by run_byzantine_sweep.py.

Run from the resilient-backbone/ root:
    python src/Utils/visualize/byzantine/compare_byzantine.py

To target a different experiment without editing this file, set the environment
variable PADRE_EXPERIMENT_NAME before running:
    PADRE_EXPERIMENT_NAME=25_03 python src/Utils/visualize/byzantine/compare_byzantine.py

Or use run_byzantine_sweep.py which sets this automatically:
    python src/Utils/run_byzantine_sweep.py --model "out/models/run/**" --name 25_03
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Change this to switch experiments — drives both input paths and output dir.
# Can also be overridden via the PADRE_EXPERIMENT_NAME environment variable,
# which run_byzantine_sweep.py sets automatically.
EXPERIMENT_NAME = os.environ.get("PADRE_EXPERIMENT_NAME", "19_03")

_BASE = f"out/evaluations/{EXPERIMENT_NAME}"

# Human-readable labels for known attack subdirectory names.
# Any subdirectory not listed here is displayed using its directory name.
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
    "learned_obs_user_dir":       "Learned Dir Lie",
    # Legacy routing-layer entries.
    "byzantine_greyhole":         "Greyhole",
    "byzantine_sinkhole":         "Sinkhole",
    "byzantine_selective_jamming":"Selective Jam.",
    "byzantine_position_spoofing":"Pos. Spoofing",
    "byzantine_all_attacks":      "All Attacks",
}

# Conditions discovered dynamically — all subdirs of the base directory.
# Keys are display labels; values are directory paths.
CONDITIONS: dict[str, str] = {
    _DISPLAY_LABELS.get(p.name, p.name): str(p)
    for p in sorted(Path(_BASE).iterdir())
    if p.is_dir()
} if Path(_BASE).exists() else {}

# Display label of the no-attack baseline condition.
_BASELINE_LABEL = _DISPLAY_LABELS.get("no_byzantine", "No Byzantine")

# Filename stems treated as baselines (plotted with dashed lines / hatching).
BASELINE_AGENTS: list[str] = ["Gredy_Trffc"]

# Steps at the end of an episode considered "stabilized".
STABILIZATION_STEPS = 500

# CSVs store throughput in Mbit/s (ThroughputInMbits logs getThroughput()/1e6).
THROUGHPUT_SCALE = 1.0

# Keep every Nth step in the PGFPlots CSV to avoid TeX list-capacity overflow
# when compiling with pdflatex.  200 points per curve is more than enough for
# smooth rendering.  Set to 1 to disable downsampling (requires lualatex).
_PGFPLOTS_DOWNSAMPLE = 10

# Label mapping matching visualize_MANETAgents.py convention.
AGENT_OR_BASE_DICT: dict = {"Agent": "PADRE", "Baseline": "GTM"}

OUTPUT_DIR = Path(f"out/visualizations/byzantine_comparison/{EXPERIMENT_NAME}")

# ---------------------------------------------------------------------------
# Styling helpers (mirror visualize_MANETAgents.py)
# ---------------------------------------------------------------------------

# IEEE single-column width.
_FIG_W = 3.5
_FIG_H = 2.6
_FIG_H_LEGEND = _FIG_H + 0.3  # extra room for below-axis legend
_DPI = 450

# Set to True only when a system LaTeX installation is available.
# If False, matplotlib's built-in mathtext is used (no external process).
USE_LATEX = False


def _set_theme() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    sns.set_palette("Set1")


def _init_figure(with_legend_space_below: bool = False) -> None:
    h = _FIG_H_LEGEND if with_legend_space_below else _FIG_H
    plt.figure(figsize=(_FIG_W, h), constrained_layout=True)


def _apply_scientific_notation(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.xaxis.get_offset_text().set_fontsize(10)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.yaxis.get_offset_text().set_fontsize(10)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


def _save(name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / f"{name}.png", dpi=_DPI, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"{name}.svg", bbox_inches="tight")
    print(f"  Saved: {name}.png / .svg")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_condition(directory: str) -> dict[str, pd.DataFrame]:
    """Return {agent_stem: DataFrame} for all CSVs found under *directory*."""
    root = Path(directory)
    if not root.exists():
        print(f"  [skip] directory not found: {root}")
        return {}
    results = {}
    for csv_path in sorted(root.rglob("*.csv")):
        df = pd.read_csv(csv_path)
        if not {"ep_step", "episode", "value_name", "value"}.issubset(df.columns):
            continue
        # Scale throughput in-place (matches LogDataProcessor behaviour).
        mask = df["value_name"] == "throughput"
        df.loc[mask, "value"] *= THROUGHPUT_SCALE
        results[csv_path.stem] = df
    return results


print("Loading evaluation data …")
all_data: dict[str, dict[str, pd.DataFrame]] = {}
for label, directory in CONDITIONS.items():
    data = load_condition(directory)
    if data:
        all_data[label] = data
        print(f"  {label!r}: {list(data.keys())}")

if not all_data:
    print("No data found. Run evaluation jobs first.")
    raise SystemExit(1)

all_agents = sorted({a for cond_data in all_data.values() for a in cond_data})
rl_agents = [a for a in all_agents if a not in BASELINE_AGENTS]
baseline_agents = [a for a in all_agents if a in BASELINE_AGENTS]
conditions_present = list(all_data.keys())

# ---------------------------------------------------------------------------
# Utility: build a long-format DataFrame for one agent across all conditions.
# ---------------------------------------------------------------------------

def _long_df_for_agent(agent: str) -> pd.DataFrame:
    """Throughput rows tagged with 'condition' column (already scaled)."""
    frames = []
    for label in conditions_present:
        if agent not in all_data.get(label, {}):
            continue
        df = all_data[label][agent]
        tp = df[df["value_name"] == "throughput"].copy()
        tp = tp.assign(condition=label)
        frames.append(tp)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _long_df_all_conditions(exclude_baselines: bool = True) -> pd.DataFrame:
    """Throughput rows for ALL agents merged per condition.

    Merging across agent names allows conditions evaluated under different
    naming conventions (e.g. heuristic sweep vs learned eval) to appear on
    the same plot.
    """
    frames = []
    for label in conditions_present:
        for agent, df in all_data.get(label, {}).items():
            if exclude_baselines and agent in BASELINE_AGENTS:
                continue
            tp = df[df["value_name"] == "throughput"].copy()
            tp = tp.assign(condition=label)
            frames.append(tp)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _stabilized_all_agents(label: str) -> np.ndarray:
    """Stabilised per-episode throughput aggregated across all agents in a condition."""
    vals = []
    for agent, df in all_data.get(label, {}).items():
        vals.extend(_stabilized_per_episode(df))
    return np.array(vals)


def _mean_stabilized(df: pd.DataFrame) -> float:
    max_step = df["ep_step"].max()
    return df[df["ep_step"] > max_step - STABILIZATION_STEPS]["value"].mean()


def _stabilized_per_episode(df: pd.DataFrame) -> np.ndarray:
    """Mean throughput in the last STABILIZATION_STEPS steps, one value per episode."""
    tp = df[df["value_name"] == "throughput"]
    max_step = tp["ep_step"].max()
    late = tp[tp["ep_step"] > max_step - STABILIZATION_STEPS]
    return late.groupby("episode")["value"].mean().values


# ---------------------------------------------------------------------------
# Plot 1 – Throughput over episode steps
# ---------------------------------------------------------------------------

if USE_LATEX:
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath, amssymb}"

n_cond = len(conditions_present)
fig_w = max(_FIG_W * 2, n_cond * 2.5)
plt.figure(figsize=(fig_w, _FIG_H_LEGEND), constrained_layout=True)
_set_theme()
ax = plt.gca()

palette = sns.color_palette("Set1", len(conditions_present))
color_map = dict(zip(conditions_present, palette))

df_rl = _long_df_all_conditions(exclude_baselines=True)
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

# Baseline agents – dashed lines, same colour palette.
df_base = _long_df_all_conditions(exclude_baselines=False)
df_base = df_base[df_base.index.isin(
    df_rl.index if not df_rl.empty else []
)] if not df_base.empty else df_base
for bname in baseline_agents:
    df_b = _long_df_for_agent(bname)
    if df_b.empty:
        continue
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

_apply_scientific_notation(ax)
plt.xlabel(r"$t$")
plt.ylabel(r"$\tau \, \text{[Mbit/s]}$")
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

_save("steps_combined")
plt.close()

# ---------------------------------------------------------------------------
# Plot 2 – Throughput degradation (%) relative to no-Byzantine baseline
#           with 95% confidence interval error bars
# ---------------------------------------------------------------------------

if _BASELINE_LABEL in all_data:
    palette = sns.color_palette("Set1", len(conditions_present))
    color_map = dict(zip(conditions_present, palette))

    base_vals = _stabilized_all_agents(_BASELINE_LABEL)
    base_mean = base_vals.mean() if len(base_vals) > 0 else 0.0
    if base_mean == 0 or np.isnan(base_mean):
        print("  [skip degradation] baseline throughput is 0 or NaN")
    else:
        attack_conditions = [c for c in conditions_present if c != _BASELINE_LABEL]
        deg_means, deg_cis, deg_labels = [], [], []
        for c in attack_conditions:
            ep_vals = _stabilized_all_agents(c)
            if len(ep_vals) == 0:
                continue
            per_ep_deg = (base_mean - ep_vals) / base_mean * 100
            deg_means.append(per_ep_deg.mean())
            deg_cis.append(1.96 * per_ep_deg.std() / np.sqrt(len(per_ep_deg)))
            deg_labels.append(c)

        if deg_labels:
            n_bars = len(deg_labels)
            fig_w = max(_FIG_W, n_bars * 1.2)
            plt.figure(figsize=(fig_w, _FIG_H), constrained_layout=True)
            _set_theme()
            ax = plt.gca()

            bar_colors = [color_map[l] for l in deg_labels]
            bars = ax.bar(
                deg_labels, deg_means,
                color=bar_colors,
                edgecolor="black",
                linewidth=0.5,
                yerr=deg_cis,
                capsize=4,
                error_kw={"linewidth": 1.0, "ecolor": "black"},
            )
            ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=7)
            ax.axhline(0, color="black", linewidth=0.8)
            plt.ylabel(r"$\Delta\tau \, [\%]$")
            plt.xticks(rotation=15, ha="right")
            ax.title.set_visible(False)

            _save("degradation_combined")
            plt.close()

if USE_LATEX:
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.default"] = "regular"

# ---------------------------------------------------------------------------
# PGFPlots export – steps plot
# For each RL agent, writes:
#   steps_{agent}.csv   – columns: x, mean{key}, up{key}, low{key}, …
#   steps_{agent}.tex   – ready-to-paste TikZ/pgfplots code
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    """Convert a condition label to a valid pgfplots key (no spaces/special chars)."""
    import re as _re
    return _re.sub(r"[^A-Za-z0-9]", "", name)


def _step_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with columns mean, up, low indexed by ep_step."""
    tp = df[df["value_name"] == "throughput"]
    grouped = tp.groupby("ep_step")["value"]
    mean = grouped.mean()
    ci = 1.96 * grouped.std() / np.sqrt(grouped.count())
    return pd.DataFrame({"mean": mean, "up": mean + ci, "low": mean - ci})


PGFPLOTS_DIR = OUTPUT_DIR / "pgfplots"

# Build condition-aggregated series (merges all agents per condition).
_pgf_combined: dict[str, pd.DataFrame] = {}
for label in conditions_present:
    frames = [df for df in all_data.get(label, {}).values()]
    if frames:
        merged = pd.concat(frames, ignore_index=True)
        _pgf_combined[label] = _step_stats(merged)

for agent in ["combined"]:
    series: dict[str, pd.DataFrame] = _pgf_combined

    if not series:
        continue

    # Build a single wide CSV aligned on ep_step.
    all_steps = sorted({s for df in series.values() for s in df.index})
    last_step = all_steps[-1] if all_steps else None
    if _PGFPLOTS_DOWNSAMPLE > 1:
        all_steps = [s for i, s in enumerate(all_steps) if i % _PGFPLOTS_DOWNSAMPLE == 0]
        if last_step is not None and last_step not in all_steps:
            all_steps.append(last_step)
    csv_rows = {"x": all_steps}
    for label, df in series.items():
        key = _sanitize(label)
        df_reindexed = df.reindex(all_steps)
        csv_rows[f"mean{key}"] = df_reindexed["mean"].values
        csv_rows[f"up{key}"]   = df_reindexed["up"].values
        csv_rows[f"low{key}"]  = df_reindexed["low"].values

    csv_df = pd.DataFrame(csv_rows)

    PGFPLOTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_name = f"steps_{agent}.csv"
    csv_path = PGFPLOTS_DIR / csv_name
    csv_df.to_csv(csv_path, index=False)

    # Build per-condition color map (Set1, same as the matplotlib plots).
    palette = sns.color_palette("Set1", len(conditions_present))
    cond_colors = {
        label: "#{:02X}{:02X}{:02X}".format(
            int(r * 255), int(g * 255), int(b * 255)
        )
        for label, (r, g, b) in zip(conditions_present, palette)
    }

    # Build .tex snippet – self-contained, paste directly into Overleaf.
    marks = ["*", "square*", "triangle*", "diamond*", "pentagon*"]

    lines = [
        r"% ============================================================",
        r"% Add these packages to your LaTeX preamble (if not present):",
        r"%   \usepackage{pgfplots}",
        r"%   \usepackage{pgfplotstable}",
        r"%   \usepackage{siunitx}",
        r"%   \usepgfplotslibrary{fillbetween}",
        r"%   \pgfplotsset{compat=1.18}",
        r"% ============================================================",
        "",
    ]

    # Color definitions.
    for label in series:
        key = _sanitize(label)
        hex_col = cond_colors.get(label, "#000000")
        lines.append(f"\\definecolor{{color{key}}}{{HTML}}{{{hex_col[1:]}}}")
    lines.append("")

    # Shared + per-condition pgfplots styles.
    lines += [
        r"\pgfplotsset{",
        r"    % Shared styles",
        r"    ci path/.style={draw=none},",
        r"    ci fill/.style={opacity=0.15},",
        r"    mean/.style={thick, mark repeat=200, mark size=1.5pt},",
        r"    % Per-condition styles",
    ]
    for i, label in enumerate(series):
        key = _sanitize(label)
        mark = marks[i % len(marks)]
        lines.append(
            f"    {key}/.style={{color=color{key}, mark={mark}}},"
        )
    lines += [r"}", ""]

    # Axis.
    lines += [
        r"\begin{tikzpicture}",
        r"    \begin{axis}[",
        r"        grid=major,",
        r"        grid style={dashed, gray!30},",
        r"        width=\linewidth,",
        r"        xlabel={\small $t$},",
        r"        ylabel={\small $\tau$ [\si{\mega\bit\per\second}]},",
        r"        legend style={at={(0.5,-0.20)}, anchor=north, legend columns=3},",
        r"        tick label style={font=\small},",
        r"        label style={font=\small},",
        r"    ]",
        "",
    ]

    for label, _ in series.items():
        key = _sanitize(label)
        lines += [
            f"    % ===== {label} =====",
            f"    \\addplot [name path={key} upper, ci path, forget plot] table [x=x, y=up{key}, col sep=comma] {{{csv_name}}};",
            f"    \\addplot [name path={key} lower, ci path, forget plot] table [x=x, y=low{key}, col sep=comma] {{{csv_name}}};",
            f"    \\addplot [{key}, ci fill, forget plot] fill between [of={key} upper and {key} lower];",
            f"    \\addplot [{key}, mean] table [x=x, y=mean{key}, col sep=comma] {{{csv_name}}};",
            f"    \\addlegendentry{{{label}}}",
            "",
        ]

    lines += [
        r"    \end{axis}",
        r"\end{tikzpicture}",
    ]

    tex_path = PGFPLOTS_DIR / f"steps_{agent}.tex"
    tex_path.write_text("\n".join(lines))
    print(f"  PGFPlots: steps_{agent}.csv / .tex")

print(f"\nAll plots saved to {OUTPUT_DIR}/")
print(f"PGFPlots files saved to {PGFPLOTS_DIR}/")
