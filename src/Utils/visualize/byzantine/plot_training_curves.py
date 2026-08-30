"""
Plot training reward curves for learned Byzantine attack runs.

Reads TensorBoard event files and produces:
  - PNG/SVG matplotlib plot
  - PGFPlots CSV + .tex for LaTeX

Usage (from resilient-backbone/):
    python src/Utils/visualize/byzantine/plot_training_curves.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RUNS = {
    "Oracle (full obs)": "out/models/learned_byzantine/29_04_2byz/tb_logs/PPO_2",
    "Local obs only":    "out/models/learned_byzantine/29_04_2byz_local/tb_logs/PPO_2",
}

COLORS = {
    "Oracle (full obs)": "#377EB8",
    "Local obs only":    "#E41A1C",
}

TAG = "rollout/ep_rew_mean"
SMOOTH_WINDOW = 20  # rolling mean over N log points

OUT_DIR = Path("out/visualizations/training_curves")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------

def load_scalar(path: str, tag: str) -> pd.DataFrame:
    ea = EventAccumulator(path)
    ea.Reload()
    events = ea.Scalars(tag)
    return pd.DataFrame({"step": [e.step for e in events], "value": [e.value for e in events]})


def smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1, center=True).mean()


def main():
    fig, ax = plt.subplots(figsize=(7, 4))

    pgf_frames = {}

    for label, tb_path in RUNS.items():
        df = load_scalar(tb_path, TAG)
        df = df.sort_values("step").reset_index(drop=True)
        df["smoothed"] = smooth(df["value"], SMOOTH_WINDOW)
        color = COLORS[label]

        ax.plot(df["step"] / 1e6, df["smoothed"], label=label, color=color, linewidth=1.5)
        ax.plot(df["step"] / 1e6, df["value"], color=color, alpha=0.15, linewidth=0.5)

        pgf_frames[label] = df[["step", "value", "smoothed"]].copy()

    ax.set_xlabel("Training steps (M)")
    ax.set_ylabel("PADRE episode reward (lower = more disruption)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"training_curves.{ext}", dpi=150)
    plt.close(fig)
    print(f"Saved plot to {OUT_DIR}/training_curves.{{png,svg}}")

    # -----------------------------------------------------------------------
    # PGFPlots export
    # -----------------------------------------------------------------------
    pgf_dir = OUT_DIR / "pgfplots"
    pgf_dir.mkdir(exist_ok=True)

    # Build combined CSV: step, oracle_raw, oracle_smooth, local_raw, local_smooth
    oracle_df = pgf_frames["Oracle (full obs)"].rename(columns={"value": "oracle_raw", "smoothed": "oracle_smooth"})
    local_df  = pgf_frames["Local obs only"].rename(columns={"value": "local_raw",  "smoothed": "local_smooth"})
    combined  = pd.merge(oracle_df, local_df, on="step", how="outer").sort_values("step")
    combined["step_M"] = combined["step"] / 1e6
    combined = combined[["step_M", "oracle_raw", "oracle_smooth", "local_raw", "local_smooth"]]
    csv_path = pgf_dir / "training_curves.csv"
    combined.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"Saved CSV to {csv_path}")

    tex = r"""\definecolor{colorOracle}{HTML}{377EB8}
\definecolor{colorLocal}{HTML}{E41A1C}

\begin{tikzpicture}
    \begin{axis}[
        grid=major, grid style={dashed, gray!30},
        width=\linewidth,
        xlabel={\small Training steps (M)},
        ylabel={\small PADRE episode reward (lower $=$ more disruption)},
        legend style={at={(0.5,-0.20)}, anchor=north, legend columns=2},
        tick label style={font=\small}, label style={font=\small},
    ]
    % Oracle — raw (faint)
    \addplot [colorOracle, opacity=0.2, forget plot, thin]
        table [x=step_M, y=oracle_raw, col sep=comma] {training_curves.csv};
    % Oracle — smoothed
    \addplot [colorOracle, thick]
        table [x=step_M, y=oracle_smooth, col sep=comma] {training_curves.csv};
    \addlegendentry{Oracle (full obs)}

    % Local — raw (faint)
    \addplot [colorLocal, opacity=0.2, forget plot, thin]
        table [x=step_M, y=local_raw, col sep=comma] {training_curves.csv};
    % Local — smoothed
    \addplot [colorLocal, thick]
        table [x=step_M, y=local_smooth, col sep=comma] {training_curves.csv};
    \addlegendentry{Local obs only}

    \end{axis}
\end{tikzpicture}
"""
    tex_path = pgf_dir / "training_curves.tex"
    tex_path.write_text(tex)
    print(f"Saved .tex to {tex_path}")


if __name__ == "__main__":
    main()
