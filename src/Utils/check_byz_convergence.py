"""
Report PPO training convergence from TensorBoard event files.

Reads the rollout/ep_rew_mean curve (for the Byzantine trainer the reward is
-throughput, so a MORE NEGATIVE value is a stronger attack) and prints a
sampled curve plus a plateau verdict, to judge whether a run converged or was
still improving at the budget cutoff.

Usage (cluster or local):
    python src/Utils/check_byz_convergence.py <tb_logs_dir> [<tb_logs_dir> ...]

Example:
    python src/Utils/check_byz_convergence.py \\
        /itet-stor/arbaur/net_scratch/out/models/learned_byzantine/29_04_2byz_heard_60M/tb_logs \\
        /itet-stor/arbaur/net_scratch/out/models/learned_byzantine/29_04_2byz_coordinated_60M/tb_logs
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator

TAGS = ("rollout/ep_rew_mean", "train/ep_rew_mean")


def load_curve(dirpath: str) -> list[tuple[int, float]]:
    pts: list[tuple[int, float]] = []
    for ev in Path(dirpath).rglob("events.out.tfevents.*"):
        ea = event_accumulator.EventAccumulator(
            str(ev), size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        scalar_tags = ea.Tags().get("scalars", [])
        tag = next((t for t in TAGS if t in scalar_tags), None)
        if tag is None:
            continue
        for s in ea.Scalars(tag):
            pts.append((s.step, s.value))
    pts.sort()
    return pts


def report(name: str, pts: list[tuple[int, float]]) -> None:
    if not pts:
        print(f"\n=== {name} ===\n  no ep_rew_mean scalar found")
        return
    steps = [p[0] for p in pts]
    vals = [p[1] for p in pts]
    n = len(pts)
    print(f"\n=== {name}  ({n} points, steps {steps[0]:,}..{steps[-1]:,}) ===")
    for i in range(0, n, max(1, n // 10)):
        print(f"  step {steps[i]:>12,}  ep_rew_mean {vals[i]:8.3f}")
    print(f"  step {steps[-1]:>12,}  ep_rew_mean {vals[-1]:8.3f}  (final)")

    q = max(1, n // 5)
    last = statistics.mean(vals[-q:])
    prev = statistics.mean(vals[-2 * q:-q]) if n >= 2 * q else last
    delta = last - prev
    rel = abs(delta) / (abs(last) + 1e-9)
    verdict = "PLATEAUED" if rel < 0.03 else "STILL IMPROVING"
    print(f"  last-20% mean {last:.3f} vs previous-20% {prev:.3f}  "
          f"(delta {delta:+.3f}, {rel*100:.1f}% relative) -> {verdict}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python src/Utils/check_byz_convergence.py <tb_logs_dir> ...")
    for d in sys.argv[1:]:
        report(d, load_curve(d))
