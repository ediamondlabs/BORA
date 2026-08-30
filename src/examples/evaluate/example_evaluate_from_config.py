"""Example: run an evaluation from a YAML config.

`evaluate_from_config(path)` reads the config and dispatches based on its `mode`
(grouped eval / filter agents / filter models). Works for plain evaluation as well as
agent/model filtering.

Config files (same directory):
    example_eval_config.yaml               — grouped evaluation
    example_obs_manipulation_eval_config.yaml — sweep over obs-manipulation attacks
    example_resiliency_eval_config.yaml    — resiliency / defense evaluation
    example_byzantine_eval_config.yaml     — Byzantine attack evaluation

Run from the `resilient-backbone/` repo root:

    python src/examples/evaluate/example_evaluate_from_config.py
"""

# Make `src/` importable regardless of the working directory.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluate_MANETAgents import evaluate_from_config

HERE = Path(__file__).resolve().parent

evaluate_from_config(str(HERE / "example_eval_config.yaml"))
