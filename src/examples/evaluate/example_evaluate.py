"""Example: evaluate trained PADRE agents (and baselines) by passing parameters.

`evaluate()` (src/evaluate_MANETAgents.py) resolves agents from `agent_patterns`
(a glob; a specific `*.zip` loads that checkpoint, otherwise the longest-trained model
in the folder is used), groups them with the baselines, runs them through identical
episodes (when `seed != 0`), and writes per-step CSVs to `log_dir`.

Baselines are `(AgentClass, ConditionStrategy)` tuples. Available baselines:
RandomAgent, GreedyMaximizer (GTM), OutAndThroughAgent, SpiralAgent, RandomSampler.
Condition strategies: SINR / Signal / Interference / TrafficHosted (local), Throughput
(global).

Environments: Static_MANETEnv, Less_Static_MANETEnv, Dynamic_MANETEnv.

Key `eval_params` (see EvalParams in evaluate_MANETAgents.py for the full list and
defaults): n_eval_eps, ep_length, jammer_ep_steps, step_jammers_start_moving, seed,
deterministic, obs_type ("full"/"random"/"real"), state_loss_rate, numbOfJammers,
numbOfUsers, attackerModel, routing_func, step_size, metric_param, filterval, and the
Byzantine parameters shown below.

Run from the `resilient-backbone/` repo root:

    python src/examples/evaluate/example_evaluate.py
"""

# Make `src/` importable regardless of the working directory.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from evaluate_MANETAgents import evaluate
import MANET.Environments as envs
import MANET.Agents as ag
import MANET.ConditionStrategy as cs

# * Agent checkpoints / folders to evaluate (glob patterns).
agent_patterns = [
    "out/models/29_04/**",
]

# * Baselines to compare against: (AgentClass, ConditionStrategy).
baselines_with_cs = [
    (ag.GreedyMaximizer, cs.TrafficHostedConditionStrategy(np.inf)),
]

# * Evaluation parameters.
eval_params = {
    "seed": 3,
    #
    "n_eval_eps": 50,
    "ep_length": 1000,
    "jammer_ep_steps": 500,
    "step_jammers_start_moving": 100,
    #
    "deterministic": True,
    "render_mode": None,
    #
    "numbOfJammers": 1,
    "numbOfUsers": (2, 21),
    #
    "networkConnectedAtStart": False,
    "jammersSpawnNextToUsers": False,
    "jammers_preposition": False,
    "allow_early_ep_finish": False,
    #
    "attackerModel": [
        "Static_GreedyJammers",
        "Static_ClusterJammers",
        "Static_TrafficJammers",
    ],
    #
    "routing_func": "sequential_max_flow",
    #
    "obs_type": "real",          # "full" (CTDE) / "random" / "real" (deployment)
    "state_loss_rate": 0.4,
    "step_size": 2,
    #
    # * Byzantine observation-manipulation attack (optional). The last
    #   `numbOfByzantineNodes` nodes broadcast a falsified state vector.
    # "numbOfByzantineNodes": 2,
    # "byzantine_attack_type": "obs_user_dir",        # dominant heuristic attack
    # "byzantine_obs_user_dir_offset": [4.0, 0.0],
    #   Other attack types and their parameters:
    #     obs_capacity          -> byzantine_obs_capacity_factor (>1 inflate, <1 deflate)
    #     obs_demand            -> byzantine_obs_demand_factor
    #     obs_interference_lie  -> byzantine_obs_interference_factor (0.0 hides interference)
    #     obs_noise             -> byzantine_obs_noise_std
    #     obs_capacity_deflate  -> byzantine_obs_capacity_deflate_factor
    #     obs_demand_deflate    -> byzantine_obs_demand_deflate_factor
    #     obs_replay            -> byzantine_obs_replay_delay (int steps)
}

evaluate(
    env_constr=envs.Static_MANETEnv,
    agent_patterns=agent_patterns,
    baselines_with_cs=baselines_with_cs,
    eval_params=eval_params,
    eval_type="grouped_eval",
    log_dir="out/evaluations/test1",
    log_results=True,
    log_state=True,
    log_eval=True,
)
