"""
Train and evaluate a CTDE BABEL attacker: coordinated actor, global-state critic.

The policy (actor) sees only the coordinated observation scope — what the
Byzantine nodes can collectively observe at decentralised execution time.
The critic sees the full CTDE global state during training, matching the
standard centralised-training / decentralised-execution convention.

Usage — train:
    python src/Utils/train_learned_byzantine_ctde.py \\
        --honest-model "out/models/29_04/**" \\
        --eval-config  src/examples/evaluate/example_obs_manipulation_eval_config.yaml \\
        --n-byz 2 --timesteps 60000000 \\
        --output out/models/learned_byzantine/29_04_2byz_coordinated_ctde_60M

Usage — evaluate:
    python src/Utils/train_learned_byzantine_ctde.py --evaluate \\
        --honest-model  "out/models/29_04/**" \\
        --byz-model     out/models/learned_byzantine/29_04_2byz_coordinated_ctde_60M/byz_agent.zip \\
        --eval-config   src/examples/evaluate/example_obs_manipulation_eval_config.yaml \\
        --n-byz 2 --eps 1000 --ep-len 2000 \\
        --log-dir out/evaluations/29_04_learned_coordinated_ctde_60M_1000ep

Run from the resilient-backbone/ root so that relative out/ paths resolve.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import yaml

sys.path.insert(0, str(Path("src").resolve()))

import MANET.RewardWrappers as rw
import Utils.training_utils as ut
from MANET.ByzantineAttackerEnv import ByzantineAttackerEnv
from MANET.Routing import MaxFlow
from functools import partial
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn

_CONDITION_NAME = "learned_obs_user_dir"
_FEATURES_DIM = 64


# ---------------------------------------------------------------------------
# Feature extractors — actor sees coordinated obs, critic sees full state
# ---------------------------------------------------------------------------

class _PolicyExtractor(BaseFeaturesExtractor):
    """Extracts obs["policy"] (coordinated scope) for the actor."""

    def __init__(self, obs_space: gym.spaces.Dict, features_dim: int = _FEATURES_DIM):
        super().__init__(obs_space, features_dim=features_dim)
        in_dim = obs_space["policy"].shape[0]
        self.net = nn.Linear(in_dim, features_dim)

    def forward(self, obs: dict) -> th.Tensor:
        return th.tanh(self.net(obs["policy"].float()))


class _ValueExtractor(BaseFeaturesExtractor):
    """Extracts obs["value"] (full global state) for the critic."""

    def __init__(self, obs_space: gym.spaces.Dict, features_dim: int = _FEATURES_DIM):
        super().__init__(obs_space, features_dim=features_dim)
        in_dim = obs_space["value"].shape[0]
        self.net = nn.Linear(in_dim, features_dim)

    def forward(self, obs: dict) -> th.Tensor:
        return th.tanh(self.net(obs["value"].float()))


# ---------------------------------------------------------------------------
# Custom PPO policy — separate actor / critic inputs
# ---------------------------------------------------------------------------

class CTDEByzantinePolicy(MultiInputActorCriticPolicy):
    """PPO policy with CTDE-style actor/critic split.

    Actor input : obs["policy"]  — coordinated Byzantine observation.
    Critic input: obs["value"]   — full centralised CTDE state (training only).

    Both feature extractors map to the same _FEATURES_DIM so that the shared
    MlpExtractor downstream can process both without dimension mismatch.
    """

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        kwargs["share_features_extractor"] = False
        kwargs["features_extractor_class"] = _PolicyExtractor
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule) -> None:
        # super()._build creates pi_features_extractor = _PolicyExtractor (correct)
        # and vf_features_extractor = _PolicyExtractor (wrong — replace below).
        # MlpExtractor is built with feature_dim = _FEATURES_DIM, which is the
        # same dim that _ValueExtractor will also output, so no mismatch occurs.
        super()._build(lr_schedule)
        self.vf_features_extractor = _ValueExtractor(self.observation_space).to(self.device)
        # Rebuild optimizer so the new extractor's parameters are included.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )


# ---------------------------------------------------------------------------
# CTDE environment wrapper
# ---------------------------------------------------------------------------

class ByzantineAttackerEnvCTDE(ByzantineAttackerEnv):
    """Gym wrapper exposing a dict observation space for CTDE training.

    obs["policy"] : coordinated Byzantine observation (what the attacker sees
                    at decentralised execution time).
    obs["value"]  : full CTDE global state (used only by the critic; not
                    available at deployment).
    """

    def __init__(self, base_env, honest_policy, n_byz: int, max_offset: float = 4.0):
        super().__init__(base_env, honest_policy, n_byz, max_offset, obs_scope="coordinated")

        coord_obs_dim: int = self.observation_space.shape[0]
        full_obs_dim: int = base_env.observation_space.shape[0]
        self.observation_space = spaces.Dict({
            "policy": spaces.Box(
                low=-np.inf, high=np.inf, shape=(coord_obs_dim,), dtype=np.float32
            ),
            "value": spaces.Box(
                low=-np.inf, high=np.inf, shape=(full_obs_dim,), dtype=np.float32
            ),
        })

    def reset(self, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        return self._dict_obs(), info

    def step(self, action):
        _, reward, terminated, truncated, info = super().step(action)
        return self._dict_obs(), reward, terminated, truncated, info

    def _dict_obs(self) -> dict:
        return {
            "policy": self._get_byz_obs(),
            "value": self._obs.astype(np.float32),
        }


# ---------------------------------------------------------------------------
# Environment construction (shared between train / eval)
# ---------------------------------------------------------------------------

def _find_model(pattern: str) -> str:
    matches = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError(f"No model found matching: {pattern!r}")
    return matches[-1]


def _load_eval_config(config_path: str) -> tuple[dict, dict]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg, cfg.get("eval_params", {})


def _build_env(honest_model_path, eval_cfg, eval_params, n_byz, obs_type, ep_length=None):
    model_dir = os.path.dirname(honest_model_path)
    config_file = ut.getMostRecentConfigFile(model_dir)
    training_config = ut.loadConfig(model_dir, config_file)

    num_nodes = training_config["numbOfNodes"]
    num_jammers = eval_params.get("numbOfJammers", 1)
    num_users = eval_params.get("numbOfUsers", [2, 21])
    if isinstance(num_users, list):
        num_users = tuple(num_users)

    attacker_name = eval_params.get("attackerModel", "Static_ClusterJammers")
    attacker_model = ut.getAttackerModelFromName(attacker_name)

    env_constr_name = eval_cfg.get("env_constr", "Dynamic_MANETEnv")
    env_constr = ut.getEnvironmentFromName(env_constr_name)

    routing_name = eval_params.get("routing_func", "sequential_max_flow")
    routing_func = getattr(MaxFlow, routing_name, MaxFlow.sequential_max_flow)

    base_env = ut.createEnv(
        env_constructor=env_constr,
        numbOfNodes=num_nodes,
        numbOfJammers=num_jammers,
        numbOfUsers=num_users,
        seed=eval_params.get("seed"),
        render_mode=None,
        networkConnectedAtStart=eval_params.get("networkConnectedAtStart", False),
        jammersSpawnNextToUsers=eval_params.get("jammersSpawnNextToUsers", False),
        attackerModel=attacker_model,
        allow_early_ep_finish=eval_params.get("allow_early_ep_finish", False),
        ep_length=ep_length or eval_params.get("ep_length", 1000),
        steps_till_jammer_active=eval_params.get("jammer_ep_steps", 200),
        routing_func=routing_func,
        step_size=eval_params.get("step_size", 2),
        numbOfByzantineNodes=n_byz,
        byzantine_attack_type="obs_user_dir",
        byzantine_obs_user_dir_offset=[0.0, 0.0],
    )

    wrapped_env = ut.wrapEnvFromConfig(
        env=base_env,
        config=training_config,
        rew_wrapper=rw.EvaluationReward,
        obs_type=obs_type,
        state_loss_rate=eval_params.get("state_loss_rate"),
    )
    return wrapped_env, training_config


def _make_ctde_env(honest_model_path, eval_cfg, eval_params, n_byz, obs_type, max_offset):
    wrapped_env, _ = _build_env(honest_model_path, eval_cfg, eval_params, n_byz, obs_type)
    honest_model = PPO.load(honest_model_path, device="cpu")
    return ByzantineAttackerEnvCTDE(
        base_env=wrapped_env,
        honest_policy=honest_model,
        n_byz=n_byz,
        max_offset=max_offset,
    )


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------

class _CsvLogger:
    HEADER = ["ep_step", "episode", "value_name", "value"]
    _BUFFER_SIZE = 500
    _LOG_EVERY = 10

    def __init__(self, log_dir: str, agent_name: str) -> None:
        out_path = Path(log_dir) / f"{agent_name}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(out_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)
        self._buf: list[tuple] = []
        print(f"  Logging to: {out_path}")

    def log(self, ep_step, episode, value_name, value) -> None:
        if ep_step % self._LOG_EVERY != 0:
            return
        self._buf.append((ep_step, episode, value_name, value))
        if len(self._buf) >= self._BUFFER_SIZE:
            self._flush()

    def close(self) -> None:
        self._flush()
        self._file.close()

    def _flush(self) -> None:
        self._writer.writerows(self._buf)
        self._buf.clear()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    honest_path = _find_model(args.honest_model)
    print(f"Honest model : {honest_path}")
    print(f"Parallel envs: {args.n_envs}")

    eval_cfg, eval_params = _load_eval_config(args.eval_config)

    env_fn = partial(
        _make_ctde_env,
        honest_path, eval_cfg, eval_params, args.n_byz, args.obs_type, args.max_offset,
    )
    env_fns = [env_fn] * args.n_envs
    vec_env = SubprocVecEnv(env_fns, start_method="spawn") if args.n_envs > 1 else DummyVecEnv(env_fns)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tb_log = str(output_dir / "tb_logs")
    byz_agent = PPO(
        CTDEByzantinePolicy,
        vec_env,
        verbose=1,
        tensorboard_log=tb_log,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        clip_range=0.2,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1_000_000 // args.n_envs, 1),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="byz_agent",
        verbose=1,
    )

    print(f"\nTraining CTDE BABEL (coordinated actor / global-state critic) for {args.timesteps:,} steps …")
    print(f"  Checkpoints every 1M steps → {output_dir}/checkpoints/")
    byz_agent.learn(total_timesteps=args.timesteps, callback=checkpoint_cb)

    save_path = str(output_dir / "byz_agent")
    byz_agent.save(save_path)
    print(f"\nSaved to: {save_path}.zip")
    vec_env.close()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    honest_path = _find_model(args.honest_model)
    print(f"Honest model    : {honest_path}")
    print(f"Byzantine model : {args.byz_model}")

    eval_cfg, eval_params = _load_eval_config(args.eval_config)

    ep_len = args.ep_len or eval_params.get("ep_length", 1000)
    n_eps = args.eps or eval_params.get("n_eval_eps", 30)

    wrapped_env, _ = _build_env(
        honest_path, eval_cfg, eval_params, args.n_byz, args.obs_type,
        ep_length=ep_len,
    )
    honest_model = PPO.load(honest_path, device="cpu")
    byz_agent = PPO.load(args.byz_model, device="cpu", custom_objects={
        "policy_class": CTDEByzantinePolicy,
    })

    byz_env = ByzantineAttackerEnvCTDE(
        base_env=wrapped_env,
        honest_policy=honest_model,
        n_byz=args.n_byz,
        max_offset=args.max_offset,
    )

    log_dir = str(Path(args.log_dir) / _CONDITION_NAME)
    agent_name = args.agent_name or Path(args.byz_model).stem
    logger = _CsvLogger(log_dir, agent_name)

    print(f"\nEvaluating: {n_eps} episodes × {ep_len} steps …")

    for episode in range(n_eps):
        obs, _ = byz_env.reset()
        for step in range(ep_len):
            # At eval time only the policy (actor) part of obs is used for action selection.
            byz_action, _ = byz_agent.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = byz_env.step(byz_action)
            throughput = byz_env.base_env.unwrapped.network.getThroughput() / 1e6
            logger.log(step, episode, "throughput", throughput)
            if terminated or truncated:
                break
        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{n_eps}")

    logger.close()
    byz_env.close()
    print(f"\nDone. Results written to: {log_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train/evaluate CTDE BABEL: coordinated actor, global-state critic."
    )
    p.add_argument("--honest-model", required=True, metavar="GLOB")
    p.add_argument("--eval-config", required=True, metavar="YAML")
    p.add_argument("--n-byz", type=int, default=2)
    p.add_argument("--max-offset", type=float, default=4.0)
    p.add_argument("--obs-type", default="full", choices=["real", "full", "gossip1", "reliable"])

    # train
    p.add_argument("--timesteps", type=int, default=60_000_000)
    p.add_argument("--n-envs", type=int, default=29)
    p.add_argument("--output", default="out/models/learned_byzantine/ctde_run")

    # eval
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--byz-model", metavar="PATH")
    p.add_argument("--eps", type=int, default=None)
    p.add_argument("--ep-len", type=int, default=None)
    p.add_argument("--log-dir", default="out/evaluations/learned_ctde_run")
    p.add_argument("--agent-name", default=None)

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.evaluate:
        if not args.byz_model:
            sys.exit("--byz-model is required with --evaluate")
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
