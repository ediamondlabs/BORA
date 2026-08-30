"""
Train and evaluate a learned Byzantine attacker against a frozen PADRE policy.

Implements the Stackelberg setup from the research plan:
  - Inner player (PADRE): frozen checkpoint; plays honestly.
  - Outer player (Byzantine): PPO agent trained to maximise -throughput by
    choosing obs_user_dir offsets injected into Byzantine nodes' broadcasts.

Usage — train:
    python src/Utils/train_learned_byzantine.py \\
        --honest-model "out/models/29_04/**" \\
        --eval-config  src/examples/evaluate/example_obs_manipulation_eval_config.yaml \\
        --n-byz 2 --timesteps 10000000 \\
        --output out/models/learned_byzantine/29_04_2byz

Usage — evaluate a trained Byzantine agent:
    python src/Utils/train_learned_byzantine.py --evaluate \\
        --honest-model  "out/models/29_04/**" \\
        --byz-model     out/models/learned_byzantine/29_04_2byz/byz_agent.zip \\
        --eval-config   src/examples/evaluate/example_obs_manipulation_eval_config.yaml \\
        --n-byz 2 --eps 100 --ep-len 2000 \\
        --log-dir out/evaluations/29_04_learned \\
        --agent-name PPO_5_1_10_100k_no_obs_attack_rl_model

Run from the resilient-backbone/ root so that relative out/ paths resolve.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root without install
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path("src").resolve()))

import MANET.RewardWrappers as rw
import Utils.training_utils as ut
from MANET.ByzantineAttackerEnv import ByzantineAttackerEnv
from MANET.Routing import MaxFlow
from functools import partial
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONDITION_NAME = "learned_obs_user_dir"


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------

def _find_model(pattern: str) -> str:
    """Return the most recently modified .zip model file matching *pattern*.

    A ``**`` glob also matches directories and config files; restricting to
    .zip avoids returning a directory (whose mtime can be newer right after a
    fresh rsync), which would make the caller derive the wrong model dir.
    """
    matches = [m for m in glob.glob(pattern, recursive=True) if m.endswith(".zip")]
    matches.sort(key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError(f"No .zip model found matching: {pattern!r}")
    return matches[-1]


def _load_eval_config(config_path: str) -> tuple[dict, dict]:
    """Return (top_level_config, eval_params_dict) from an eval YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg, cfg.get("eval_params", {})


def _build_env(
    honest_model_path: str,
    eval_cfg: dict,
    eval_params: dict,
    n_byz: int,
    obs_type: str,
    ep_length: int = None,
    attack_type: str = "obs_user_dir",
) -> tuple[object, dict]:
    """Create wrapped PADRE env with Byzantine nodes.

    Returns (wrapped_env, training_config).
    ep_length overrides the value in eval_params when provided (e.g. from --ep-len).
    attack_type selects which obs-manipulation the Byzantine nodes use.
    """
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
        byzantine_attack_type=attack_type,
        byzantine_obs_user_dir_offset=[0.0, 0.0],  # overridden each step (obs_user_dir only)
    )

    wrapped_env = ut.wrapEnvFromConfig(
        env=base_env,
        config=training_config,
        rew_wrapper=rw.EvaluationReward,
        obs_type=obs_type,
        state_loss_rate=eval_params.get("state_loss_rate"),
    )

    return wrapped_env, training_config


# ---------------------------------------------------------------------------
# Vectorised-env factory (module-level so functools.partial stays picklable)
# ---------------------------------------------------------------------------

def _make_byz_env(
    honest_model_path: str,
    eval_cfg: dict,
    eval_params: dict,
    n_byz: int,
    obs_type: str,
    max_offset: float,
    obs_scope: str = "full",
    attack_type: str = "obs_user_dir",
) -> ByzantineAttackerEnv:
    """Create one ByzantineAttackerEnv instance — called inside each subprocess."""
    import torch
    torch.set_num_threads(1)  # prevent PyTorch thread oversubscription across 29 workers
    wrapped_env, _ = _build_env(
        honest_model_path, eval_cfg, eval_params, n_byz, obs_type,
        attack_type=attack_type,
    )
    honest_model = PPO.load(honest_model_path, device="cpu")
    return ByzantineAttackerEnv(
        base_env=wrapped_env,
        honest_policy=honest_model,
        n_byz=n_byz,
        max_offset=max_offset,
        obs_scope=obs_scope,
        attack_type=attack_type,
    )


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

class _CsvLogger:
    """Minimal buffered CSV logger matching compare_byzantine.py's expected format."""

    HEADER = ["ep_step", "episode", "value_name", "value"]
    _BUFFER_SIZE = 500
    _LOG_EVERY = 10  # write every Nth step (matches BufferedEvaluationLogger default)

    def __init__(self, log_dir: str, agent_name: str, episode_offset: int = 0) -> None:
        suffix = f"_{episode_offset}" if episode_offset else ""
        out_path = Path(log_dir) / f"{agent_name}{suffix}.csv"
        self._episode_offset = episode_offset
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(out_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)
        self._buf: list[tuple] = []
        print(f"  Logging to: {out_path}")

    def log(self, ep_step: int, episode: int, value_name: str, value: float) -> None:
        if ep_step % self._LOG_EVERY != 0:
            return
        self._buf.append((ep_step, episode + self._episode_offset, value_name, value))
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
        _make_byz_env,
        honest_path, eval_cfg, eval_params, args.n_byz, args.obs_type, args.max_offset,
        args.obs_scope, args.attack_type,
    )
    env_fns = [env_fn] * args.n_envs

    if args.n_envs > 1:
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        vec_env = DummyVecEnv(env_fns)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tb_log = str(output_dir / "tb_logs")

    steps_done = 0
    if args.resume_from:
        resume_path = Path(args.resume_from)
        # Parse completed steps from filename: byz_agent_16999626_steps.zip → 16999626
        stem = resume_path.stem  # strips .zip
        parts = stem.split("_")
        try:
            steps_done = int(parts[-2])  # second-to-last token before "steps"
        except (ValueError, IndexError):
            sys.exit(f"Cannot parse step count from checkpoint filename: {resume_path.name}")
        remaining = args.timesteps - steps_done
        if remaining <= 0:
            print(f"Checkpoint already at {steps_done:,} steps; target {args.timesteps:,} already reached.")
            vec_env.close()
            return
        print(f"\nResuming from {resume_path} ({steps_done:,} steps done, {remaining:,} remaining) …")
        byz_agent = PPO.load(str(resume_path), env=vec_env, verbose=1, tensorboard_log=tb_log)
    else:
        remaining = args.timesteps
        byz_agent = PPO(
            "MlpPolicy",
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

    print(f"\nTraining Byzantine PPO agent for {remaining:,} steps …")
    print(f"  Checkpoints every 1M steps → {output_dir}/checkpoints/")
    byz_agent.learn(
        total_timesteps=remaining,
        callback=checkpoint_cb,
        reset_num_timesteps=(args.resume_from is None),
    )

    save_path = str(output_dir / "byz_agent")
    byz_agent.save(save_path)
    print(f"\nSaved Byzantine agent to: {save_path}.zip")

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
        ep_length=ep_len, attack_type=args.attack_type,
    )

    honest_model = PPO.load(honest_path, device="cpu")
    byz_agent = PPO.load(args.byz_model, device="cpu")

    byz_env = ByzantineAttackerEnv(
        base_env=wrapped_env,
        honest_policy=honest_model,
        n_byz=args.n_byz,
        max_offset=args.max_offset,
        obs_scope=args.obs_scope,
        attack_type=args.attack_type,
    )

    condition_name = f"learned_{args.attack_type}"
    log_dir = str(Path(args.log_dir) / condition_name)
    agent_name = args.agent_name or Path(args.byz_model).stem
    logger = _CsvLogger(log_dir, agent_name, episode_offset=args.episode_offset)

    # Seed each shard distinctly (by --episode-offset) so parallel shards evaluate
    # DIFFERENT episodes; within a shard the RNG then advances freely, giving
    # independent episodes — instead of every shard replaying the same fixed-seed
    # sequence (the env constructor seeds from the config, so without this every
    # shard would produce identical episodes and only relabel them).
    base_seed = eval_params.get("seed") or 0
    shard_seed = base_seed + args.episode_offset
    print(f"\nEvaluating: {n_eps} episodes × {ep_len} steps (shard seed {shard_seed}) …")

    for episode in range(n_eps):
        obs, _ = byz_env.reset(seed=shard_seed if episode == 0 else None)
        for step in range(ep_len):
            byz_action, _ = byz_agent.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = byz_env.step(byz_action)
            throughput = byz_env.base_env.unwrapped.network.getThroughput() / 1e6  # Mbps, matching heuristic sweep CSV convention
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
        description="Train or evaluate a learned Byzantine attacker (Stackelberg setup)."
    )
    p.add_argument(
        "--honest-model", required=True, metavar="GLOB",
        help='Glob pattern for the frozen PADRE checkpoint, e.g. "out/models/29_04/**".',
    )
    p.add_argument(
        "--eval-config", required=True, metavar="YAML",
        help="Eval config YAML specifying env parameters (same format as the sweep).",
    )
    p.add_argument(
        "--n-byz", type=int, default=2,
        help="Number of Byzantine nodes (default: 2).",
    )
    p.add_argument(
        "--max-offset", type=float, default=4.0,
        help="Action scaling: agent action in [-1,1] maps to offset in [-max_offset, max_offset] (default: 4.0).",
    )
    p.add_argument(
        "--obs-type", default="full", choices=["real", "full", "gossip1", "reliable"],
        help=(
            "Observation type for the honest policy's env wrapping. "
            "'full' (default): centralized CTDE obs, matching PADRE training. "
            "'real': multi-hop flooding, decentralized (requires --node-idx). "
            "'gossip1': 1-hop BFS only — Byzantine lies reach direct neighbors only. "
            "'reliable': flooding + min-merge cache against capacity/demand inflation."
        ),
    )
    p.add_argument(
        "--obs-scope", default="full", choices=["full", "local", "heard", "coordinated"],
        help=(
            "Byzantine agent observation scope. "
            "'full' (default): full CTDE obs vector — oracle-info upper bound. "
            "'local': each Byzantine node sees only its own per-node obs slice (Stage A). "
            "'heard': own obs + in-range neighbor obs, zero-padded (Stage B). "
            "'coordinated': Stage B + covert channel to all other Byzantine nodes (Stage C)."
        ),
    )
    p.add_argument(
        "--attack-type", default="obs_user_dir", choices=["obs_user_dir", "obs_full"],
        help=(
            "Falsification strategy for Byzantine nodes. "
            "'obs_user_dir' (default): falsify only the 2D user-direction offset; "
            "action space (n_byz * 2,). "
            "'obs_full': falsify the entire per-node obs vector; "
            "action space (n_byz * per_node_obs_dim,)."
        ),
    )

    # Training-specific
    p.add_argument(
        "--resume-from", metavar="CHECKPOINT_ZIP", default=None,
        help=(
            "Path to a CheckpointCallback .zip to resume from. "
            "Step count is parsed from the filename (e.g. byz_agent_16999626_steps.zip → 16999626). "
            "Training continues for (--timesteps - steps_done) additional steps."
        ),
    )
    p.add_argument(
        "--timesteps", type=int, default=10_000_000,
        help="Total training timesteps for Byzantine PPO agent (default: 10 000 000).",
    )
    p.add_argument(
        "--n-envs", type=int, default=29,
        help="Number of parallel environments for training (default: 29, matching PADRE training).",
    )
    p.add_argument(
        "--output", default="out/models/learned_byzantine/run",
        help="Output directory for trained Byzantine model and logs.",
    )

    # Evaluation-specific
    p.add_argument(
        "--evaluate", action="store_true",
        help="Run in evaluation mode instead of training.",
    )
    p.add_argument(
        "--byz-model", metavar="PATH",
        help="Path to a trained Byzantine model .zip (required with --evaluate).",
    )
    p.add_argument(
        "--eps", type=int, default=None,
        help="Number of evaluation episodes (overrides eval config n_eval_eps).",
    )
    p.add_argument(
        "--ep-len", type=int, default=None,
        help="Episode length in steps (overrides eval config ep_length).",
    )
    p.add_argument(
        "--log-dir", default="out/evaluations/learned_run",
        help="Output directory for evaluation CSVs (default: out/evaluations/learned_run).",
    )
    p.add_argument(
        "--agent-name", default=None,
        help="Stem used as the CSV filename (default: Byzantine model filename stem).",
    )
    p.add_argument(
        "--episode-offset", type=int, default=0,
        help="Offset added to episode numbers in CSV output (for parallel eval shards).",
    )

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
