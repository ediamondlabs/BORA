"""
Adversarial-training defense — train an honest PADRE policy FROM SCRATCH against
a frozen, *learned* Byzantine attacker (e.g. the obs_full BORA model).

This mirrors the standard honest training (same obs wrapper from the victim's
config, SquaredReward, PPO hyperparameters via ut.get_model) but keeps f
Byzantine nodes present throughout, with their per-step falsification offset
driven by a loaded Byzantine policy (AdvHonestEnv) instead of the heuristic
direction lie. The honest swarm must therefore learn relocation strategies that
remain effective against the strongest known attacker.

Usage:
    python src/Utils/train_adv_honest_vs_full_obs.py \\
        --honest-model "out/models/29_04/**" \\
        --byz-model    out/models/learned_byzantine/29_04_2byz_full_obs_60M/byz_agent.zip \\
        --eval-config  src/examples/evaluate/example_obs_manipulation_eval_config.yaml \\
        --n-byz 2 --timesteps 30000000 --n-envs 29 \\
        --output out/models/adv_honest_2byz_full_obs

Run from the resilient-backbone/ root.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

import MANET.RewardWrappers as rw
import Utils.training_utils as ut
from MANET.ByzantineAttackerEnv import AdvHonestEnv
from MANET.Routing import MaxFlow
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Reuse the model-finder and config-loader from the learned-Byzantine trainer.
from Utils.train_learned_byzantine import _find_model, _load_eval_config


def _build_train_env(honest_model_path, eval_cfg, eval_params, n_byz, obs_type,
                     attack_type, ep_length):
    """Honest TRAINING env: victim's obs wrapper (from its config) + SquaredReward,
    with n_byz Byzantine nodes in `attack_type` mode. The Byzantine offset itself
    is set later, per step, by the frozen Byzantine policy inside AdvHonestEnv."""
    model_dir = os.path.dirname(honest_model_path)
    config_file = ut.getMostRecentConfigFile(model_dir)
    training_config = ut.loadConfig(model_dir, config_file)

    num_nodes = training_config["numbOfNodes"]
    num_jammers = eval_params.get("numbOfJammers", 1)
    num_users = eval_params.get("numbOfUsers", [2, 21])
    if isinstance(num_users, list):
        num_users = tuple(num_users)

    attacker_model = ut.getAttackerModelFromName(
        eval_params.get("attackerModel", "Static_ClusterJammers"))
    env_constr = ut.getEnvironmentFromName(eval_cfg.get("env_constr", "Dynamic_MANETEnv"))
    routing_func = getattr(MaxFlow, eval_params.get("routing_func", "sequential_max_flow"),
                           MaxFlow.sequential_max_flow)

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
        ep_length=ep_length,
        steps_till_jammer_active=eval_params.get("jammer_ep_steps", 200),
        routing_func=routing_func,
        step_size=eval_params.get("step_size", 2),
        numbOfByzantineNodes=n_byz,
        byzantine_attack_type=attack_type,
        byzantine_obs_user_dir_offset=[0.0, 0.0],
    )
    wrapped_env = ut.wrapEnvFromConfig(
        env=base_env,
        config=training_config,
        rew_wrapper=rw.SquaredReward,
        obs_type=obs_type,
        state_loss_rate=eval_params.get("state_loss_rate"),
    )
    return wrapped_env


def _make_adv_env(honest_path, byz_path, eval_cfg, eval_params, n_byz, obs_type,
                  obs_scope, attack_type, max_offset, ep_length):
    wrapped = _build_train_env(honest_path, eval_cfg, eval_params, n_byz, obs_type,
                               attack_type, ep_length)
    byz = PPO.load(byz_path, device="cpu")
    return AdvHonestEnv(
        base_env=wrapped, byz_policy=byz, n_byz=n_byz,
        max_offset=max_offset, obs_scope=obs_scope, attack_type=attack_type,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Adversarial honest training vs a frozen learned Byzantine attacker.")
    p.add_argument("--honest-model", required=True, metavar="GLOB",
                   help="victim policy glob — used only for its obs-wrapper config")
    p.add_argument("--byz-model", required=True, metavar="PATH",
                   help="frozen learned Byzantine policy (e.g. obs_full BORA)")
    p.add_argument("--eval-config", required=True, metavar="YAML")
    p.add_argument("--n-byz", type=int, default=2)
    p.add_argument("--max-offset", type=float, default=4.0)
    p.add_argument("--obs-type", default="full", choices=["real", "full", "gossip1", "reliable"])
    p.add_argument("--obs-scope", default="heard", choices=["full", "local", "heard", "coordinated"],
                   help="must match the scope the Byzantine model was trained with")
    p.add_argument("--attack-type", default="obs_full", choices=["obs_user_dir", "obs_full"])
    p.add_argument("--ep-len", type=int, default=2000)
    p.add_argument("--timesteps", type=int, default=30_000_000)
    p.add_argument("--n-envs", type=int, default=29)
    p.add_argument("--output", default="out/models/adv_honest_2byz_full_obs")
    p.add_argument("--seed", type=int, default=9)
    p.add_argument("--resume-from", default=None,
                   help="Checkpoint .zip to resume from; its filename must contain the "
                        "completed step count (e.g. honest_adv_30000000_steps.zip). "
                        "--timesteps is then the TOTAL target.")
    p.add_argument("--init-from", default=None,
                   help="Warm-start the trainable honest policy from this model "
                        "(e.g. the undefended policy out/models/29_04/**) and train a "
                        "FRESH --timesteps budget (fine-tuning). Mutually exclusive with --resume-from.")
    args = p.parse_args()
    if args.resume_from and args.init_from:
        sys.exit("Use only one of --resume-from / --init-from.")

    honest_path = _find_model(args.honest_model)
    eval_cfg, eval_params = _load_eval_config(args.eval_config)

    if not os.path.isfile(args.byz_model):
        sys.exit(f"Byzantine model not found: {args.byz_model}")

    env_fn = partial(
        _make_adv_env, honest_path, args.byz_model, eval_cfg, eval_params,
        args.n_byz, args.obs_type, args.obs_scope, args.attack_type,
        args.max_offset, args.ep_len,
    )
    env_fns = [env_fn] * args.n_envs
    vec_env = (SubprocVecEnv(env_fns, start_method="spawn")
               if args.n_envs > 1 else DummyVecEnv(env_fns))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Copy the victim's training config into the output dir so the standard
    # evaluation tools (run_byzantine_sweep / train_learned_byzantine) can
    # rebuild the env from this model dir — the defended policy shares the
    # victim's observation wrapper and network layout.
    _cfg_dir = os.path.dirname(honest_path)
    _cfg_name = ut.getMostRecentConfigFile(_cfg_dir)
    shutil.copyfile(os.path.join(_cfg_dir, _cfg_name), out / _cfg_name)
    print(f"Copied victim config {_cfg_name} -> {out}")

    if args.resume_from:
        resume_path = Path(args.resume_from)
        try:
            steps_done = int(resume_path.stem.split("_")[-2])  # honest_adv_30000000_steps -> 30000000
        except (ValueError, IndexError):
            sys.exit(f"Cannot parse step count from checkpoint filename: {resume_path.name}")
        remaining = args.timesteps - steps_done
        if remaining <= 0:
            sys.exit(f"Checkpoint already at {steps_done:,} steps; target {args.timesteps:,} reached.")
        print(f"Resuming from {resume_path} ({steps_done:,} done, {remaining:,} remaining) …")
        agent = PPO.load(str(resume_path), env=vec_env, tensorboard_log=str(out / "tb_logs"))
    elif args.init_from:
        init_path = _find_model(args.init_from)
        remaining = args.timesteps
        print(f"Fine-tuning: warm-starting from {init_path}, training {remaining:,} fresh steps …")
        agent = PPO.load(init_path, env=vec_env, tensorboard_log=str(out / "tb_logs"))
    else:
        remaining = args.timesteps
        agent = ut.get_model("PPO", vec_env, stats_window_size=10,
                             log_dir=str(out / "tb_logs"), seed=args.seed)

    ckpt = CheckpointCallback(
        save_freq=max(1_000_000 // args.n_envs, 1),
        save_path=str(out / "checkpoints"), name_prefix="honest_adv", verbose=1,
    )

    print(f"Victim config from : {honest_path}")
    print(f"Frozen Byzantine   : {args.byz_model}  (scope={args.obs_scope}, attack={args.attack_type})")
    print(f"Honest training    : {remaining:,} steps ({'resume' if args.resume_from else 'from scratch'}), {args.n_envs} envs, f={args.n_byz}")
    agent.learn(total_timesteps=remaining, callback=ckpt, progress_bar=True,
                reset_num_timesteps=(args.resume_from is None))

    save_path = str(out / "honest_adv")
    agent.save(save_path)
    print(f"\nSaved adversarially-trained honest policy to: {save_path}.zip")
    vec_env.close()


if __name__ == "__main__":
    main()
