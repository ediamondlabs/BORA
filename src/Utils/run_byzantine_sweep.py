"""
Run a Byzantine observation-manipulation sweep for a trained PADRE model.

Evaluates a model against every observation-manipulation attack type (plus the
no_byzantine baseline) and saves results under:
    out/evaluations/<experiment_name>/<attack_type>/

After all evaluations complete, compare_byzantine.py is run automatically to
produce the comparison plots in:
    out/visualizations/byzantine_comparison/<experiment_name>/

Usage (run from the resilient-backbone/ root):
    python src/Utils/run_byzantine_sweep.py \\
        --model "out/models/my_run/**" \\
        --name  "25_03"

    # Single-magnitude sweep with a custom magnitude value:
    python src/Utils/run_byzantine_sweep.py \\
        --model "out/models/my_run/**" \\
        --name  "25_03_mag2" \\
        --magnitude 2.0

Arguments:
    --model      Glob pattern passed to agent_patterns in the eval config.
                 Use the same pattern format as the existing eval YAML files.
    --name       Experiment name — used as the output subdirectory under
                 out/evaluations/ and out/visualizations/byzantine_comparison/.
    --no-plot    Skip running compare_byzantine.py after evaluations finish.
    --eps        Number of evaluation episodes per condition (default: 30).
    --ep-len     Episode length in steps (default: 1000).
    --magnitude  Attack magnitude (default: use per-condition SWEEP defaults).
                 Scales the primary attack parameter for each obs-manipulation type:
                   obs_user_dir         → byzantine_obs_user_dir_offset: [m, 0]
                   obs_capacity         → byzantine_obs_capacity_factor: 1+m
                   obs_demand           → byzantine_obs_demand_factor:   1+m
                   obs_interference_lie     → byzantine_obs_interference_factor: 1-m
                   obs_interference_amplify → byzantine_obs_interference_factor: 1+m
                   obs_noise                → byzantine_obs_noise_std: m
                   obs_capacity_deflate     → byzantine_obs_capacity_deflate_factor: 1-m
                   obs_demand_deflate       → byzantine_obs_demand_deflate_factor:   1-m
                   obs_replay               → byzantine_obs_replay_delay: int(m*50) steps
                 At m=0.0 all attacks are neutral (honest behaviour).
                 For obs_interference_lie, m=1 means fully hidden (max attack).
                 For obs_interference_amplify, m=4 means factor=5 (false alarm).
                 For obs_replay, m=1 → 50-step delay, m=4 → 200-step delay.
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Attack sweep definition
# Each key becomes a subdirectory under out/evaluations/<name>/.
# Values are merged on top of BASE_EVAL_PARAMS.
# ---------------------------------------------------------------------------

# Parameters shared across all conditions — mirrors the existing example configs.
BASE_EVAL_PARAMS: dict = {
    "allow_early_ep_finish": False,
    "attackerModel": "Static_ClusterJammers",
    "deterministic": True,
    "ep_length": 1000,
    "jammer_ep_steps": 200,
    "jammersSpawnNextToUsers": False,
    "n_eval_eps": 30,
    "networkConnectedAtStart": False,
    "numbOfJammers": 1,
    "numbOfUsers": [2, 21],
    "obs_type": "real",
    "render_mode": None,
    "routing_func": "sequential_max_flow",
    "seed": 22,
    "state_loss_rate": None,
    "step_size": 2,
}

# Each entry: attack subdirectory name → overrides applied to BASE_EVAL_PARAMS.
# Default parameter values correspond to magnitude=1.0.
SWEEP: dict[str, dict] = {
    # No Byzantine nodes — clean baseline.
    "no_byzantine": {
        "numbOfByzantineNodes": 0,
    },
    # obs_user_dir: lies about direction to closest user/sender/receiver.
    # Completely undetectable — neighbours cannot verify node's closest user.
    "obs_user_dir": {
        "numbOfByzantineNodes": 1,
        "byzantine_attack_type": "obs_user_dir",
        "byzantine_obs_user_dir_offset": [0.5, 0.0],
    },
    # obs_capacity: inflates reported in/out capacity to attract routing.
    "obs_capacity": {
        "numbOfByzantineNodes": 1,
        "byzantine_attack_type": "obs_capacity",
        "byzantine_obs_capacity_factor": 3.0,
    },
    # obs_demand: inflates reported traffic demand — "demand sinkhole" that
    # clusters legitimate nodes without any detectable physics signature.
    "obs_demand": {
        "numbOfByzantineNodes": 1,
        "byzantine_attack_type": "obs_demand",
        "byzantine_obs_demand_factor": 3.0,
    },
    # obs_interference_lie: reports near-zero interference even near jammers,
    # luring legitimate nodes into the jammed zone.
    "obs_interference_lie": {
        "numbOfByzantineNodes": 1,
        "byzantine_attack_type": "obs_interference_lie",
        "byzantine_obs_interference_factor": 0.0,
    },
    # obs_interference_amplify: reports exaggerated interference (factor >> 1),
    # creating a false alarm that scares honest nodes away from good relay positions.
    # Opposite direction to obs_interference_lie. Requires jammers to be meaningful
    # since it amplifies real interference; factor=5 at magnitude=4.
    "obs_interference_amplify": {
        "numbOfByzantineNodes": 1,
        "byzantine_attack_type": "obs_interference_lie",
        "byzantine_obs_interference_factor": 5.0,
    },
    # obs_noise: adds Gaussian noise to all broadcast observation fields.
    # Maximally detectable; useful as a noisy-baseline reference.
    "obs_noise": {
        "numbOfByzantineNodes": 1,
        "byzantine_attack_type": "obs_noise",
        "byzantine_obs_noise_std": 0.1,
    },
    # obs_all: all five obs-manipulation attacks active simultaneously on each
    # Byzantine node. With n_byz Byzantine nodes, each gets all five attacks
    # (multi-attack mode: len(attack_types) > n_byz triggers per-node stacking).
    # numbOfByzantineNodes here is the default; --n-byz overrides it at runtime.
    "obs_all": {
        "numbOfByzantineNodes": 1,
        "byzantine_attack_type": [
            "obs_user_dir",
            "obs_capacity",
            "obs_demand",
            "obs_interference_lie",
            "obs_noise",
        ],
        "byzantine_obs_user_dir_offset": [0.5, 0.0],
        "byzantine_obs_capacity_factor": 3.0,
        "byzantine_obs_demand_factor": 3.0,
        "byzantine_obs_interference_factor": 0.0,
        "byzantine_obs_noise_std": 0.1,
    },
    # obs_capacity_deflate: reports lower capacity than real, repelling honest nodes from
    # good relay positions (opposite direction to obs_capacity which inflates).
    "obs_capacity_deflate": {
        "numbOfByzantineNodes": 1,
        "byzantine_attack_type": "obs_capacity_deflate",
        "byzantine_obs_capacity_deflate_factor": 0.0,
    },
    # obs_demand_deflate: reports lower traffic demand, causing honest nodes to abandon
    # high-traffic areas (opposite direction to obs_demand which inflates).
    "obs_demand_deflate": {
        "numbOfByzantineNodes": 1,
        "byzantine_attack_type": "obs_demand_deflate",
        "byzantine_obs_demand_deflate_factor": 0.0,
    },
    # obs_replay: rebroadcasts own state from T steps ago — stale topology snapshot.
    # Delay of 50 steps corresponds to magnitude=1.0 (1/20th of a 1000-step episode).
    "obs_replay": {
        "numbOfByzantineNodes": 1,
        "byzantine_attack_type": "obs_replay",
        "byzantine_obs_replay_delay": 50,
    },
    # Combined attack using only the two confirmed harmful attacks.
    "obs_harmful": {
        "numbOfByzantineNodes": 1,
        "byzantine_attack_type": [
            "obs_user_dir",
            "obs_interference_lie",
        ],
        "byzantine_obs_user_dir_offset": [0.5, 0.0],
        "byzantine_obs_interference_factor": 0.0,
    },
}

# Legacy routing-layer attack conditions — disabled by default.
# Run manually by passing SWEEP = {**SWEEP, **LEGACY_SWEEP} if needed.
LEGACY_SWEEP: dict[str, dict] = {
    "byzantine_greyhole": {
        "numbOfByzantineNodes": 1,
        "byzantine_attack_type": "greyhole",
        "byzantine_drop_rate": 0.5,
    },
    "byzantine_sinkhole": {
        "numbOfByzantineNodes": 1,
        "byzantine_attack_type": "sinkhole",
        "byzantine_drop_rate": 0.5,
        "byzantine_capacity_inflation_factor": 3.0,
    },
    "byzantine_selective_jamming": {
        "numbOfByzantineNodes": 1,
        "byzantine_attack_type": "selective_jamming",
        "byzantine_jamming_power_factor": 1.5,
    },
    "byzantine_position_spoofing": {
        "numbOfByzantineNodes": 1,
        "byzantine_attack_type": "position_spoofing",
        "byzantine_pos_offset": [50.0, 50.0],
    },
    "byzantine_all_attacks": {
        "numbOfByzantineNodes": 1,
        "byzantine_attack_type": [
            "greyhole",
            "sinkhole",
            "selective_jamming",
            "position_spoofing",
        ],
        "byzantine_drop_rate": 0.5,
        "byzantine_capacity_inflation_factor": 3.0,
        "byzantine_jamming_power_factor": 1.5,
        "byzantine_pos_offset": [50.0, 50.0],
    },
}

# Maps byzantine_attack_type → (parameter key, scaling mode).
# Scaling modes:
#   "vector"          — magnitude → [magnitude, 0.0]      (direction offset; 0=no offset)
#   "direct"          — magnitude → magnitude              (std dev: 0=no noise)
#   "direct_inflation"— magnitude → 1.0 + magnitude       (factor; 0=honest, 1=2×, 4=5×)
#   "inverse"         — magnitude → max(0, 1.0−magnitude) (factor; 0=honest, 1=fully hidden)
#
# For capacity and demand the neutral value is 1.0 (honest), so "direct_inflation"
# keeps magnitude=0.0 at the honest baseline instead of factor=0.0 (strong deflation).
_MAGNITUDE_MAP: dict[str, tuple[str, str]] = {
    "obs_user_dir":              ("byzantine_obs_user_dir_offset",    "vector"),
    "obs_capacity":              ("byzantine_obs_capacity_factor",    "direct_inflation"),
    "obs_demand":                ("byzantine_obs_demand_factor",      "direct_inflation"),
    "obs_interference_lie":      ("byzantine_obs_interference_factor","inverse"),
    "obs_interference_amplify":  ("byzantine_obs_interference_factor","amplify"),
    "obs_noise":                 ("byzantine_obs_noise_std",          "direct"),
    "obs_capacity_deflate":      ("byzantine_obs_capacity_deflate_factor", "inverse"),
    "obs_demand_deflate":        ("byzantine_obs_demand_deflate_factor",   "inverse"),
    "obs_replay":                ("byzantine_obs_replay_delay",            "replay_steps"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COMPARE_SCRIPT = Path("src/Utils/visualize/byzantine/compare_byzantine.py")


def _apply_magnitude(
    params: dict,
    magnitude: float,
    condition_name: str = "",
    angle: float | None = None,
) -> None:
    """Overwrite the primary attack parameter(s) with the scaled magnitude value.

    For single-type attacks the _MAGNITUDE_MAP entry is used, keyed first by
    condition_name (to disambiguate obs_interference_amplify from obs_interference_lie),
    then by attack_type as fallback.
    For obs_all (list of all obs attack types) all five parameters are scaled
    simultaneously using their individual modes.

    angle (degrees): if provided, rotates the obs_user_dir offset vector from the
    default (magnitude, 0) to (magnitude*cos(θ), magnitude*sin(θ)).  Ignored for all
    other attack types.
    """
    attack_type = params.get("byzantine_attack_type")
    if isinstance(attack_type, list):
        # Combined attack — scale each obs parameter independently.
        if angle is not None:
            rad = math.radians(angle)
            params["byzantine_obs_user_dir_offset"] = [
                float(magnitude) * math.cos(rad),
                float(magnitude) * math.sin(rad),
            ]
        else:
            params["byzantine_obs_user_dir_offset"] = [float(magnitude), 0.0]
        params["byzantine_obs_capacity_factor"] = 1.0 + float(magnitude)
        params["byzantine_obs_demand_factor"] = 1.0 + float(magnitude)
        params["byzantine_obs_interference_factor"] = max(0.0, 1.0 - float(magnitude))
        params["byzantine_obs_noise_std"] = float(magnitude)
        return
    key = condition_name if condition_name in _MAGNITUDE_MAP else attack_type
    if not isinstance(key, str) or key not in _MAGNITUDE_MAP:
        return
    param_key, mode = _MAGNITUDE_MAP[key]
    if mode == "vector":
        if angle is not None:
            rad = math.radians(angle)
            params[param_key] = [
                float(magnitude) * math.cos(rad),
                float(magnitude) * math.sin(rad),
            ]
        else:
            params[param_key] = [float(magnitude), 0.0]
    elif mode == "inverse":
        params[param_key] = max(0.0, 1.0 - float(magnitude))
    elif mode == "direct_inflation":
        params[param_key] = 1.0 + float(magnitude)
    elif mode == "amplify":
        params[param_key] = 1.0 + float(magnitude)
    elif mode == "replay_steps":
        # magnitude → delay in steps: 0→0, 0.5→25, 1→50, 2→100, 4→200
        params[param_key] = int(magnitude * 50)
    else:
        params[param_key] = float(magnitude)


def _build_config(
    model_pattern: str,
    attack_name: str,
    overrides: dict,
    experiment_name: str,
    n_eps: int,
    ep_len: int,
    magnitude: float | None,
    n_byz: int | None = None,
    angle: float | None = None,
    n_envs: int = 1,
    results_base: str = "out/evaluations",
) -> dict:
    params = {**BASE_EVAL_PARAMS, **overrides}
    params["n_eval_eps"] = n_eps
    params["ep_length"] = ep_len
    params["n_envs"] = n_envs
    # Override Byzantine node count for all attack conditions (keep baseline at 0).
    if n_byz is not None and params.get("numbOfByzantineNodes", 0) > 0:
        params["numbOfByzantineNodes"] = n_byz
    if magnitude is not None:
        _apply_magnitude(params, magnitude, condition_name=attack_name, angle=angle)
    return {
        "mode": "evaluate",
        "agent_patterns": [model_pattern],
        "baselines_with_cs": [],
        "env_constr": "Dynamic_MANETEnv",
        "eval_params": params,
        "eval_type": "grouped_eval",
        "log_dir": f"{results_base}/{experiment_name}/{attack_name}",
        "log_results": True,
        "log_state": True,
        "log_eval": True,
    }


def _run_evaluation(config: dict) -> None:
    """Write config to a temp YAML file and call evaluate_from_config on it."""
    # Import here so the script can be run from the repo root without install.
    sys.path.insert(0, str(Path("src").resolve()))
    from evaluate_MANETAgents import evaluate_from_config  # noqa: PLC0415

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as tmp:
        yaml.dump(config, tmp, default_flow_style=False)
        tmp_path = tmp.name

    try:
        evaluate_from_config(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _spawn_evaluation(config: dict) -> tuple["subprocess.Popen[bytes]", str]:
    """Write config to a temp YAML and launch a subprocess for it.

    Returns (process, tmp_path). The caller must wait() on the process and
    then delete tmp_path.
    """
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(config, tmp, default_flow_style=False)
    tmp.close()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        [sys.executable, __file__, "--_run-single", tmp.name],
        env=env,
    )
    return proc, tmp.name


def _run_compare(experiment_name: str) -> None:
    """Run compare_byzantine.py with the correct EXPERIMENT_NAME."""
    if not COMPARE_SCRIPT.exists():
        print(f"  [skip] compare script not found at {COMPARE_SCRIPT}")
        return
    subprocess.run(
        [sys.executable, str(COMPARE_SCRIPT)],
        env={**__import__("os").environ,
             "PADRE_EXPERIMENT_NAME": experiment_name},
        check=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _run_learned_evaluation(args) -> None:
    """Evaluate a trained learned Byzantine agent and write CSVs for compare_byzantine.py.

    Calls train_learned_byzantine.py in --evaluate mode as a subprocess so
    it runs in the same Python environment without circular-import issues.
    """
    import glob as _glob
    import os as _os

    eval_config = args.learned_eval_config or str(
        Path("src/examples/evaluate/example_obs_manipulation_eval_config.yaml")
    )
    log_dir = f"{args.results_base}/{args.name}"

    # Derive agent name from the honest model glob (use most-recently-modified .zip).
    matches = sorted(
        _glob.glob(args.model, recursive=True),
        key=_os.path.getmtime,
    )
    agent_name = Path(matches[-1]).stem if matches else "learned_agent"

    n_byz = args.n_byz if args.n_byz is not None else 2

    cmd = [
        sys.executable,
        str(Path("src/Utils/train_learned_byzantine.py").resolve()),
        "--evaluate",
        "--honest-model", args.model,
        "--byz-model", args.learned_byz_model,
        "--eval-config", eval_config,
        "--n-byz", str(n_byz),
        "--max-offset", str(args.learned_max_offset),
        "--obs-scope", args.learned_obs_scope,
        "--eps", str(args.eps),
        "--ep-len", str(args.ep_len),
        "--log-dir", log_dir,
        "--agent-name", agent_name,
    ]

    print("\nEvaluating learned Byzantine agent → learned_obs_user_dir …")
    ret = subprocess.run(cmd).returncode
    status = "done" if ret == 0 else f"FAILED (exit {ret})"
    print(f"  learned_obs_user_dir: {status} → {log_dir}/learned_obs_user_dir/")


def main() -> None:
    # Fast path for subprocess workers — must happen before argparse so that the
    # required --model/--name flags don't cause a parse error.
    if "--_run-single" in sys.argv:
        yaml_path = sys.argv[sys.argv.index("--_run-single") + 1]
        sys.path.insert(0, str(Path("src").resolve()))
        from evaluate_MANETAgents import evaluate_from_config  # noqa: PLC0415
        evaluate_from_config(yaml_path)
        return

    parser = argparse.ArgumentParser(
        description="Run a Byzantine observation-manipulation sweep for a PADRE model."
    )
    parser.add_argument(
        "--model", required=True,
        help='Glob pattern for trained model checkpoints, e.g. "out/models/run/**"',
    )
    parser.add_argument(
        "--name", required=True,
        help='Experiment name used for output directories, e.g. "25_03"',
    )
    parser.add_argument(
        "--results-base", default="out/evaluations",
        help="Base directory for evaluation outputs (default: out/evaluations). "
             "Set to an absolute net_scratch path to write there directly.",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip running compare_byzantine.py after evaluations complete.",
    )
    parser.add_argument(
        "--eps", type=int, default=30,
        help="Number of evaluation episodes per condition (default: 30).",
    )
    parser.add_argument(
        "--ep-len", type=int, default=1000,
        help="Episode length in steps (default: 1000).",
    )
    parser.add_argument(
        "--magnitude", type=float, default=None,
        help=(
            "Attack magnitude (default: use per-condition defaults). "
            "Scales the primary attack parameter for each obs-manipulation type. "
            "See module docstring for the per-attack mapping."
        ),
    )
    parser.add_argument(
        "--n-byz", type=int, default=None,
        help=(
            "Override number of Byzantine nodes for all attack conditions "
            "(no_byzantine baseline always stays at 0). Default: use per-condition value."
        ),
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Run all attack conditions concurrently as subprocesses.",
    )
    parser.add_argument(
        "--no-jammers", action="store_true",
        help="Disable physical jammers (numbOfJammers=0). Isolates Byzantine behaviour.",
    )
    parser.add_argument(
        "--obs-type", default=None,
        choices=["real", "full", "gossip1", "reliable"],
        help=(
            "Override the obs_type for all honest nodes in the sweep. "
            "Default None uses the value in BASE_EVAL_PARAMS ('real'). "
            "'gossip1': 1-hop BFS only — Byzantine lies reach direct neighbors only. "
            "'reliable': flooding + min-merge cache against capacity/demand inflation."
        ),
    )
    parser.add_argument(
        "--angle", type=float, default=None, metavar="DEGREES",
        help=(
            "Rotate the obs_user_dir offset vector by this angle (degrees). "
            "Default (None) uses angle=0°, i.e. offset=[magnitude, 0]. "
            "0°=east, 90°=north, 180°=west, 270°=south. "
            "Has no effect on attack types other than obs_user_dir."
        ),
    )
    parser.add_argument(
        "--conditions", nargs="+", default=None, metavar="COND",
        help=(
            "Subset of conditions to run, e.g. --conditions no_byzantine obs_user_dir. "
            "Default: all conditions in SWEEP. 'no_byzantine' is always included."
        ),
    )
    parser.add_argument(
        "--learned-byz-model", default=None, metavar="PATH",
        help=(
            "Path to a trained learned Byzantine agent .zip (from train_learned_byzantine.py). "
            "When provided, an additional 'learned_obs_user_dir' condition is evaluated "
            "using ByzantineAttackerEnv after the regular heuristic sweep completes."
        ),
    )
    parser.add_argument(
        "--learned-max-offset", type=float, default=4.0,
        help="max_offset for ByzantineAttackerEnv when evaluating the learned agent (default: 4.0).",
    )
    parser.add_argument(
        "--learned-eval-config", default=None, metavar="YAML",
        help=(
            "Eval config YAML for the learned Byzantine evaluation. "
            "Defaults to src/examples/evaluate/example_obs_manipulation_eval_config.yaml."
        ),
    )
    parser.add_argument(
        "--n-envs", type=int, default=1,
        help=(
            "Number of parallel SubprocVecEnv workers per condition (default: 1 = sequential). "
            "Set to e.g. 4 to run 4 episodes concurrently and speed up each condition. "
            "Only single-agent RL evaluation is vectorised; baselines always run sequentially."
        ),
    )
    parser.add_argument(
        "--learned-obs-scope", default="full", choices=["full", "local", "heard", "coordinated"],
        help=(
            "obs_scope for the learned Byzantine agent — must match the scope used during training. "
            "'full' (default): oracle-info upper bound. "
            "'local': Stage A (own obs only). "
            "'heard': Stage B (own + in-range neighbor obs). "
            "'coordinated': Stage C (Stage B + covert channel to all Byzantine peers)."
        ),
    )
    args = parser.parse_args()

    sweep = dict(SWEEP)
    if args.conditions is not None:
        keep = set(args.conditions) | {"no_byzantine"}
        unknown = keep - set(sweep)
        if unknown:
            parser.error(f"Unknown conditions: {', '.join(sorted(unknown))}. "
                         f"Valid: {', '.join(sweep)}")
        sweep = {k: v for k, v in sweep.items() if k in keep}

    n_conditions = len(sweep)
    mag_str = f"{args.magnitude:.3g}" if args.magnitude is not None else "default"
    byz_str = str(args.n_byz) if args.n_byz is not None else "per-condition"
    angle_str = f"{args.angle:.1f}°" if args.angle is not None else "0° (default)"
    print(f"\nByzantine sweep — experiment: '{args.name}'")
    print(f"Model pattern : {args.model}")
    print(f"Conditions    : {n_conditions} {list(sweep)}")
    print(f"Episodes/cond : {args.eps}  |  Episode length: {args.ep_len}  |  Magnitude: {mag_str}  |  Byzantine nodes: {byz_str}")
    print(f"Parallel      : {args.parallel}  |  Jammers: {'off' if args.no_jammers else 'on'}  |  Angle: {angle_str}\n")

    base_overrides: dict = {}
    if args.no_jammers:
        base_overrides["numbOfJammers"] = 0
        base_overrides["jammer_ep_steps"] = 0
    if args.obs_type is not None:
        base_overrides["obs_type"] = args.obs_type

    configs = {
        attack_name: _build_config(
            model_pattern=args.model,
            attack_name=attack_name,
            overrides={**overrides, **base_overrides},
            experiment_name=args.name,
            n_eps=args.eps,
            ep_len=args.ep_len,
            magnitude=args.magnitude,
            n_byz=args.n_byz,
            angle=args.angle,
            n_envs=args.n_envs,
            results_base=args.results_base,
        )
        for attack_name, overrides in sweep.items()
    }

    if args.parallel:
        procs: list[tuple[subprocess.Popen, str, str]] = []
        for attack_name, config in configs.items():
            print(f"  spawning {attack_name} …")
            proc, tmp_path = _spawn_evaluation(config)
            procs.append((proc, tmp_path, attack_name))

        print(f"\nWaiting for {len(procs)} conditions to finish …")
        remaining = list(procs)
        while remaining:
            for item in remaining[:]:
                proc, tmp_path, attack_name = item
                ret = proc.poll()
                if ret is not None:
                    remaining.remove(item)
                    Path(tmp_path).unlink(missing_ok=True)
                    status = "done" if ret == 0 else f"FAILED (exit {ret})"
                    print(f"  {attack_name}: {status} → out/evaluations/{args.name}/{attack_name}/", flush=True)
            if remaining:
                time.sleep(5)
    else:
        for i, (attack_name, config) in enumerate(configs.items(), 1):
            print(f"[{i}/{n_conditions}] {attack_name} …")
            _run_evaluation(config)
            print(f"  → out/evaluations/{args.name}/{attack_name}/")


    print("\nAll evaluations complete.")

    if args.learned_byz_model:
        _run_learned_evaluation(args)

    if not args.no_plot:
        print(f"\nRunning compare_byzantine.py for experiment '{args.name}' …")
        _run_compare(args.name)
        print("Done.")


if __name__ == "__main__":
    main()
