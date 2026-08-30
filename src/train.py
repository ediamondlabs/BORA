import argparse
from train_MANETAgent import train_or_retrain


def main():
    parser = argparse.ArgumentParser(description="Train or retrain the agent.")
    parser.add_argument(
        "path", type=str, help="Path to the folder containing the model."
    )
    parser.add_argument(
        "--specific_model_step", type=str, help="Specific model step to load."
    )
    # Storage params
    parser.add_argument("--model_dir", type=str, help="Directory to save the model.")
    parser.add_argument("--log_dir", type=str, help="Directory to save the logs.")
    parser.add_argument("--log_name", type=str, help="Name of the log file.")
    # Model
    parser.add_argument("--RL_model_name", type=str, help="Name of the RL model.")
    # Environment
    parser.add_argument("--env", type=str, help="Environment name.")
    parser.add_argument(
        "--render", action="store_true", help="Whether to render the environment."
    )
    # Environment parameters
    parser.add_argument(
        "--numbOfNodes", type=int, help="Number of nodes in the environment."
    )
    parser.add_argument(
        "--numbOfJammers",
        type=int,
        nargs="+",
        help="Number of jammers in the environment (single value or range).",
    )
    parser.add_argument(
        "--numbOfUsers",
        type=int,
        nargs="+",
        help="Number of users in the environment (single value or range).",
    )
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument(
        "--networkConnectedAtStart",
        action="store_true",
        help="Whether the network is connected at the start.",
    )
    parser.add_argument(
        "--jammersSpawnNextToUsers",
        action="store_true",
        help="Whether jammers spawn next to users.",
    )
    parser.add_argument(
        "--allow_early_ep_finish",
        action="store_true",
        help="Allow early episode finish.",
    )
    parser.add_argument(
        "--step_size", type=int, help="Maximum step size of MANET node."
    )
    # Attacker model
    parser.add_argument("--attackerModel", type=str, help="Attacker model name.")
    parser.add_argument(
        "--steps_till_jammer_active",
        type=int,
        help="Steps till jammer becomes active.",
    )
    parser.add_argument(
        "--step_jammers_start_moving",
        type=int,
        help="Step when jammers start moving.",
    )
    # Byzantine params
    parser.add_argument(
        "--numbOfByzantineNodes", type=int, help="Number of Byzantine attacker nodes."
    )
    parser.add_argument(
        "--byzantine_drop_rate", type=float, help="Drop rate for Byzantine nodes."
    )
    parser.add_argument(
        "--byzantine_attack_type", type=str, help="Byzantine attack type (greyhole, sinkhole, selective_jamming, position_spoofing)."
    )
    parser.add_argument(
        "--byzantine_capacity_inflation_factor", type=float, help="Capacity inflation factor for sinkhole nodes."
    )
    parser.add_argument(
        "--byzantine_jamming_power_factor", type=float, help="Jamming power factor for selective_jamming nodes."
    )
    # Observation wrapper
    parser.add_argument("--obs_wrapper", type=str, help="Observation wrapper name.")
    parser.add_argument(
        "--obs_config",
        type=str,
        help="Configuration for the observation wrapper (JSON string).",
    )
    # Reward wrapper
    parser.add_argument("--rew_wrapper", type=str, help="Reward wrapper name.")
    # Training params
    parser.add_argument(
        "--timesteps", type=int, help="Number of timesteps for training."
    )
    parser.add_argument("--ep_length", type=int, help="Episode length.")
    parser.add_argument("--save_freq", type=int, help="Frequency of saving the model.")
    parser.add_argument("--stat_size", type=int, help="Size of the statistics buffer.")
    parser.add_argument("--n_envs", type=int, help="Number of parallel environments for training.")

    args = parser.parse_args()

    # Convert JSON string to dictionary for obs_config if provided
    if args.obs_config:
        import json

        args.obs_config = json.loads(args.obs_config)

    # Handle range arguments for numbOfJammers and numbOfUsers
    if args.numbOfJammers and len(args.numbOfJammers) == 2:
        args.numbOfJammers = tuple(args.numbOfJammers)
    if args.numbOfUsers and len(args.numbOfUsers) == 2:
        args.numbOfUsers = tuple(args.numbOfUsers)

    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    train_or_retrain(**kwargs)


if __name__ == "__main__":
    main()
