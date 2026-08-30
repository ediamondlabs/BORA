# Operating sys libs
import os
from pathlib import Path
import re
import glob

# File types
import yaml

# For type declaration
from typing import Type, Union, Callable, Tuple, List

# To store models and logs in folder with date
from datetime import datetime

# Gymnasium
import gymnasium as gym

# Stable Baselines
import functools
from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

# Own Classes
import MANET.ObservationWrappers as ow
import MANET.RewardWrappers as rw
import MANET.Environments as envs
import MANET.Attackers as at
import MANET.Agents as ag
import MANET.ConditionStrategy as cs
from MANET.Routing import MCMF, MaxFlow

"""
Helper functions for train_MANETAgent.py and evaluate_MANETAgents.py
"""

MANET_ENV_CLASSES = {
    "Basic_MANETEnv": envs.Basic_MANETEnv,
    "Static_MANETEnv": envs.Static_MANETEnv,
    "Less_Static_MANETEnv": envs.Less_Static_MANETEnv,
    "Dynamic_MANETEnv": envs.Dynamic_MANETEnv,
}

ATTACKER_MODEL_CLASSES = {
    "Static_GreedyJammers": at.Static_GreedyJammers,
    "Moving_GreedyJammers": at.Moving_GreedyJammers,
    "Static_ClusterJammers": at.Static_ClusterJammers,
    "Moving_ClusterJammers": at.Moving_ClusterJammers,
    "Static_TrafficJammers": at.Static_TrafficJammers,
    "Moving_TrafficJammers": at.Moving_TrafficJammers,
}

BASELINE_AGENT_CLASSES = {
    "GreedyMaximizer": ag.GreedyMaximizer,
    "SpiralSearch": ag.SpiralSearch,
    "OutAndThrough": ag.OutAndThroughAgent,
    "RandomSampler": ag.RandomSampler,
    "RandomAgent": ag.RandomAgent,
}

CONDITION_STRATEGY_CLASSES = {
    "TrafficHosted": cs.TrafficHostedConditionStrategy,
    "Throughput": cs.ThroughputConditionStrategy,
    "SINR": cs.SINRConditionStrategy,
    "Signal": cs.SignalConditionStrategy,
    "Interference": cs.InterferenceConditionStrategy,
}

OBSERVATION_WRAPPER_CLASSES = {
    "FS_COW": ow.FS_COW,
    "VS_COW": ow.VS_COW,
    "FS_iCOW": ow.FS_iCOW,
    "VS_iCOW": ow.VS_iCOW,
}

BASE_WRAPPER_CLASSES = {
    "full": ow.Base_ObservationWrapper,
    "random": ow.Random_ObservationWrapper,
    "real": ow.Real_ObservationWrapper,
    "gossip1": ow.Real_ObservationWrapper,           # Real + gossip_hops=1
    "reliable": ow.ReliableBroadcast_ObservationWrapper,
}

REWARD_WRAPPER_CLASSES = {
    "MovementCost": rw.MovementCost,
    "LinearReward": rw.LinearReward,
    "SquaredReward": rw.SquaredReward,
}

ROUTING_FUNCTIONS = {
    "multi_commodity_max_flow_pulp_1": MCMF.multi_commodity_max_flow_pulp_1,
    "multi_commodity_max_flow_pulp_2": MCMF.multi_commodity_max_flow_pulp_2,
    "multi_commodity_max_flow_gurobi": MCMF.multi_commodity_max_flow_gurobi,
    "multi_commodity_max_flow_xpress": MCMF.multi_commodity_max_flow_xpress,
    "sequential_max_flow": MaxFlow.sequential_max_flow,
    "artifical_src_sink_max_flow": MaxFlow.artifical_src_sink_max_flow,
}

AVAILABLE_MODELS = ["PPO", "A2C", "SAC", "TD3", "DDPG"]

DEFAULT_CONFIG = {
    "model_dir": "out/models",
    "log_dir": "out/logs",
    "log_name": "",
    "RL_model_name": "PPO",
    "env": "Static_MANETEnv",
    "render": False,
    "numbOfNodes": 5,
    "numbOfJammers": 1,
    "numbOfUsers": 10,
    "seed": 9,
    "networkConnectedAtStart": False,
    "jammersSpawnNextToUsers": True,
    "allow_early_ep_finish": True,
    "routing_func": "sequential_max_flow",
    "attackerModel": "Static_ClusterJammers",
    "steps_till_jammer_active": 0,
    "step_jammers_start_moving": None,
    "step_size": None,
    "reset": None,
    "numbOfByzantineNodes": 0,
    "byzantine_drop_rate": 0.5,
    "byzantine_attack_type": "greyhole",
    "byzantine_pos_offset": None,
    "byzantine_capacity_inflation_factor": None,
    "byzantine_jamming_power_factor": None,
    "byzantine_obs_user_dir_offset": None,
    "byzantine_obs_capacity_factor": None,
    "byzantine_obs_demand_factor": None,
    "byzantine_obs_interference_factor": None,
    "byzantine_obs_noise_std": None,
    "byzantine_obs_capacity_deflate_factor": None,
    "byzantine_obs_demand_deflate_factor": None,
    "byzantine_obs_replay_delay": None,
    "byzantine_obs_magnitude_range": None,
    "obs_wrapper": "VS_iCOW",
    "obs_config": None,
    "obs_type": "full",
    "state_loss_rate": 0.0,
    "rew_wrapper": "SquaredReward",
    "timesteps": 30_000_000,
    "ep_length": 10_000,
    "save_freq": 500_000,
    "stat_size": 10,
    "n_envs": 1,
}


# * Save and load files
def create_dirs(*args: str) -> None:
    """
    Create directories if they do not exist.

    Args:
        args: Variable length argument list of directory paths.
    """
    for arg in args:
        os.makedirs(arg, exist_ok=True)


# Save config
def document_config(path: str, file_name: str, **config: dict) -> None:
    """
    Document any kind of configuration by saving it to a YAML file.

    Args:
        path (str): The directory path to save the configuration file.
        file_name (str): The name of the configuration file.
        **config (dict): Configuration dictionary.
    """
    if not file_name.endswith(".yaml"):
        file_name += ".yaml"

    file_path = os.path.join(path, file_name)

    create_dirs(path)
    # Write to models directory
    with open(file_path, "w") as f:
        yaml.dump(config, f)


def document_as_nested_config(path: str, file_name: str, **config: dict) -> None:
    """
    Save a configuration dictionary as a nested YAML structure.

    Args:
        path (str): The directory path to save the configuration file.
        file_name (str): The name of the configuration file.
        **config (dict): Configuration dictionary.
    """
    if not file_name.endswith(".yaml"):
        file_name += ".yaml"

    file_path = os.path.join(path, file_name)
    create_dirs(path)

    # Reverse flattening to nest the dictionary
    nested_config = {
        "mode": config.get("mode"),
        "storing_and_logging": {
            "model_path": config.get("model_path"),
            "log_path": config.get("log_path"),
            "tb_logname": config.get("tb_logname"),
        },
        "model": {
            "RL_model_name": config.get("RL_model_name"),
        },
        "env": {
            "name": config.get("env"),
            "render": config.get("render"),
            "parameters": {
                "numbOfNodes": config.get("numbOfNodes"),
                "numbOfJammers": config.get("numbOfJammers"),
                "numbOfUsers": config.get("numbOfUsers"),
                "seed": config.get("seed"),
                "networkConnectedAtStart": config.get("networkConnectedAtStart"),
                "jammersSpawnNextToUsers": config.get("jammersSpawnNextToUsers"),
                "allow_early_ep_finish": config.get("allow_early_ep_finish"),
                "routing_func": config.get("routing_func"),
                "step_size": config.get("step_size"),
                "numbOfByzantineNodes": config.get("numbOfByzantineNodes"),
                "byzantine_drop_rate": config.get("byzantine_drop_rate"),
                "byzantine_attack_type": config.get("byzantine_attack_type"),
                "byzantine_capacity_inflation_factor": config.get("byzantine_capacity_inflation_factor"),
                "byzantine_jamming_power_factor": config.get("byzantine_jamming_power_factor"),
                "byzantine_pos_offset": config.get("byzantine_pos_offset"),
                "byzantine_obs_user_dir_offset": config.get("byzantine_obs_user_dir_offset"),
                "byzantine_obs_capacity_factor": config.get("byzantine_obs_capacity_factor"),
                "byzantine_obs_demand_factor": config.get("byzantine_obs_demand_factor"),
                "byzantine_obs_interference_factor": config.get("byzantine_obs_interference_factor"),
                "byzantine_obs_noise_std": config.get("byzantine_obs_noise_std"),
                "byzantine_obs_capacity_deflate_factor": config.get("byzantine_obs_capacity_deflate_factor"),
                "byzantine_obs_demand_deflate_factor": config.get("byzantine_obs_demand_deflate_factor"),
                "byzantine_obs_replay_delay": config.get("byzantine_obs_replay_delay"),
                "byzantine_obs_magnitude_range": config.get("byzantine_obs_magnitude_range"),
            },
        },
        "attacker": {
            "model": config.get("attackerModel"),
            "steps_till_jammer_active": config.get("steps_till_jammer_active"),
            "step_jammers_start_moving": config.get("step_jammers_start_moving"),
        },
        "obs_wrapper": {
            "type": config.get("obs_wrapper"),
            "config": config.get("obs_config"),
            "obs_type": config.get("obs_type"),
            "state_loss_rate": config.get("state_loss_rate"),
        },
        "rew_wrapper": {
            "type": config.get("rew_wrapper"),
        },
        "training": {
            "timesteps": config.get("timesteps"),
            "ep_length": config.get("ep_length"),
            "save_freq": config.get("save_freq"),
            "stat_size": config.get("stat_size"),
            "n_envs": config.get("n_envs"),
            "reset": config.get("reset"),
        },
        "loading_model": {
            "dir_of_agent_to_load": config.get("dir_of_agent_to_load"),
            "steps_of_agent_to_load": config.get("steps_of_agent_to_load"),
        },
    }

    def clean_dict(d):
        """Recursively remove None, empty dicts, and empty lists from a nested dictionary."""
        if isinstance(d, dict):
            cleaned = {k: clean_dict(v) for k, v in d.items()}  # Recurse first
            return {
                k: v
                for k, v in cleaned.items()
                if v is not None and v != {} and v != []
            } or None
        elif isinstance(d, tuple):  # Convert tuples to lists
            return list(d)
        return d

    nested_config = clean_dict(nested_config)

    # Save to YAML file
    with open(file_path, "w") as f:
        yaml.dump(nested_config, f)


# Load Config
def loadConfig(config_dir: str, config_name: str) -> dict:
    """
    Load the training configuration from a YAML file.

    Args:
        config_dir (str): Path to the directory containing the configuration file.
        config_name (str): Name of the configuration file.

    Returns:
        dict: Loaded configuration.

    Raises:
        FileNotFoundError: If the configuration file is not found.
    """
    config_path = os.path.join(config_dir, config_name)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if is_nested_config(config):
        config = loadNestedConfig(config_dir, config_name)
    return config


def loadNestedConfig(config_dir: str, config_name: str) -> dict:
    """
    Load and transform the stacked training configurations from a YAML file.

    Args:
        config_dir (str): Path to the directory containing the configuration file.
        config_name (str): Name of the configuration file.

    Returns:
        dict: Transformed configuration dictionary.
    """
    config_path = os.path.join(config_dir, config_name)

    with open(config_path, "r") as f:
        config: dict = yaml.safe_load(f)

    transformed_config = {
        "mode": config.get("mode"),
        # * Storing and logging nesting
        "model_dir": config.get("storing_and_logging", {}).get("model_dir"),
        "log_dir": config.get("storing_and_logging", {}).get("log_dir"),
        "log_name": config.get("storing_and_logging", {}).get("log_name"),
        "model_path": config.get("storing_and_logging", {}).get("model_path"),
        "log_path": config.get("storing_and_logging", {}).get("log_path"),
        "tb_logname": config.get("storing_and_logging", {}).get("tb_logname"),
        # * model
        "RL_model_name": config.get("model", {}).get("RL_model_name"),
        # * Enviorment
        "env": config.get("env", {}).get("name"),
        "render": config.get("env", {}).get("render"),
        # parameters
        "numbOfNodes": config.get("env", {}).get("parameters", {}).get("numbOfNodes"),
        "numbOfJammers": config.get("env", {})
        .get("parameters", {})
        .get("numbOfJammers"),
        "numbOfUsers": config.get("env", {}).get("parameters", {}).get("numbOfUsers"),
        "seed": config.get("env", {}).get("parameters", {}).get("seed"),
        "networkConnectedAtStart": config.get("env", {})
        .get("parameters", {})
        .get("networkConnectedAtStart"),
        "jammersSpawnNextToUsers": config.get("env", {})
        .get("parameters", {})
        .get("jammersSpawnNextToUsers"),
        "allow_early_ep_finish": config.get("env", {})
        .get("parameters", {})
        .get("allow_early_ep_finish"),
        "routing_func": config.get("env", {}).get("parameters", {}).get("routing_func"),
        "step_size": config.get("env", {}).get("parameters", {}).get("step_size"),
        "numbOfByzantineNodes": config.get("env", {}).get("parameters", {}).get("numbOfByzantineNodes"),
        "byzantine_drop_rate": config.get("env", {}).get("parameters", {}).get("byzantine_drop_rate"),
        "byzantine_attack_type": config.get("env", {}).get("parameters", {}).get("byzantine_attack_type"),
        "byzantine_capacity_inflation_factor": config.get("env", {}).get("parameters", {}).get("byzantine_capacity_inflation_factor"),
        "byzantine_jamming_power_factor": config.get("env", {}).get("parameters", {}).get("byzantine_jamming_power_factor"),
        "byzantine_pos_offset": config.get("env", {}).get("parameters", {}).get("byzantine_pos_offset"),
        "byzantine_obs_user_dir_offset": config.get("env", {}).get("parameters", {}).get("byzantine_obs_user_dir_offset"),
        "byzantine_obs_capacity_factor": config.get("env", {}).get("parameters", {}).get("byzantine_obs_capacity_factor"),
        "byzantine_obs_demand_factor": config.get("env", {}).get("parameters", {}).get("byzantine_obs_demand_factor"),
        "byzantine_obs_interference_factor": config.get("env", {}).get("parameters", {}).get("byzantine_obs_interference_factor"),
        "byzantine_obs_noise_std": config.get("env", {}).get("parameters", {}).get("byzantine_obs_noise_std"),
        "byzantine_obs_capacity_deflate_factor": config.get("env", {}).get("parameters", {}).get("byzantine_obs_capacity_deflate_factor"),
        "byzantine_obs_demand_deflate_factor": config.get("env", {}).get("parameters", {}).get("byzantine_obs_demand_deflate_factor"),
        "byzantine_obs_replay_delay": config.get("env", {}).get("parameters", {}).get("byzantine_obs_replay_delay"),
        "byzantine_obs_magnitude_range": config.get("env", {}).get("parameters", {}).get("byzantine_obs_magnitude_range"),
        # * Attacker
        "attackerModel": config.get("attacker", {}).get("model"),
        "steps_till_jammer_active": config.get("attacker", {}).get(
            "steps_till_jammer_active"
        ),
        "step_jammers_start_moving": config.get("attacker", {}).get(
            "step_jammers_start_moving"
        ),
        # * Obswrapper
        "obs_wrapper": config.get("obs_wrapper", {}).get("type"),
        "obs_config": config.get("obs_wrapper", {}).get("config"),
        "obs_type": config.get("obs_wrapper", {}).get("obs_type"),
        "state_loss_rate": config.get("obs_wrapper", {}).get("state_loss_rate"),
        # * Reward wrapper
        "rew_wrapper": config.get("rew_wrapper", {}).get("type"),
        "timesteps": config.get("training", {}).get("timesteps"),
        # * Training parameters
        "ep_length": config.get("training", {}).get("ep_length"),
        "save_freq": config.get("training", {}).get("save_freq"),
        "stat_size": config.get("training", {}).get("stat_size"),
        "n_envs": config.get("training", {}).get("n_envs"),
        "reset": config.get("training", {}).get("reset"),
        # * Retrain parameters
        "dir_of_agent_to_load": config.get("loading_model", {}).get(
            "dir_of_agent_to_load"
        ),
        "steps_of_agent_to_load": config.get("loading_model", {}).get(
            "steps_of_agent_to_load"
        ),
    }

    # Remove keys with None values
    transformed_config = {k: v for k, v in transformed_config.items() if v is not None}

    return transformed_config


def is_nested_config(config: dict) -> bool:
    """
    Check whether a given configuration dictionary follows the deeply nested format or a flatter format.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        bool: True if the config is deeply nested, False if it is flat.
    """
    # Deeply nested configs have subkeys under 'env', 'attacker', etc.
    nested_indicators = [
        isinstance(config.get("env"), dict)
        and isinstance(config["env"].get("parameters"), dict),
        isinstance(config.get("storing_and_logging"), dict),
        isinstance(config.get("attacker"), dict),
        isinstance(config.get("training"), dict)
        and isinstance(config["training"].get("timesteps"), int),
    ]

    return any(nested_indicators)


def load_eval_config(config_dir: str, config_name: str) -> dict:
    """Load params in to a uniform dict, from an evaluation configuration as a .yaml file.

    Args:
        config_dir (str): Directory, where the .yaml file is located
        config_name (str): Name of the config .yaml file.

    Returns:
        dict: Uniform dictionary, with strings translated to classes where needed.
    """
    config_path = os.path.join(config_dir, config_name)

    with open(config_path, "r") as f:
        config: dict = yaml.safe_load(f)

    env_constr = config.get("env_constr")
    if env_constr is not None:
        env_constr = getEnvironmentFromName(env_constr)
        config["env_constr"] = env_constr
    if "baselines_with_cs" in config.keys():
        baselines_with_cs: list = config.get("baselines_with_cs", [])
        new_base_with_cs = []
        for base_and_cs in baselines_with_cs:
            base_and_cs: dict
            baseline = getBaselineFromName(base_and_cs.get("baseline"))
            condition_strat = getConditionStrategyFromName(base_and_cs.get("cs"))
            threshold = base_and_cs.get("threshold")
            base_and_cs = (baseline, condition_strat(threshold))
            new_base_with_cs.append(base_and_cs)

        config["baselines_with_cs"] = new_base_with_cs

    return config


# Check certain config files
def hasConfigFiles(folder_path: str) -> bool:
    """
    Check if the given folder contains any configuration files (.yaml).

    Args:
        folder_path (str): The directory path to check for configuration files.

    Returns:
        bool: True if any .yaml configuration files are found, False otherwise.
    """
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".yaml"):
            return True
    return False


def loadConfigFileNames(path: str) -> list:
    """
    List all config file names found in a path.

    Args:
        path (str): The directory path to search for config files.

    Returns:
        list: A list of config file names.
    """
    config_files = []
    for file_name in os.listdir(path):
        if file_name.endswith(".yaml") and (
            "training_config" in file_name or "retraining_config" in file_name
        ):
            config_files.append(file_name)
    return config_files


def hasSpecificConfigFile(folder_path: str, config_name: str) -> bool:
    """
    Check if the given folder contains a specific configuration file.

    Args:
        folder_path (str): The directory path to check for the configuration file.
        config_name (str): The name of the configuration file to check for.

    Returns:
        bool: True if the specific configuration file is found, False otherwise.
    """
    config_file = (
        f"{config_name}.yaml" if not config_name.endswith(".yaml") else config_name
    )
    return config_file in os.listdir(folder_path)


def getMostRecentConfigFile(folder_path: str) -> str:
    """
    Get the path of the most recently created configuration file (.yaml) in the specified folder.

    Args:
        folder_path (str): The directory path to search for configuration files.

    Returns:
        str: Path to the most recently created configuration file.

    Raises:
        FileNotFoundError: If no configuration files are found in the directory.
    """
    config_files = [f for f in os.listdir(folder_path) if f.endswith(".yaml")]

    if not config_files:
        raise FileNotFoundError("No configuration files found in the directory.")

    config_files = [os.path.join(folder_path, f) for f in config_files]
    most_recent_file = max(config_files, key=os.path.getctime)

    file_name = os.path.basename(most_recent_file)
    return file_name


# * get config path of agent
def getPathOfAgent(
    dir_of_agent_to_load: str, steps_of_agent_to_load: int = None
) -> str:
    """
    Get the path of the agent file to load.

    Args:
        dir_of_agent_to_load (str): Directory containing the agent files.
        steps_of_agent_to_load (int, optional): Specific steps of the agent to load. If None, the latest agent will be loaded.

    Returns:
        str: Path to the agent file.
    """
    if steps_of_agent_to_load is None:
        agent_file = getFileNameOfLatestAgent(agent_folder_path=dir_of_agent_to_load)
    else:
        agent_file = getSpecificAgentFileName(
            dir_of_agent_to_load=dir_of_agent_to_load,
            steps_of_agent_to_load=steps_of_agent_to_load,
        )

    path_of_agent_to_load: str = os.path.join(dir_of_agent_to_load, agent_file)
    path_of_agent_to_load = convertPathsToPyPath(path_of_agent_to_load)[0]

    return path_of_agent_to_load


# Get file names
def getFileNameOfLatestAgent(agent_folder_path: str) -> str:
    """
    Load the agent with the highest steps number from the specified directory.

    Args:
        agent_folder_path (str): Directory containing the agent files.

    Returns:
        str: Name of the agent file with the highest steps number.

    Raises:
        FileNotFoundError: If no agent files are found in the directory.
    """
    # load latest agent
    agent_files = os.listdir(agent_folder_path)
    agent_files = [f for f in agent_files if f.endswith(".zip")]

    max_steps = -1
    latest_agent_file = None

    for agent_file in agent_files:
        try:
            steps = getStepsFromFileName(agent_file)
            if steps > max_steps:
                max_steps = steps
                latest_agent_file = agent_file
        except ValueError:
            continue

    if latest_agent_file is None:
        raise FileNotFoundError("No agent files found in the directory.")

    return latest_agent_file


def getSpecificAgentFileName(
    dir_of_agent_to_load: str, steps_of_agent_to_load: int
) -> str:
    """
    Load the specified agent file based on the steps number.

    Args:
        dir_of_agent_to_load (str): Directory containing the agent files.
        steps_of_agent_to_load (int): Specific steps of the agent to load.

    Returns:
        str: Name of the specified agent file.

    Raises:
        FileNotFoundError: If the specified agent file is not found in the directory.
    """
    # load specified agent
    specified_agent_file: str = f"rl_model_{int(steps_of_agent_to_load)}_steps.zip"
    full_path: str = os.path.join(dir_of_agent_to_load, specified_agent_file)
    # convert to use forward slashes
    normed_full_path = convertPathsToPyPath(full_path)[0]
    if not os.path.exists(normed_full_path):
        raise FileNotFoundError(
            f"Specified agent {specified_agent_file} in {dir_of_agent_to_load} not found."
        )

    return specified_agent_file


# Find agents matching pattern
def findAgentsMatchingPattern(*patterns: str) -> list:
    """Given paths in pattern form, find all paths of agent files that exist, which match the pattern. If agent file
    is specified load that, if not load the latest

    Returns:
        list: agent paths found, that match the patterns
    """
    patterns = convertPathsToPyPath(*patterns)

    agent_paths = []
    dir_patterns = []

    for pattern in patterns:
        # Resolve pattern to matching paths
        pattern = (
            escape_glob_literal_brackets(pattern)
            if "[" in pattern or "]" in pattern
            else pattern
        )

        if pattern.endswith(".zip"):
            matched_paths = glob.glob(pattern)
            agent_paths.extend(matched_paths)
        else:
            dir_patterns.append(pattern)

    for pattern in dir_patterns:
        # Resolve pattern to matching paths
        matched_paths = glob.glob(pattern)
        dir_paths = set()
        for path in matched_paths:
            if os.path.isdir(path):
                dir_paths.add(path)
            elif path.endswith(".zip"):
                dir_paths.add(os.path.dirname(path))

        for dir_path in dir_paths:
            try:
                agent_file = getFileNameOfLatestAgent(dir_path)
                agent_paths.append(os.path.join(dir_path, agent_file))
            except FileNotFoundError:
                print(f"Couldn't find an agent file for {os.path.dirname(dir_path)}")

    # ensure to return a list of paths which are all formatted the same
    converted_paths = convertPathsToPyPath(*agent_paths)
    if isinstance(converted_paths, str):
        agent_paths = [converted_paths]
    else:
        agent_paths = list(converted_paths)

    return sorted(agent_paths)


def findAgentFoldersPaths(*patterns: str) -> list:
    """Finds the path of all agent folders that match the pattern(s) given.

    Args:
        patterns (str): pattern or multiple patterns that describe an agents folder path

    Returns:
        list: list of the folder paths of an agent (where the individiual saved agent files are found)
    """
    patterns = convertPathsToPyPath(*patterns)
    # Set list as a set to prevent duplicates
    agent_folder_paths = set()

    def depthSearchTilZip(path: str) -> None:
        """Recursive depth search to find folder that holds the rl_model_*_steps.zip"""
        # If file found return the name of its directory
        if path.endswith(".zip"):
            dir_name = os.path.dirname(path)
            agent_folder_paths.add(dir_name)
        # Else go one deepr
        else:
            path = escape_glob_literal_brackets(path)
            pattern = os.path.join(path, "*")
            matched_paths = glob.glob(pattern)
            for path in matched_paths:
                depthSearchTilZip(path)

    for pattern in patterns:
        pattern = escape_glob_literal_brackets(pattern)
        matched_paths = glob.glob(pattern)

        for path in matched_paths:
            depthSearchTilZip(path)

    # (Optional): converts paths such that they surely are the same
    converted_paths = convertPathsToPyPath(*agent_folder_paths)

    if isinstance(converted_paths, str):
        converted_paths = [converted_paths]
    else:
        converted_paths = list(converted_paths)

    return converted_paths


def findAllAgentFiles(*patterns: str) -> list:
    """Finds all agent files specifed by a or mutliple pattern(s) which they must match

    Args:
        patterns (str): patterns that specify the agents' file paths.

    Returns:
        list: list of agents file paths
    """
    patterns = convertPathsToPyPath(*patterns)

    # Set agent files list as a set to prevent duplicates
    agent_files = set()

    def depthSearchTilZip(path: str) -> None:
        """Recursive depth search to find the rl_model_*_steps.zip files"""
        # If path points to a .zip file the add the path
        if path.endswith(".zip") and os.path.isfile(path):
            agent_files.add(path)

        # Go one depth further
        else:
            path = (
                escape_glob_literal_brackets(path)
                if "[" in path or "]" in path
                else path
            )
            pattern = os.path.join(path, "*")
            matched_paths = glob.glob(pattern)
            for path in matched_paths:
                depthSearchTilZip(path)

    # Search through patters
    for pattern in patterns:
        # Resolve pattern to matching paths
        pattern = (
            escape_glob_literal_brackets(pattern)
            if "[" in pattern or "]" in pattern
            else pattern
        )
        matched_paths = glob.glob(pattern)

        # Resolve path to .zip files
        for path in matched_paths:
            depthSearchTilZip(path)

    # (Optional): converts paths such that they surely are the same
    converted_paths = convertPathsToPyPath(*agent_files)

    if isinstance(converted_paths, str):
        converted_paths = [converted_paths]
    else:
        converted_paths = list(converted_paths)

    return converted_paths


def findAllAgentFilesWithInterval(*patterns: str, filterval: int = 1_000_000) -> list:
    """Given an interval and path patterns retrieves the agent paths which are divisible by the filterval

    Args:
        filterval (int, optional): Interval to divide by. Defaults to 1_000_000.

    Returns:
        list: Agent paths for interval
    """
    patterns = convertPathsToPyPath(*patterns)

    paths = findAllAgentFiles(*patterns)
    filtered_paths = []
    for path in paths:
        act_path = Path(path)
        steps = getStepsFromFileName(act_path.name)
        if steps % filterval == 0:
            filtered_paths.append(path)

    # Sort by steps (ascending)
    filtered_paths.sort(key=lambda p: getStepsFromFileName(Path(p).name))
    return filtered_paths


# Folder has agent
def hasAgentFiles(folder_path: str) -> bool:
    """
    Check if the given folder contains any agent files (rl_model_*_steps.zip).

    Args:
        folder_path (str): The directory path to check for agent files.

    Returns:
        bool: True if any agent files are found, False otherwise.
    """
    pattern = re.compile(r"rl_model_(\d+)_steps\.zip")
    for file_name in os.listdir(folder_path):
        if pattern.match(file_name):
            return True
    return False


# Get steps
def getStepsFromFileName(agent_file: str) -> int:
    """
    Extract the steps number from the agent file name.

    Args:
        agent_file (str): The agent file name.

    Returns:
        int: The steps number extracted from the file name.

    Raises:
        ValueError: If the file name does not match the expected pattern.
    """
    pattern = re.compile(r"rl_model_(\d+)_steps\.zip")
    match = pattern.match(agent_file)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"File name {agent_file} does not match the expected pattern.")


# * Load RL-model
def get_available_models() -> list:
    """
    Get the list of available RL models.

    Returns:
        list: List of available RL models.
    """
    return AVAILABLE_MODELS


def get_model(
    model_name: str,
    env: gym.Env,
    stats_window_size: int = 1,
    log_dir: str = None,
    seed: int = None,
) -> BaseAlgorithm:
    """
    Get the specified RL model.

    Args:
        model_name (str): Name of the RL model.
        env (gym.Env): The environment to train the model in.
        stats_window_size (int, optional): Size of the statistics window. Defaults to 1.
        log_dir (str, optional): Directory to save the logs. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        (BaseAlgorithm): The specified RL model.

    Raises:
        ValueError: If the model name is unknown.
    """
    if model_name == "PPO":
        return PPO(
            "MlpPolicy",
            env,
            verbose=0,
            ent_coef=0.01,
            gamma=0.99,
            tensorboard_log=log_dir,
            stats_window_size=stats_window_size,
            seed=seed,
        )
    elif model_name == "A2C":
        return A2C(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log=log_dir,
            stats_window_size=stats_window_size,
            seed=seed,
        )
    elif model_name == "SAC":
        return SAC(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log=log_dir,
            stats_window_size=stats_window_size,
            seed=seed,
        )
    elif model_name == "TD3":
        return TD3(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log=log_dir,
            stats_window_size=stats_window_size,
            seed=seed,
        )
    elif model_name == "DDPG":
        return DDPG("MlpPolicy", env, verbose=0, tensorboard_log=log_dir, seed=seed)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# * Parallel vectorised envs
def _env_factory(**kwargs) -> gym.Env:
    """Top-level factory required so functools.partial objects are picklable by SubprocVecEnv."""
    return createAndWrapEnv(**kwargs)


def make_vec_env(
    n_envs: int,
    env_constructor: Type[envs.Basic_MANETEnv],
    rew_wrapper: Type[gym.RewardWrapper],
    obs_wrapper: Type[gym.ObservationWrapper],
    numbOfNodes: int,
    numbOfJammers: Union[int, Tuple[int, int]],
    numbOfUsers: Union[int, Tuple[int, int]],
    seed: int = None,
    networkConnectedAtStart: bool = False,
    jammersSpawnNextToUsers: bool = False,
    attackerModel=at.Static_GreedyJammers,
    allow_early_ep_finish: bool = False,
    ep_length: int = 20_000,
    steps_till_jammer_active: int = 0,
    obs_config: dict = None,
    routing_func: Callable = MaxFlow.sequential_max_flow,
    step_size: int = 1,
    step_jammers_start_moving: int = 0,
    numbOfByzantineNodes: int = 0,
    byzantine_drop_rate: float = 0.5,
    byzantine_attack_type: Union[str, List[str]] = "greyhole",
    byzantine_capacity_inflation_factor: float = 3.0,
    byzantine_jamming_power_factor: float = 1.0,
    byzantine_pos_offset=None,
    byzantine_obs_user_dir_offset=None,
    byzantine_obs_capacity_factor: float = 1.0,
    byzantine_obs_demand_factor: float = 1.0,
    byzantine_obs_interference_factor: float = 1.0,
    byzantine_obs_noise_std: float = 0.0,
    byzantine_obs_capacity_deflate_factor: float = 1.0,
    byzantine_obs_demand_deflate_factor: float = 1.0,
    byzantine_obs_replay_delay: int = 0,
    byzantine_obs_magnitude_range=None,
    obs_type: str = "full",
    state_loss_rate: float = 0.0,
) -> SubprocVecEnv:
    """Create n_envs parallel environments using SubprocVecEnv.

    Each environment receives a unique seed offset (seed+i) so episodes are independent.
    Rendering is always disabled for parallel envs.

    Args:
        n_envs (int): Number of parallel environments.
        All other args match createAndWrapEnv.

    Returns:
        SubprocVecEnv: Vectorised environment with n_envs workers.
    """
    fns = [
        functools.partial(
            _env_factory,
            env_constructor=env_constructor,
            rew_wrapper=rew_wrapper,
            obs_wrapper=obs_wrapper,
            numbOfNodes=numbOfNodes,
            numbOfJammers=numbOfJammers,
            numbOfUsers=numbOfUsers,
            render_mode=None,
            seed=(seed + i) if seed is not None else i,
            networkConnectedAtStart=networkConnectedAtStart,
            jammersSpawnNextToUsers=jammersSpawnNextToUsers,
            attackerModel=attackerModel,
            allow_early_ep_finish=allow_early_ep_finish,
            ep_length=ep_length,
            steps_till_jammer_active=steps_till_jammer_active,
            obs_config=obs_config,
            obs_type=obs_type,
            state_loss_rate=state_loss_rate,
            routing_func=routing_func,
            step_size=step_size,
            step_jammers_start_moving=step_jammers_start_moving,
            numbOfByzantineNodes=numbOfByzantineNodes,
            byzantine_drop_rate=byzantine_drop_rate,
            byzantine_attack_type=byzantine_attack_type,
            byzantine_capacity_inflation_factor=byzantine_capacity_inflation_factor,
            byzantine_jamming_power_factor=byzantine_jamming_power_factor,
            byzantine_pos_offset=byzantine_pos_offset,
            byzantine_obs_user_dir_offset=byzantine_obs_user_dir_offset,
            byzantine_obs_capacity_factor=byzantine_obs_capacity_factor,
            byzantine_obs_demand_factor=byzantine_obs_demand_factor,
            byzantine_obs_interference_factor=byzantine_obs_interference_factor,
            byzantine_obs_noise_std=byzantine_obs_noise_std,
            byzantine_obs_capacity_deflate_factor=byzantine_obs_capacity_deflate_factor,
            byzantine_obs_demand_deflate_factor=byzantine_obs_demand_deflate_factor,
            byzantine_obs_replay_delay=byzantine_obs_replay_delay,
            byzantine_obs_magnitude_range=byzantine_obs_magnitude_range,
        )
        for i in range(n_envs)
    ]
    return SubprocVecEnv(fns)


# * Creating and wrapping envs
# From config
def createAndWrapEnvFromConfig(
    config: dict,
    env_constructor: Type[envs.Basic_MANETEnv] = None,
    rew_wrapper: gym.RewardWrapper = None,
    obs_wrapper: gym.ObservationWrapper = None,
    numbOfNodes: int = None,
    numbOfJammers: int = None,
    numbOfUsers: int = None,
    render_mode: str = None,
    seed: int = None,
    networkConnectedAtStart: bool = None,
    jammersSpawnNextToUsers: bool = None,
    attackerModel=at.Static_GreedyJammers,
    allow_early_ep_finish: bool = None,
    ep_length: int = None,
    steps_till_jammer_active: int = None,
    obs_config: dict = None,
    obs_type: str = None,
    state_loss_rate: float = None,
    node_idx: int = None,
    routing_func: Callable = None,
    step_size: int = None,
    step_jammers_start_moving: int = None,
    numbOfByzantineNodes: int = None,
    byzantine_drop_rate: float = None,
    byzantine_attack_type: Union[str, List[str]] = None,
    byzantine_capacity_inflation_factor: float = None,
    byzantine_jamming_power_factor: float = None,
    byzantine_pos_offset=None,
    byzantine_obs_user_dir_offset=None,
    byzantine_obs_capacity_factor: float = None,
    byzantine_obs_demand_factor: float = None,
    byzantine_obs_interference_factor: float = None,
    byzantine_obs_noise_std: float = None,
    byzantine_obs_capacity_deflate_factor: float = None,
    byzantine_obs_demand_deflate_factor: float = None,
    byzantine_obs_replay_delay: int = None,
    byzantine_obs_magnitude_range=None,
) -> gym.Env:
    """
    Creates and wraps an environment from a configuration dictionary, with optional overrides.

    Args:
        config (dict): Configuration dictionary containing environment and wrapper parameters.
        env_constructor (Type[envs.Basic_MANETEnv], optional): Environment constructor.
        rew_wrapper (gym.RewardWrapper, optional): The reward wrapper class.
        obs_wrapper (gym.ObservationWrapper, optional): The observation wrapper class.
        numbOfNodes (int, optional): Number of nodes in the environment.
        numbOfJammers (int, optional): Number of jammers in the environment.
        numbOfUsers (int, optional): Number of users in the environment.
        render_mode (str, optional): Whether to render the environment.
        seed (int, optional): Random number seed for the environment.
        networkConnectedAtStart (bool, optional): Whether nodes should start at a connected position.
        jammersSpawnNextToUsers (bool, optional): Whether the jammers should be initiated next to a user.
        attackerModel (optional): How the attacker determines its positioning.
        allow_early_ep_finish (bool, optional): Whether to allow early finishes before episode is complete.
        ep_length (int, optional): Length of the episode.
        steps_till_jammer_active (int, optional): Number of steps until the jammer turns active.
        obs_config (dict, optional): The observation configuration.
        obs_type (str, optional): _description_. Defaults to "full".
        state_loss_rate (float, optional): _description_. Defaults to 0.3.
        node_idx (int, optional): _description_. Defaults to None.
        routing_func (Callable): Function determining how the data is routed in the network.
        byzantine_capacity_inflation_factor (float, optional): Capacity multiplier for sinkhole nodes.
        byzantine_jamming_power_factor (float, optional): Interference multiplier for selective_jamming nodes.
        byzantine_pos_offset (array-like or None, optional): Position offset for position_spoofing nodes.
        byzantine_obs_user_dir_offset (array-like or None, optional): Direction offset for obs_user_dir nodes.
        byzantine_obs_capacity_factor (float, optional): Capacity scale factor for obs_capacity nodes.
        byzantine_obs_demand_factor (float, optional): Demand scale factor for obs_demand nodes.
        byzantine_obs_interference_factor (float, optional): Interference scale factor for obs_interference_lie nodes.
        byzantine_obs_noise_std (float, optional): Noise std dev for obs_noise nodes.

    Returns:
        gym.Env: The wrapped environment.
    """
    env = createEnvFromConfig(
        config,
        env_constructor=env_constructor,
        numbOfNodes=numbOfNodes,
        numbOfJammers=numbOfJammers,
        numbOfUsers=numbOfUsers,
        render_mode=render_mode,
        seed=seed,
        networkConnectedAtStart=networkConnectedAtStart,
        jammersSpawnNextToUsers=jammersSpawnNextToUsers,
        attackerModel=attackerModel,
        allow_early_ep_finish=allow_early_ep_finish,
        ep_length=ep_length,
        steps_till_jammer_active=steps_till_jammer_active,
        routing_func=routing_func,
        step_size=step_size,
        step_jammers_start_moving=step_jammers_start_moving,
        numbOfByzantineNodes=numbOfByzantineNodes,
        byzantine_drop_rate=byzantine_drop_rate,
        byzantine_attack_type=byzantine_attack_type,
        byzantine_capacity_inflation_factor=byzantine_capacity_inflation_factor,
        byzantine_jamming_power_factor=byzantine_jamming_power_factor,
        byzantine_pos_offset=byzantine_pos_offset,
        byzantine_obs_user_dir_offset=byzantine_obs_user_dir_offset,
        byzantine_obs_capacity_factor=byzantine_obs_capacity_factor,
        byzantine_obs_demand_factor=byzantine_obs_demand_factor,
        byzantine_obs_interference_factor=byzantine_obs_interference_factor,
        byzantine_obs_noise_std=byzantine_obs_noise_std,
        byzantine_obs_capacity_deflate_factor=byzantine_obs_capacity_deflate_factor,
        byzantine_obs_demand_deflate_factor=byzantine_obs_demand_deflate_factor,
        byzantine_obs_replay_delay=byzantine_obs_replay_delay,
        byzantine_obs_magnitude_range=byzantine_obs_magnitude_range,
    )
    wrapped_env = wrapEnvFromConfig(
        env,
        config,
        rew_wrapper,
        obs_wrapper,
        obs_config,
        obs_type,
        state_loss_rate,
        node_idx,
    )
    return wrapped_env


def createEnvFromConfig(
    config: dict,
    env_constructor: Type[envs.Basic_MANETEnv] = None,
    numbOfNodes: int = None,
    numbOfJammers: int = None,
    numbOfUsers: int = None,
    render_mode: str = None,
    seed: int = None,
    networkConnectedAtStart: bool = None,
    jammersSpawnNextToUsers: bool = None,
    attackerModel: Union[at.AttackerModel, List[at.AttackerModel]] = None,
    allow_early_ep_finish: bool = None,
    ep_length: int = None,
    steps_till_jammer_active: int = None,
    routing_func: Callable = None,
    step_size: int = None,
    step_jammers_start_moving: int = None,
    numbOfByzantineNodes: int = None,
    byzantine_drop_rate: float = None,
    byzantine_attack_type: Union[str, List[str]] = None,
    byzantine_capacity_inflation_factor: float = None,
    byzantine_jamming_power_factor: float = None,
    byzantine_pos_offset=None,
    byzantine_obs_user_dir_offset=None,
    byzantine_obs_capacity_factor: float = None,
    byzantine_obs_demand_factor: float = None,
    byzantine_obs_interference_factor: float = None,
    byzantine_obs_noise_std: float = None,
    byzantine_obs_capacity_deflate_factor: float = None,
    byzantine_obs_demand_deflate_factor: float = None,
    byzantine_obs_replay_delay: int = None,
    byzantine_obs_magnitude_range=None,
) -> envs.Basic_MANETEnv:
    """
    Creates an environment from a configuration dictionary, with optional overrides.

    Args:
        config (dict): Configuration dictionary containing environment parameters.
        env_constructor (Type[envs.Basic_MANETEnv], optional): Environment constructor.
        numbOfNodes (int, optional): Number of nodes in the environment.
        numbOfJammers (int, optional): Number of jammers in the environment.
        numbOfUsers (int, optional): Number of users in the environment.
        render_mode (str, optional): Whether to render the environment.
        seed (int, optional): Random number seed for the environment.
        networkConnectedAtStart (bool, optional): Whether nodes should start at a connected position.
        jammersSpawnNextToUsers (bool, optional): Whether the jammers should be initiated next to a user.
        attackerModel (optional): How the attacker determines its positioning.
        allow_early_ep_finish (bool, optional): Whether to allow early finishes before episode is complete.
        ep_length (int, optional): Length of the episode.
        steps_till_jammer_active (int, optional): Number of steps until the jammer turns active.
        routing_func (Callable): Function determining how the data is routed in the network.
        byzantine_capacity_inflation_factor (float, optional): Capacity multiplier for sinkhole nodes.
        byzantine_jamming_power_factor (float, optional): Interference multiplier for selective_jamming nodes.
        byzantine_pos_offset (array-like or None, optional): Position offset for position_spoofing nodes.
        byzantine_obs_user_dir_offset (array-like or None, optional): Direction offset for obs_user_dir nodes.
        byzantine_obs_capacity_factor (float, optional): Capacity scale factor for obs_capacity nodes.
        byzantine_obs_demand_factor (float, optional): Demand scale factor for obs_demand nodes.
        byzantine_obs_interference_factor (float, optional): Interference scale factor for obs_interference_lie nodes.
        byzantine_obs_noise_std (float, optional): Noise std dev for obs_noise nodes.

    Returns:
        envs.Basic_MANETEnv: The created environment.
    """
    env_constructor = (
        env_constructor
        if env_constructor is not None
        else getEnvironmentFromName(config["env"])
    )
    return createEnv(
        env_constructor=env_constructor,
        numbOfNodes=numbOfNodes if numbOfNodes is not None else config["numbOfNodes"],
        numbOfJammers=(
            numbOfJammers if numbOfJammers is not None else config["numbOfJammers"]
        ),
        numbOfUsers=numbOfUsers if numbOfUsers is not None else config["numbOfUsers"],
        render_mode=render_mode if render_mode is not None else config.get("render"),
        seed=seed if seed is not None else config.get("seed"),
        networkConnectedAtStart=(
            networkConnectedAtStart
            if networkConnectedAtStart is not None
            else config.get("networkConnectedAtStart")
        ),
        jammersSpawnNextToUsers=(
            jammersSpawnNextToUsers
            if jammersSpawnNextToUsers is not None
            else config.get("jammersSpawnNextToUsers")
        ),
        attackerModel=(
            attackerModel
            if attackerModel is not None
            else getAttackerModelFromName(config.get("attackerModel"))
        ),
        allow_early_ep_finish=(
            allow_early_ep_finish
            if allow_early_ep_finish is not None
            else config.get("allow_early_ep_finish")
        ),
        ep_length=ep_length if ep_length is not None else config.get("ep_length"),
        steps_till_jammer_active=(
            steps_till_jammer_active
            if steps_till_jammer_active is not None
            else config.get("steps_till_jammer_active")
        ),
        routing_func=(
            routing_func
            if routing_func is not None
            else getRoutingFunction(config.get("routing_func"))
        ),
        step_size=(
            step_size
            if step_size is not None
            else getRoutingFunction(config.get("step_size"))
        ),
        step_jammers_start_moving=(
            step_jammers_start_moving
            if step_jammers_start_moving is not None
            else getRoutingFunction(config.get("step_jammers_start_moving"))
        ),
        numbOfByzantineNodes=(
            numbOfByzantineNodes
            if numbOfByzantineNodes is not None
            else config.get("numbOfByzantineNodes", 0)
        ),
        byzantine_drop_rate=(
            byzantine_drop_rate
            if byzantine_drop_rate is not None
            else config.get("byzantine_drop_rate", 0.5)
        ),
        byzantine_attack_type=(
            byzantine_attack_type
            if byzantine_attack_type is not None
            else config.get("byzantine_attack_type", "greyhole")
        ),
        byzantine_capacity_inflation_factor=(
            byzantine_capacity_inflation_factor
            if byzantine_capacity_inflation_factor is not None
            else config.get("byzantine_capacity_inflation_factor", 3.0)
        ),
        byzantine_jamming_power_factor=(
            byzantine_jamming_power_factor
            if byzantine_jamming_power_factor is not None
            else config.get("byzantine_jamming_power_factor", 1.0)
        ),
        byzantine_pos_offset=(
            byzantine_pos_offset
            if byzantine_pos_offset is not None
            else config.get("byzantine_pos_offset")
        ),
        byzantine_obs_user_dir_offset=(
            byzantine_obs_user_dir_offset
            if byzantine_obs_user_dir_offset is not None
            else config.get("byzantine_obs_user_dir_offset")
        ),
        byzantine_obs_capacity_factor=(
            byzantine_obs_capacity_factor
            if byzantine_obs_capacity_factor is not None
            else config.get("byzantine_obs_capacity_factor", 1.0)
        ),
        byzantine_obs_demand_factor=(
            byzantine_obs_demand_factor
            if byzantine_obs_demand_factor is not None
            else config.get("byzantine_obs_demand_factor", 1.0)
        ),
        byzantine_obs_interference_factor=(
            byzantine_obs_interference_factor
            if byzantine_obs_interference_factor is not None
            else config.get("byzantine_obs_interference_factor", 1.0)
        ),
        byzantine_obs_noise_std=(
            byzantine_obs_noise_std
            if byzantine_obs_noise_std is not None
            else config.get("byzantine_obs_noise_std", 0.0)
        ),
        byzantine_obs_capacity_deflate_factor=(
            byzantine_obs_capacity_deflate_factor
            if byzantine_obs_capacity_deflate_factor is not None
            else config.get("byzantine_obs_capacity_deflate_factor", 1.0)
        ),
        byzantine_obs_demand_deflate_factor=(
            byzantine_obs_demand_deflate_factor
            if byzantine_obs_demand_deflate_factor is not None
            else config.get("byzantine_obs_demand_deflate_factor", 1.0)
        ),
        byzantine_obs_replay_delay=(
            int(byzantine_obs_replay_delay)
            if byzantine_obs_replay_delay is not None
            else int(config.get("byzantine_obs_replay_delay", 0))
        ),
        byzantine_obs_magnitude_range=(
            byzantine_obs_magnitude_range
            if byzantine_obs_magnitude_range is not None
            else config.get("byzantine_obs_magnitude_range")
        ),
    )


def wrapEnvFromConfig(
    env: gym.Env,
    config: dict,
    rew_wrapper: gym.RewardWrapper = None,
    obs_wrapper: gym.ObservationWrapper = None,
    obs_config: dict = None,
    obs_type: str = None,
    state_loss_rate: float = None,
    node_idx: int = None,
) -> gym.Env:
    """
    Wraps an environment from a configuration dictionary, with optional overrides.

    Args:
        env (gym.Env): The environment to wrap.
        config (dict): Configuration dictionary containing wrapper parameters.
        rew_wrapper (gym.RewardWrapper, optional): The reward wrapper class.
        obs_wrapper (gym.ObservationWrapper, optional): The observation wrapper class.
        obs_config (dict, optional): The observation configuration.
        obs_type (str, optional): _description_. Defaults to "full".
        state_loss_rate (float, optional): _description_. Defaults to 0.3.
        node_idx (int, optional): _description_. Defaults to None.

    Returns:
        gym.Env: The wrapped environment.
    """
    rew_wrapper = rew_wrapper or getRewardWrapperFromName(config["rew_wrapper"])
    obs_wrapper = obs_wrapper or getObservationWrapperFromName(config["obs_wrapper"])
    obs_config = obs_config or config.get("obs_config")
    return wrapEnv(
        env=env,
        rew_wrapper=rew_wrapper,
        obs_wrapper=obs_wrapper,
        obs_config=obs_config,
        obs_type=obs_type,
        state_loss_rate=state_loss_rate,
        node_idx=node_idx,
    )


# From params
def createAndWrapEnv(
    env_constructor: Type[envs.Basic_MANETEnv],
    rew_wrapper: Type[gym.RewardWrapper],
    obs_wrapper: Type[gym.ObservationWrapper],
    numbOfNodes: int,
    numbOfJammers: Union[int, Tuple[int, int]],
    numbOfUsers: Union[int, Tuple[int, int]],
    render_mode: str = None,
    seed: int = None,
    networkConnectedAtStart: bool = False,
    jammersSpawnNextToUsers: bool = False,
    attackerModel=at.Static_GreedyJammers,
    allow_early_ep_finish: bool = False,
    ep_length: int = 20_000,
    steps_till_jammer_active: int = 0,
    obs_config: dict = None,
    obs_type: str = "full",
    state_loss_rate: float = 0.3,
    node_idx: int = None,
    routing_func: Callable = MaxFlow.sequential_max_flow,
    step_size: int = 1,
    step_jammers_start_moving: int = 0,
    numbOfByzantineNodes: int = 0,
    byzantine_drop_rate: float = 0.5,
    byzantine_attack_type: Union[str, List[str]] = "greyhole",
    byzantine_capacity_inflation_factor: float = 3.0,
    byzantine_jamming_power_factor: float = 1.0,
    byzantine_pos_offset=None,
    byzantine_obs_user_dir_offset=None,
    byzantine_obs_capacity_factor: float = 1.0,
    byzantine_obs_demand_factor: float = 1.0,
    byzantine_obs_interference_factor: float = 1.0,
    byzantine_obs_noise_std: float = 0.0,
    byzantine_obs_capacity_deflate_factor: float = 1.0,
    byzantine_obs_demand_deflate_factor: float = 1.0,
    byzantine_obs_replay_delay: int = 0,
    byzantine_obs_magnitude_range=None,
) -> gym.Env:
    """Creates and wraps and environment.

    Args:
        env_constructor (Type[envs.Basic_MANETEnv]): environment constructor
        rew_wrapper (gym.RewardWrapper): The reward wrapper class.
        obs_wrapper (gym.ObservationWrapper): The observation wrapper class.
        numbOfNodes (int): number of nodes in the environment
        numbOfJammers (int): number of jammers in the environment
        numbOfUsers (int): number of users in the environmnet
        render_mode (str, optional): whether to render the environment render_mode = "plot" or not render_mode = None. Defaults to None.
        seed (int, optional): random number seed for the environmnet. Defaults to None.
        networkConnectedAtStart (bool, optional): whether nodes should start at a connected position or uniform randomly. Defaults to False.
        jammersSpawnNextToUsers (bool, optional): whether the jammers should be initiated next to a user or uniform randomly distributed. Defaults to False.
        attackerModel (_type_, optional): how the attacker determines its positionning. Defaults to at.Static_GreedyJammers.
        allow_early_ep_finish (bool, optional): whether to allow early finishes before episode is complete. Defaults to False.
        ep_length (int, optional): legnth of the episode. Defaults to 20_000.
        steps_till_jammer_active (int, optional): number of steps until the jammer turns active. Defaults to 0.
        obs_config (dict, optional): The observation configuration. Defaults to None.
        obs_type (str, optional): _description_. Defaults to "full".
        state_loss_rate (float, optional): _description_. Defaults to 0.3.
        node_idx (int, optional): _description_. Defaults to None.
        routing_func (Callable): Function determining how the data is routed in the network.
        step_size (int): The maximum step size for each nodes action.
        byzantine_capacity_inflation_factor (float): Capacity multiplier for sinkhole nodes.
        byzantine_jamming_power_factor (float): Interference multiplier for selective_jamming nodes.
        byzantine_pos_offset (array-like or None): Position offset for position_spoofing nodes.
        byzantine_obs_user_dir_offset (array-like or None): Direction offset for obs_user_dir nodes.
        byzantine_obs_capacity_factor (float): Capacity scale factor for obs_capacity nodes.
        byzantine_obs_demand_factor (float): Demand scale factor for obs_demand nodes.
        byzantine_obs_interference_factor (float): Interference scale factor for obs_interference_lie nodes.
        byzantine_obs_noise_std (float): Noise std dev for obs_noise nodes.

    Returns:
        gym.Env: The wrapped environment.
    """
    env = createEnv(
        env_constructor=env_constructor,
        numbOfNodes=numbOfNodes,
        numbOfJammers=numbOfJammers,
        numbOfUsers=numbOfUsers,
        render_mode=render_mode,
        seed=seed,
        networkConnectedAtStart=networkConnectedAtStart,
        jammersSpawnNextToUsers=jammersSpawnNextToUsers,
        attackerModel=attackerModel,
        allow_early_ep_finish=allow_early_ep_finish,
        ep_length=ep_length,
        steps_till_jammer_active=steps_till_jammer_active,
        routing_func=routing_func,
        step_size=step_size,
        step_jammers_start_moving=step_jammers_start_moving,
        numbOfByzantineNodes=numbOfByzantineNodes,
        byzantine_drop_rate=byzantine_drop_rate,
        byzantine_attack_type=byzantine_attack_type,
        byzantine_capacity_inflation_factor=byzantine_capacity_inflation_factor,
        byzantine_jamming_power_factor=byzantine_jamming_power_factor,
        byzantine_pos_offset=byzantine_pos_offset,
        byzantine_obs_user_dir_offset=byzantine_obs_user_dir_offset,
        byzantine_obs_capacity_factor=byzantine_obs_capacity_factor,
        byzantine_obs_demand_factor=byzantine_obs_demand_factor,
        byzantine_obs_interference_factor=byzantine_obs_interference_factor,
        byzantine_obs_noise_std=byzantine_obs_noise_std,
        byzantine_obs_capacity_deflate_factor=byzantine_obs_capacity_deflate_factor,
        byzantine_obs_demand_deflate_factor=byzantine_obs_demand_deflate_factor,
        byzantine_obs_replay_delay=byzantine_obs_replay_delay,
        byzantine_obs_magnitude_range=byzantine_obs_magnitude_range,
    )
    wrapped_env = wrapEnv(
        env, rew_wrapper, obs_wrapper, obs_config, obs_type, state_loss_rate, node_idx
    )

    return wrapped_env


def wrapEnv(
    env: gym.Env,
    rew_wrapper: Type[gym.RewardWrapper],
    obs_wrapper: Type[gym.ObservationWrapper],
    obs_config: dict = None,
    obs_type: str = "full",
    state_loss_rate: float = None,
    node_idx: int = None,
    use_monitor: bool = True,
) -> gym.Env:
    """Wrap the environment with the specified observation wrapper and reward wrapper.

    Args:
        env (gym.Env): The environment to wrap.
        rew_wrapper (gym.RewardWrapper): The reward wrapper class.
        obs_wrapper (gym.ObservationWrapper): The observation wrapper class.
        obs_config (dict, optional): The observation configuration. Defaults to None.
        obs_type (str, optional): _description_. Defaults to "full".
        state_loss_rate (float, optional): _description_. Defaults to 0.3.
        node_idx (int, optional): _description_. Defaults to None.
        use_monitor (bool, optional): Applies monitor wrapper if true.

    Returns:
        gym.Env: The wrapped environment.
    """
    # First unwrap environment to ensure no other wrappers have been applied
    unwrapped_env: envs.Basic_MANETEnv = env.unwrapped

    # Then wrap it up
    wrapped_env = wrapWithBaseWrapper(
        env=unwrapped_env,
        obs_config=obs_config,
        obs_type=obs_type,
        state_loss_rate=state_loss_rate,
        node_idx=node_idx,
    )

    wrapped_wrapped_env = obs_wrapper(wrapped_env)
    wrapped_wrapped_wrapped_env = rew_wrapper(wrapped_wrapped_env)
    wrapped_wrapped_wrapped_wrapped_env = (
        (Monitor(wrapped_wrapped_wrapped_env, allow_early_resets=True))
        if use_monitor
        else wrapped_wrapped_wrapped_env
    )

    return wrapped_wrapped_wrapped_wrapped_env


def createEnv(
    env_constructor: Type[envs.Basic_MANETEnv],
    numbOfNodes: int,
    numbOfJammers: Union[int, Tuple[int, int]],
    numbOfUsers: Union[int, Tuple[int, int]],
    render_mode: str = None,
    seed: int = None,
    networkConnectedAtStart: bool = False,
    jammersSpawnNextToUsers: bool = False,
    attackerModel=at.Static_GreedyJammers,
    allow_early_ep_finish: bool = False,
    ep_length: int = 20_000,
    steps_till_jammer_active: int = 0,
    routing_func: Callable = MaxFlow.sequential_max_flow,
    step_size: int = 2,
    step_jammers_start_moving: int = 0,
    numbOfByzantineNodes: int = 0,
    byzantine_drop_rate: float = 0.5,
    byzantine_attack_type: Union[str, List[str]] = "greyhole",
    byzantine_capacity_inflation_factor: float = 3.0,
    byzantine_jamming_power_factor: float = 1.0,
    byzantine_pos_offset=None,
    byzantine_obs_user_dir_offset=None,
    byzantine_obs_capacity_factor: float = 1.0,
    byzantine_obs_demand_factor: float = 1.0,
    byzantine_obs_interference_factor: float = 1.0,
    byzantine_obs_noise_std: float = 0.0,
    byzantine_obs_capacity_deflate_factor: float = 1.0,
    byzantine_obs_demand_deflate_factor: float = 1.0,
    byzantine_obs_replay_delay: int = 0,
    byzantine_obs_magnitude_range=None,
) -> envs.Basic_MANETEnv:
    """Creates an environment

    Args:
        env_constructor (Type[envs.Basic_MANETEnv]): environment constructor
        numbOfNodes (int): number of nodes in the environment
        numbOfJammers (int): number of jammers in the environment
        numbOfUsers (int): number of users in the environmnet
        render_mode (str, optional): whether to render the environment render_mode = "plot" or not render_mode = None. Defaults to None.
        seed (int, optional): random number seed for the environmnet. Defaults to None.
        networkConnectedAtStart (bool, optional): whether nodes should start at a connected position or uniform randomly. Defaults to False.
        jammersSpawnNextToUsers (bool, optional): whether the jammers should be initiated next to a user or uniform randomly distributed. Defaults to False.
        attackerModel (_type_, optional): how the attacker determines its positionning. Defaults to at.Static_GreedyJammers.
        allow_early_ep_finish (bool, optional): whether to allow early finishes before episode is complete. Defaults to False.
        ep_length (int, optional): legnth of the episode. Defaults to 20_000.
        steps_till_jammer_active (int, optional): number of steps until the jammer turns active. Defaults to 0.
        routing_func (Callable): Function determining how the data is routed in the network.
        step_size (int): The maximum step size for each nodes action.
        numbOfByzantineNodes (int): Number of Byzantine nodes. Defaults to 0.
        byzantine_drop_rate (float): Drop rate for greyhole/sinkhole Byzantine nodes.
        byzantine_attack_type (str): Attack type for Byzantine nodes.
        byzantine_capacity_inflation_factor (float): Capacity multiplier for sinkhole nodes.
        byzantine_jamming_power_factor (float): Interference multiplier for selective_jamming nodes.
        byzantine_pos_offset (array-like or None): Position offset for position_spoofing nodes.
        byzantine_obs_user_dir_offset (array-like or None): Direction offset for obs_user_dir nodes.
        byzantine_obs_capacity_factor (float): Capacity scale factor for obs_capacity nodes.
        byzantine_obs_demand_factor (float): Demand scale factor for obs_demand nodes.
        byzantine_obs_interference_factor (float): Interference scale factor for obs_interference_lie nodes.
        byzantine_obs_noise_std (float): Noise std dev for obs_noise nodes.

    Returns:
        envs.Basic_MANETEnv: the created environment
    """

    # Create a dictionary for optional parameters
    optional_params = {
        "render_mode": render_mode,
        "seed": seed,
        "networkConnectedAtStart": networkConnectedAtStart,
        "jammersSpawnNextToUsers": jammersSpawnNextToUsers,
        "attackerModel": attackerModel,
        "allow_early_ep_finish": allow_early_ep_finish,
    }

    # Create the environment with the optional parameters
    env: envs.Basic_MANETEnv = env_constructor(
        numbOfNodes, numbOfJammers, numbOfUsers, **optional_params
    )

    env.set_ep_length(ep_length)
    env.set_steps_till_activate_jammers(steps_till_jammer_active)
    env.set_routing_function(routing_func)
    env.set_node_step_size(step_size)
    env.set_step_jammers_start_moving(steps=step_jammers_start_moving)
    env.set_byzantine_params(
        numbOfByzantineNodes=numbOfByzantineNodes,
        byzantine_drop_rate=byzantine_drop_rate,
        byzantine_attack_type=byzantine_attack_type,
        byzantine_capacity_inflation_factor=byzantine_capacity_inflation_factor,
        byzantine_jamming_power_factor=byzantine_jamming_power_factor,
        byzantine_pos_offset=byzantine_pos_offset,
        byzantine_obs_user_dir_offset=byzantine_obs_user_dir_offset,
        byzantine_obs_capacity_factor=byzantine_obs_capacity_factor,
        byzantine_obs_demand_factor=byzantine_obs_demand_factor,
        byzantine_obs_interference_factor=byzantine_obs_interference_factor,
        byzantine_obs_noise_std=byzantine_obs_noise_std,
        byzantine_obs_capacity_deflate_factor=byzantine_obs_capacity_deflate_factor,
        byzantine_obs_demand_deflate_factor=byzantine_obs_demand_deflate_factor,
        byzantine_obs_replay_delay=byzantine_obs_replay_delay,
        byzantine_obs_magnitude_range=byzantine_obs_magnitude_range,
    )
    env.reset(seed=seed)

    return env


def wrapWithBaseWrapper(
    env: gym.Env,
    obs_config: dict,
    obs_type: str = "full",
    state_loss_rate: float = 0.3,
    node_idx: int = None,
) -> ow.Base_ObservationWrapper:

    # Then wrap it up
    base_obs_wrapper = getBaseObsWrapper(obs_type)

    if obs_type == "full":
        base_obs_wrapper: Type[ow.Base_ObservationWrapper]
        wrapped_env = base_obs_wrapper(env=env, config=obs_config)
    elif obs_type == "random":
        base_obs_wrapper: Type[ow.Random_ObservationWrapper]
        wrapped_env = base_obs_wrapper(
            env=env, config=obs_config, state_loss_rate=state_loss_rate
        )
    elif obs_type == "real":
        base_obs_wrapper: Type[ow.Real_ObservationWrapper]
        wrapped_env = base_obs_wrapper(
            env=env,
            config=obs_config,
            node_idx=node_idx,
            state_loss_rate=state_loss_rate,
        )
    elif obs_type == "gossip1":
        wrapped_env = ow.Real_ObservationWrapper(
            env=env,
            config=obs_config,
            node_idx=node_idx,
            state_loss_rate=state_loss_rate,
            gossip_hops=1,
        )
    elif obs_type == "reliable":
        wrapped_env = ow.ReliableBroadcast_ObservationWrapper(
            env=env,
            config=obs_config,
            node_idx=node_idx,
            state_loss_rate=state_loss_rate,
        )
    else:
        base_obs_wrapper: Type[ow.Base_ObservationWrapper]
        wrapped_env = base_obs_wrapper(env=env, config=obs_config)

    return wrapped_env


# * Object retrieval
def getRewardWrapperFromName(rew_wrapper_name: str) -> Type[gym.RewardWrapper]:
    """
    Translates reward wrapper name into actual reward wrapper class if available. Else returns the LinearReward reward wrapper.

    Args:
        rew_wrapper_name (str): name of the reward wrapper class desired

    Returns:
        gym.RewardWrapper: desired reward wrapper
    """
    rew_wrapper = REWARD_WRAPPER_CLASSES.get(rew_wrapper_name, rw.LinearReward)

    if rew_wrapper_name not in REWARD_WRAPPER_CLASSES.keys():
        print(
            f"Reward wrapper {rew_wrapper_name} not found in {REWARD_WRAPPER_CLASSES.keys()}."
        )
        print(f"Using {rew_wrapper.__name__}")

    return rew_wrapper


def getObservationWrapperFromName(
    obs_wrapper_name: str,
) -> Type[gym.ObservationWrapper]:
    """
    Get the observation wrapper class corresponding to the observation wrapper name given. If not exists, then returns VS_COW wrapper.

    Args:
        obs_wrapper_name (str): name of the observation wrapper desired

    Returns:
        Type[gym.ObservationWrapper]: desired observation wrapper
    """
    obs_wrapper = OBSERVATION_WRAPPER_CLASSES.get(obs_wrapper_name, ow.VS_COW)

    if obs_wrapper_name not in OBSERVATION_WRAPPER_CLASSES.keys():
        print(
            f"Observation wrapper {obs_wrapper_name} not found in {OBSERVATION_WRAPPER_CLASSES.keys()}."
        )
        print(f"Using {obs_wrapper.__name__}")

    return obs_wrapper


def getEnvironmentFromName(env_name: str) -> Type[envs.Basic_MANETEnv]:
    """Returns a constructor for a MANETEnvironment given the name

    Args:
        env_name (str): name of the environment

    Returns:
        Type[envs.Basic_MANETEnv]: environment's constructor
    """
    env = MANET_ENV_CLASSES.get(env_name, envs.Static_MANETEnv)

    if env_name not in MANET_ENV_CLASSES.keys():
        print(f"Environment {env_name} not found in {MANET_ENV_CLASSES.keys()}.")
        print(f"Using {env.__name__}")

    return env


def getAttackerModelFromName(
    attacker_model_name: Union[str, list],
) -> Union[Type[at.AttackerModel], list]:
    """Returns a constructor for an AttackerModel or a list of constructors given the name(s).

    Args:
        attacker_model_name (Union[str, list]): Name or list of names of the attacker_model(s).

    Returns:
        Union[Type[at.AttackerModel], list]: Attacker model's constructor or a list of constructors.
    """
    if isinstance(attacker_model_name, list):
        attacker_models = set()  # Use a set to avoid duplicates
        for name in attacker_model_name:
            if isinstance(name, str):
                attacker_model = ATTACKER_MODEL_CLASSES.get(
                    name, at.Static_GreedyJammers
                )
                if name not in ATTACKER_MODEL_CLASSES.keys():
                    print(
                        f"Attacker model {name} not found in {ATTACKER_MODEL_CLASSES.keys()}."
                    )
                    print(f"Using {attacker_model.__name__}")
                attacker_models.add(attacker_model)
            elif attacker_model_name in ATTACKER_MODEL_CLASSES.values():
                attacker_model = attacker_model_name
                attacker_models.add(attacker_model)
        return list(attacker_models)  # Convert set to list before returning
    else:
        attacker_model = ATTACKER_MODEL_CLASSES.get(
            attacker_model_name, at.Static_GreedyJammers
        )
        if attacker_model_name not in ATTACKER_MODEL_CLASSES.keys():
            print(
                f"Attacker model {attacker_model_name} not found in {ATTACKER_MODEL_CLASSES.keys()}."
            )
            print(f"Using {attacker_model.__name__}")
        return attacker_model


def getAttackerModelNameFromInstance(
    attacker_model_instance: Union[Type[at.AttackerModel], list],
) -> Union[str, list]:
    """Returns the string representation(s) of an AttackerModel or a list of AttackerModels.

    Args:
        attacker_model_instance (Union[Type[at.AttackerModel], list]):
            An AttackerModel class or a list of AttackerModel classes.

    Returns:
        Union[str, list]: String representation of the attacker model or a list of string representations.
    """
    # Create a reverse mapping from class to name
    reverse_mapping = {v: k for k, v in ATTACKER_MODEL_CLASSES.items()}

    if isinstance(attacker_model_instance, list):
        attacker_model_names = []
        for model in attacker_model_instance:
            model_name = reverse_mapping.get(model, "UnknownModel")
            if model_name == "UnknownModel":
                print(
                    f"Attacker model {model} not found in {list(reverse_mapping.values())}."
                )
            attacker_model_names.append(model_name)
        return attacker_model_names
    else:
        model_name = reverse_mapping.get(attacker_model_instance, "UnknownModel")
        if model_name == "UnknownModel":
            print(
                f"Attacker model {attacker_model_instance} not found in {list(reverse_mapping.values())}."
            )
        return model_name


def getBaselineFromName(agent_baseline_name: str) -> Type[ag.AgentBase]:
    """Returns a constructor for an Agent Baseline given the name

    Args:
        agent_baseline_name (str): name of the baseline agent

    Returns:
        Type[ag.AgentBase]: attacker model's constructor
    """
    agent_baseline = BASELINE_AGENT_CLASSES.get(agent_baseline_name, ag.GreedyMaximizer)

    if agent_baseline_name not in BASELINE_AGENT_CLASSES.keys():
        print(
            f"Attacker model {agent_baseline_name} not found in {BASELINE_AGENT_CLASSES.keys()}."
        )
        print(f"Using {agent_baseline.__name__}")

    return agent_baseline


def getConditionStrategyFromName(
    condition_strategy_name: str,
) -> Type[cs.ConditionStrategy]:
    """Returns a constructor for ConditionStrategy given the name

    Args:
        condition_strategy_name (str): name of the conditionStrategy agent

    Returns:
        Type[cs.ConditionStrategy]: attacker model's constructor
    """
    condition_strategy = CONDITION_STRATEGY_CLASSES.get(
        condition_strategy_name, ag.GreedyMaximizer
    )

    if condition_strategy_name not in CONDITION_STRATEGY_CLASSES.keys():
        print(
            f"Attacker model {condition_strategy_name} not found in {CONDITION_STRATEGY_CLASSES.keys()}."
        )
        print(f"Using {condition_strategy.__name__}")

    return condition_strategy


def getRoutingFunction(routing_func_name: str) -> Callable:
    """Retrives the routing function given its name.

    Args:
        routing_func_name (str): Name of the routing function

    Returns:
        Callable: Routing function fitting the name.
    """

    routing_func = ROUTING_FUNCTIONS.get(routing_func_name, MaxFlow.sequential_max_flow)

    if routing_func_name not in ROUTING_FUNCTIONS.keys():
        print(
            f"Routing function {routing_func_name} not found in {ROUTING_FUNCTIONS.keys()}."
        )
        print(f"Using {routing_func.__name__}")

    return routing_func


def getBaseObsWrapper(type: str) -> Type[gym.ObservationWrapper]:
    base_obs_wrapper = BASE_WRAPPER_CLASSES.get(type, ow.Base_ObservationWrapper)

    if type not in BASE_WRAPPER_CLASSES.keys():
        print(
            f"Base observation wrapper {type} not found in {BASE_WRAPPER_CLASSES.keys()}."
        )
        print(f"Using {base_obs_wrapper.__name__}")

    return base_obs_wrapper


#  * Load params
def loadParametersFromConfig(
    # Config
    config,
    # storing and logging directories and names
    model_dir: str = None,
    log_dir: str = None,
    log_name: str = None,
    # env
    env: Union[str, Type[envs.Basic_MANETEnv]] = "Static_MANETEnv",
    # whether to render env
    render: bool = None,
    # env parameters
    numbOfNodes: int = None,
    numbOfJammers: Union[int, Tuple[int, int]] = None,
    numbOfUsers: Union[int, Tuple[int, int]] = None,
    seed: int = None,
    networkConnectedAtStart: bool = None,
    jammersSpawnNextToUsers: bool = None,
    allow_early_ep_finish: bool = None,
    routing_func: Callable = None,
    step_size: int = None,
    # attacker model
    attackerModel: Type[at.AttackerModel] = None,
    steps_till_jammer_active: int = None,
    step_jammers_start_moving: int = None,
    # byzantine
    numbOfByzantineNodes: int = None,
    byzantine_drop_rate: float = None,
    byzantine_attack_type: Union[str, List[str]] = None,
    # obs config
    obs_config: dict = None,
    # rew wrapper
    rew_wrapper: Type[gym.RewardWrapper] = None,
    # training params
    timesteps: int = None,
    ep_length: int = None,
    save_freq: int = None,
    stat_size: int = None,
    reset: bool = None,
    # To handle newer configs
    model_path: str = None,
    log_path: str = None,
    tb_logname: str = None,
) -> tuple:
    """
    Load the parameters for training and environment configuration, either from the provided values or the config file.

    Args:
        config (dict): The configuration dictionary containing default values and other settings.
        model_dir (str, optional): Directory to store the model.
        log_dir (str, optional): Directory to store the logs.
        log_name (str, optional): Name of the log file.
        render (bool, optional): Whether to render the environment.
        numbOfNodes (int, optional): Number of nodes in the environment.
        numbOfJammers (int, optional): Number of jammers in the environment.
        numbOfUsers (int, optional): Number of users in the environment.
        seed (int, optional): Random seed for reproducibility.
        networkConnectedAtStart (bool, optional): Whether the network is connected at the start.
        jammersSpawnNextToUsers (bool, optional): Whether jammers spawn next to users.
        allow_early_ep_finish (bool, optional): Whether to allow early episode finish.
        attackerModel (Type[at.AttackerModel], optional): The attacker model to use.
        steps_till_jammer_active (int, optional): Steps until jammers become active.
        step_jammers_start_moving (int, optional): Step from which on the jammers start movig to position themselves.
        obs_config (dict, optional): Configuration for the observation wrapper.
        rew_wrapper (Type[gym.RewardWrapper], optional): The reward wrapper to use.
        timesteps (int, optional): Number of timesteps to train.
        ep_length (int, optional): Length of each episode.
        save_freq (int, optional): Frequency to save the model.
        stat_size (int, optional): Size of the statistics buffer.

    Returns:
        tuple: A tuple containing the following:
            - model_path (str): Path to store the model.
            - log_path (str): Path to store the logs.
            - tb_logname (str): Tensorboard log name.
            - RL_model_name (str): Name of the RL model.
            - env (Type[envs.Basic_MANETEnv]): The environment class.
            - render (bool): Whether to render the environment.
            - numbOfNodes (int): Number of nodes in the environment.
            - numbOfJammers (int): Number of jammers in the environment.
            - numbOfUsers (int): Number of users in the environment.
            - seed (int): Random seed for reproducibility.
            - networkConnectedAtStart (bool): Whether the network is connected at the start.
            - jammersSpawnNextToUsers (bool): Whether jammers spawn next to users.
            - allow_early_ep_finish (bool): Whether to allow early episode finish.
            - attackerModel (Type[at.AttackerModel]): The attacker model to use.
            - steps_till_jammer_active (int): Steps until jammers become active.
            - step_jammers_start_moving (int, optional): Step from which on the jammers start movig to position themselves.
            - obs_wrapper (Type[gym.ObservationWrapper]): The observation wrapper to use.
            - obs_config (dict): Configuration for the observation wrapper.
            - rew_wrapper (Type[gym.RewardWrapper]): The reward wrapper to use.
            - timesteps (int): Number of timesteps to train.
            - ep_length (int): Length of each episode.
            - save_freq (int): Frequency to save the model.
            - stat_size (int): Size of the statistics buffer.

    Notes:
        This function loads the parameters for training and environment setup either from
        provided arguments or the configuration dictionary. It returns the appropriate
        values for further processing, such as model training and environment interaction.
    """

    def get_param(param, key):
        if param is not None:
            return param
        if config.get(key) is not None:
            return config[key]
        return DEFAULT_CONFIG[key]

    numbOfNodes = get_param(numbOfNodes, "numbOfNodes")
    numbOfJammers = get_param(numbOfJammers, "numbOfJammers")
    numbOfUsers = get_param(numbOfUsers, "numbOfUsers")
    render = get_param(render, "render")
    seed = get_param(seed, "seed")
    networkConnectedAtStart = get_param(
        networkConnectedAtStart, "networkConnectedAtStart"
    )
    jammersSpawnNextToUsers = get_param(
        jammersSpawnNextToUsers, "jammersSpawnNextToUsers"
    )
    allow_early_ep_finish = get_param(allow_early_ep_finish, "allow_early_ep_finish")
    steps_till_jammer_active = get_param(
        steps_till_jammer_active, "steps_till_jammer_active"
    )
    step_jammers_start_moving = get_param(
        step_jammers_start_moving, "step_jammers_start_moving"
    )
    obs_config = get_param(obs_config, "obs_config")
    timesteps = get_param(timesteps, "timesteps")
    ep_length = get_param(ep_length, "ep_length")
    save_freq = get_param(save_freq, "save_freq")
    stat_size = get_param(stat_size, "stat_size")
    reset = get_param(reset, "reset")
    routing_func = get_param(routing_func, "routing_func")
    step_size = get_param(step_size, "step_size")
    attackerModel = get_param(attackerModel, "attackerModel")
    numbOfByzantineNodes = get_param(numbOfByzantineNodes, "numbOfByzantineNodes")
    byzantine_drop_rate = get_param(byzantine_drop_rate, "byzantine_drop_rate")
    byzantine_attack_type = get_param(byzantine_attack_type, "byzantine_attack_type")
    byzantine_capacity_inflation_factor = get_param(None, "byzantine_capacity_inflation_factor")
    byzantine_jamming_power_factor = get_param(None, "byzantine_jamming_power_factor")
    byzantine_pos_offset = get_param(None, "byzantine_pos_offset")
    byzantine_obs_user_dir_offset = get_param(None, "byzantine_obs_user_dir_offset")
    byzantine_obs_capacity_factor = get_param(None, "byzantine_obs_capacity_factor")
    byzantine_obs_demand_factor = get_param(None, "byzantine_obs_demand_factor")
    byzantine_obs_interference_factor = get_param(None, "byzantine_obs_interference_factor")
    byzantine_obs_noise_std = get_param(None, "byzantine_obs_noise_std")
    byzantine_obs_capacity_deflate_factor = get_param(None, "byzantine_obs_capacity_deflate_factor")
    byzantine_obs_demand_deflate_factor = get_param(None, "byzantine_obs_demand_deflate_factor")
    byzantine_obs_replay_delay = get_param(None, "byzantine_obs_replay_delay")
    byzantine_obs_magnitude_range = get_param(None, "byzantine_obs_magnitude_range")
    obs_type = get_param(None, "obs_type")
    state_loss_rate = get_param(None, "state_loss_rate")
    obs_wrapper = config["obs_wrapper"]  # config only
    rew_wrapper = get_param(rew_wrapper, "rew_wrapper")
    env = get_param(env, "env")

    # specials
    model_path = config["model_path"] if model_path is None else model_path
    log_path = config["log_path"] if log_path is None else log_path
    tb_logname = config["tb_logname"] if tb_logname is None else tb_logname
    RL_model_name = config["RL_model_name"]  # config only

    # Overwrite
    tb_logname = (
        tb_logname
        if log_name is None
        else getLogname(
            log_name, RL_model_name, numbOfNodes, numbOfJammers, numbOfUsers, ep_length
        )
    )
    log_path = log_path if log_dir is None else log_dir
    model_path = (
        model_path if model_dir is None else os.path.join(model_dir, tb_logname)
    )

    return (
        model_path,
        log_path,
        tb_logname,
        RL_model_name,
        env,
        render,
        numbOfNodes,
        numbOfJammers,
        numbOfUsers,
        seed,
        networkConnectedAtStart,
        jammersSpawnNextToUsers,
        allow_early_ep_finish,
        attackerModel,
        steps_till_jammer_active,
        step_jammers_start_moving,
        obs_wrapper,
        obs_config,
        rew_wrapper,
        timesteps,
        ep_length,
        save_freq,
        stat_size,
        reset,
        routing_func,
        step_size,
        numbOfByzantineNodes,
        byzantine_drop_rate,
        byzantine_attack_type,
        byzantine_capacity_inflation_factor,
        byzantine_jamming_power_factor,
        byzantine_pos_offset,
        byzantine_obs_user_dir_offset,
        byzantine_obs_capacity_factor,
        byzantine_obs_demand_factor,
        byzantine_obs_interference_factor,
        byzantine_obs_noise_std,
        byzantine_obs_capacity_deflate_factor,
        byzantine_obs_demand_deflate_factor,
        byzantine_obs_replay_delay,
        byzantine_obs_magnitude_range,
        obs_type,
        state_loss_rate,
    )


# * Logname
def setDirectories(
    RL_model_name: str,
    model_dir: str,
    log_dir: str,
    log_name: str,
    numbOfNodes: int,
    numbOfJammers: Union[int, Tuple[int, int]],
    numbOfUsers: Union[int, Tuple[int, int]],
    ep_length: int,
) -> tuple:
    """
    Sets and creates the directories for storing models and logs.

    Args:
        RL_model_name (str): The RL model's name one wants to use.
        model_dir (str): Directory to store the model.
        log_dir (str): Directory to store the logs.
        log_name (str): Name of the log file.
        numbOfNodes (int): Number of nodes in the environment.
        numbOfJammers (int): Number of jammers in the environment.
        numbOfUsers (int): Number of users in the environment.
        ep_length (int): Length of each episode.

    Returns:
        tuple: A tuple containing the log path, tensorboard log name, and model path.

    Notes:
        This function ensures the specified directories for storing models and logs exist,
        and it returns paths formatted based on the provided parameters.
    """

    date: str = getDayMonthAsString()

    log_path: str = os.path.join(log_dir, date)

    # Determine log name (always includes RL_model_name)
    tb_logname: str = getLogname(
        log_name=log_name,
        RL_model_name=RL_model_name,
        numbOfNodes=numbOfNodes,
        numbOfJammers=numbOfJammers,
        numbOfUsers=numbOfUsers,
        ep_length=ep_length,
    )

    # Set and create the directory in which the model should be stored
    model_path: str = os.path.join(model_dir, date, tb_logname)

    model_path, log_path = convertPathsToPyPath(model_path, log_path)
    # Convert paths to use forward slashes
    create_dirs(model_path, log_path)

    return log_path, tb_logname, model_path


def getLogname(
    log_name: str,
    RL_model_name: str,
    numbOfNodes: int,
    numbOfJammers: Union[int, Tuple[int, int]],
    numbOfUsers: Union[int, Tuple[int, int]],
    ep_length: int,
) -> str:
    """
    Constructs the log name based on the provided parameters.

    Args:
        log_name (str): Base name of the log file.
        RL_model_name (str): Name of the RL model class.
        numbOfNodes (int): Number of nodes in the environment.
        numbOfJammers (int): Number of jammers in the environment.
        numbOfUsers (int): Number of users in the environment.
        ep_length (int): Length of each episode.

    Returns:
        str: The constructed log name.

    Notes:
        This function combines the base log name with other parameters to generate
        a descriptive and unique log name for storing training or evaluation data.
    """

    def rangeStrRepr(int_or_range: Union[int, Tuple[int, int]]) -> str:
        if isinstance(int_or_range, int):
            return str(int_or_range)
        else:
            strrepr = f"[{int_or_range[0]},{int_or_range[1]}]"
            return strrepr

    tb_logname = (
        f"{RL_model_name}_{numbOfNodes}_{rangeStrRepr(numbOfJammers)}_{rangeStrRepr(numbOfUsers)}_{ep_length/1e3:.0f}k"
        + (f"_{log_name}" if log_name else "")
    )
    return tb_logname


# * General
def convertPathsToPyPath(*paths: str) -> str:
    """
    Takes paths of "out\\model\\..." and turns them to "out/model/...".

    Args:
        paths (str): Any number of path strings.

    Returns:
        tuple or str: A tuple containing the normalized paths with forward slashes if more than one path is provided, otherwise a single normalized path string.
    """
    normalized_paths = [path.replace("\\", "/") for path in paths]

    return normalized_paths


def getFileNamesFromPaths(*paths: str) -> str:
    """Returns the file names from the paths to a file given.

    Returns:
        str: file names as a tuple
    """
    file_names = []
    for path in paths:
        file_name = os.path.basename(path)  # Extract file name with extension
        name_without_extension, _ = os.path.splitext(file_name)  # Remove the extension
        file_names.append(name_without_extension)

    if len(file_names) == 1:
        return file_names[0]
    return tuple(file_names)


def getDayMonthAsString() -> str:
    """
    Get the current day and month as a string.

    Returns:
        str: The current day and month in the format "DD_MM".

    Notes:
        This function retrieves the current date and returns it as a string in the
        format "DD_MM", representing the day and month.
    """
    return datetime.now().strftime("%d_%m")


def filterNone(**parameters) -> dict:
    """Returns a dictionary only with key not containing empty values such as empty list, dict or None

    Returns:
        dict: _description_
    """
    return {k: v for k, v in parameters.items() if v not in [None, {}, []]}


# Escape brackets for glob
def escape_glob_literal_brackets(path: str) -> str:
    escaped_path = ""
    for char in path:
        if char == "[":
            escaped_path += "[[]"
        elif char == "]":
            escaped_path += "[]]"
        else:
            escaped_path += char
    return escaped_path
