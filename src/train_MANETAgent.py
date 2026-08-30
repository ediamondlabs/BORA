# OS stuff
import os
import re

# For type declaration
from typing import Type, Union, List, Callable, Tuple

# Gymnasium
import gymnasium as gym

# Stable baselines 3
from stable_baselines3.common.callbacks import CheckpointCallback

# Own classes
import MANET.Environments as envs
import MANET.Attackers as at
import Utils.training_utils as ut
from MANET.Routing import MaxFlow


# * Almighty train or retrain
def train_or_retrain(
    path: str,
    specific_model_step: str = None,
    # Storage params
    model_dir: str = None,
    log_dir: str = None,
    log_name: str = None,
    # model
    RL_model_name: str = None,
    # env
    env: str = None,
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
    routing_func: Union[Callable, str] = None,
    step_size: int = None,
    # attacker model
    attackerModel: Union[
        at.AttackerModel, List[at.AttackerModel], str, List[str]
    ] = None,
    steps_till_jammer_active: int = None,
    step_jammers_start_moving: int = None,
    # obswrapper
    obs_wrapper: str = None,
    # obs config
    obs_config: dict = None,
    # rew wrapper
    rew_wrapper: str = None,
    # training params
    timesteps: int = None,
    ep_length: int = None,
    save_freq: int = None,
    stat_size: int = None,
    n_envs: int = None,
):
    """
    Trains or retrains an agent, depending on what the given path contains.

    1. If the path points to a file:
        - If the file is a config file:
            - Trains or retrains using that config file.
        - If the file is an agent file:
            - Retrains that specific agent file based on `01_training_config.yaml`.

    2. If the path is a directory:
        - Checks whether the directory contains agent files:
            - If it contains agent files:
                - Checks `01_training_config.yaml`.
                - If the first training hasn't been completed:
                    - Retrains to complete the initially specified number of timesteps.
                - Else:
                    - Retrains from `01_training_config.yaml` but restarts the number of timesteps.
            - If it doesn't contain agent files:
                - Checks for training or retraining config:
                    - If it contains any of the two:
                        - Retrains or trains based on those configs.
                    - Else:
                        - Trains with default values.

    Note:
        Any parameters for training and retraining done from a config or from default values can be overwritten by passing any of those arguments to the function.

    Args:
        path (str): Path to the file or directory.
        specific_model_step (str, optional): Specific model step to load. Defaults to None.

        model_dir (str, optional): Directory to save the model. Defaults to None.
        log_dir (str, optional): Directory to save the logs. Defaults to None.
        log_name (str, optional): Name of the log file. Defaults to None.

        RL_model_name (str, optional): Name of the RL model. Defaults to None.

        env (str, optional): Environment to use. Defaults to None.

        render (bool, optional): Whether to render the environment. Defaults to None.

        numbOfNodes (int, optional): Number of nodes in the environment. Defaults to None.
        numbOfJammers (int, optional): Number of jammers in the environment. Defaults to None.
        numbOfUsers (int, optional): Number of users in the environment. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        networkConnectedAtStart (bool, optional): Whether the network is connected at the start. Defaults to None.
        jammersSpawnNextToUsers (bool, optional): Whether jammers spawn next to users. Defaults to None.
        allow_early_ep_finish (bool, optional): Whether to allow early episode finish. Defaults to None.
        step_size (int): Maximum step size of MANET node. Since, per default an action is in [-1,1]^2 it simply scales the action.

        attackerModel (str, optional): Attacker model to use. Defaults to None.
        steps_till_jammer_active (int, optional): Steps until jammers become active. Defaults to None.
        step_jammers_start_moving (int, optional): Step from which on the jammers start movig to position themselves.

        obs_wrapper (str, optional): Observation wrapper to use. Defaults to None.
        obs_config (dict, optional): Configuration for the observation wrapper. Defaults to None.

        rew_wrapper (str, optional): Reward wrapper to use. Defaults to None.

        timesteps (int, optional): Number of timesteps to train. Defaults to None.
        ep_length (int, optional): Length of each episode. Defaults to None.
        save_freq (int, optional): Frequency to save the model. Defaults to None.
        stat_size (int, optional): Size of the statistics buffer. Defaults to None.

    Raises:
        ValueError: If the path is neither a folder nor a file.
    """
    # The training conig's file name
    TRAINING_CONFIG_FILE = "01_training_config.yaml"

    folder_path = None
    # If the path is a file
    if os.path.isfile(path):
        folder_path = os.path.dirname(path)
        file_name = os.path.basename(path)
        if path.endswith(".zip"):
            specific_step = ut.getStepsFromFileName(file_name)
            retrain_agent(
                folder_path,
                specific_step,
                timesteps=timesteps,
                # storing and logging directories and names
                model_dir=model_dir,
                log_dir=log_dir,
                log_name=log_name,
                # whether to render env
                render=render,
                # env parameters
                numbOfJammers=numbOfJammers,
                seed=seed,
                networkConnectedAtStart=networkConnectedAtStart,
                jammersSpawnNextToUsers=jammersSpawnNextToUsers,
                allow_early_ep_finish=allow_early_ep_finish,
                routing_func=routing_func,
                step_size=step_size,
                # attacker model
                attackerModel=attackerModel,
                steps_till_jammer_active=steps_till_jammer_active,
                step_jammers_start_moving=step_jammers_start_moving,
                # obs config
                obs_config=obs_config,
                # rew wrapper
                rew_wrapper=rew_wrapper,
                # training params
                ep_length=ep_length,
                save_freq=save_freq,
                stat_size=stat_size,
                reset=False,
                n_envs=n_envs or 1,
            )
        elif path.endswith(".yaml"):
            train_or_retrain_from_config(folder_path, file_name)

        return

    # Path should be dir
    if os.path.isdir(path):
        folder_path = path
    else:
        raise ValueError(f"Path neither of a folder nor of a file: {path}")

    # If a trained model exists, determine retrain conditions
    if ut.hasAgentFiles(path):
        config_file_names: list = ut.loadConfigFileNames(path)
        file_name = (
            TRAINING_CONFIG_FILE
            if ut.hasSpecificConfigFile(path, TRAINING_CONFIG_FILE)
            else ut.getMostRecentConfigFile(path)
        )
        config = ut.loadConfig(path, file_name)

        # Agent files
        latest_rl_agent: str = ut.getFileNameOfLatestAgent(path)
        rl_agent_steps: int = ut.getStepsFromFileName(latest_rl_agent)

        # A previous training was finished if the full timesteps were completed
        # or if there already has been retraining taken place (retrain config exists)
        training_was_finished = (
            len(config_file_names) > 1 or config["timesteps"] <= rl_agent_steps
        )

        # get remaining steps
        remaining_steps = (
            config["timesteps"] - rl_agent_steps
            if not training_was_finished
            else timesteps
        )

        retrain_agent(
            dir_of_agent_to_load=path,
            steps_of_agent_to_load=specific_model_step,
            timesteps=timesteps or remaining_steps,
            # storing and logging directories and names
            model_dir=model_dir,
            log_dir=log_dir,
            log_name=log_name,
            # whether to render env
            render=render,
            # env parameters
            numbOfJammers=numbOfJammers,
            seed=seed,
            networkConnectedAtStart=networkConnectedAtStart,
            jammersSpawnNextToUsers=jammersSpawnNextToUsers,
            allow_early_ep_finish=allow_early_ep_finish,
            routing_func=routing_func,
            step_size=step_size,
            # attacker model
            attackerModel=attackerModel,
            steps_till_jammer_active=steps_till_jammer_active,
            step_jammers_start_moving=step_jammers_start_moving,
            # obs config
            obs_config=obs_config,
            # rew wrapper
            rew_wrapper=rew_wrapper,
            # training params
            ep_length=ep_length,
            save_freq=save_freq,
            stat_size=stat_size,
            reset=False,
            n_envs=n_envs or 1,
        )
        return

    # If no trained model but config files exist, use the latest one
    if ut.hasConfigFiles(path):
        config_name = ut.getMostRecentConfigFile(path)
        train_or_retrain_from_config(path, config_name)
        return

    # Otherwise, train from scratch with default values
    train_agent(
        **ut.filterNone(
            model_dir=model_dir,
            log_dir=log_dir,
            log_name=log_name,
            RL_model_name=RL_model_name,
            env=env,
            render=render,
            numbOfNodes=numbOfNodes,
            numbOfJammers=numbOfJammers,
            numbOfUsers=numbOfUsers,
            seed=seed,
            routing_func=routing_func,
            step_size=step_size,
            networkConnectedAtStart=networkConnectedAtStart,
            jammersSpawnNextToUsers=jammersSpawnNextToUsers,
            allow_early_ep_finish=allow_early_ep_finish,
            attackerModel=attackerModel,
            steps_till_jammer_active=steps_till_jammer_active,
            step_jammers_start_moving=step_jammers_start_moving,
            obs_wrapper=obs_wrapper,
            obs_config=obs_config,
            rew_wrapper=rew_wrapper,
            timesteps=timesteps,
            ep_length=ep_length,
            save_freq=save_freq,
            stat_size=stat_size,
        )
    )


# * Train or retrain from config
def train_or_retrain_from_config(folder_path: str, config_name: str) -> None:
    """
    Load configuration and run training or retraining based on the config.

    Args:
        folder_path (str): Path to the folder containing the configuration file.
        config_name (str): Name of the configuration file.
    """
    config = ut.loadConfig(folder_path, config_name)
    mode = config.pop("mode")

    if mode == "train":
        train_agent(**config)
    elif mode == "retrain":
        retrain_agent(**config)
    else:
        print(f"Mode {mode} not recognized")


def train_agent_from_config(
    config_path: str,
) -> None:
    """This script trains an agent using a configuration file. You don't need to specify all parameters
    in the configuration file; unspecified values will default to their predefined values.

    Args:
        config_path (str): The path of the config file

    ### Default Parameters:
    1. **Storing and Logging**:
    - `model_dir`: "out/models" (Directory to save the trained model)
    - `log_dir`: "out/logs" (Directory to save logs)
    - `log_name`: "" (Name of the log file)

    2. **Model**:
    - `RL_model_name`: "PPO" (Reinforcement Learning model to use)

    3. **Environment**:
    - `env`: "Static_MANETEnv" (Environment to train the agent in)
    - `render`: False (Whether to render the environment during training)

    4. **Environment Parameters**:
    - `numbOfNodes`: 5 (Number of MANET nodes)
    - `numbOfJammers`: 1 (Number of jammers)
    - `numbOfUsers`: 10 (Number of users)
    - `seed`: 9 (Random seed for reproducibility)
    - `networkConnectedAtStart`: False (Whether the network is connected at the start)
    - `jammersSpawnNextToUsers`: True (Whether jammers spawn near users)
    - `allow_early_ep_finish`: True (Allow early episode termination)
    - `step_size` (int): 1 (Maximum step size of MANET node. Since, per default an action is in [-1,1]^2 it simply scales the action.)


    5. **Attacker Model**:
    - `attackerModel`: "Static_ClusterJammers" (Type of attacker model)
    - `steps_till_jammer_active`: 0 (Steps before jammers become active)
    - `step_jammers_start_moving`: 0 (Step from which on the jammers start moving)

    5b. **Byzantine Parameters**:
    - `numbOfByzantineNodes`: 0 (Number of Byzantine attacker nodes)
    - `byzantine_drop_rate`: 0.5 (Fraction of traffic each Byzantine node drops)
    - `byzantine_attack_type`: "greyhole" (Type of Byzantine attack)

    6. **Observation Wrapper**:
    - `obs_wrapper`: "VS_iCOW" (Wrapper for observations)
    - `obs_config`: None (Configuration for the observation wrapper)

    7. **Reward Wrapper**:
    - `rew_wrapper`: "SquaredReward" (Wrapper for rewards)

    8. **Training Parameters**:
    - `timesteps`: 30,000,000 (Total training timesteps)
    - `ep_length`: 10,000 (Maximum episode length)
    - `save_freq`: 500,000 (Frequency of saving the model)
    - `stat_size`: 10 (Size of statistics buffer)

    ### Usage:
    - Specify the path to the configuration file in the `config_path` parameter.
    - Call the `train_agent_from_config` function to start training.
    """
    config = ut.loadConfig(os.path.dirname(config_path), os.path.basename(config_path))
    if "mode" in config:
        config.pop("mode")
    train_agent(**config)


def retrain_agent_from_config(
    config_path: str,
) -> None:
    """Retrain an agent from a config file.

    Args:
        config_path (str): The path of the config file

    ### Overview:
    The configuration file specifies the folder containing the agent's model files to be retrained.
    The retraining process loads all parameters from the agent's existing configuration files,
    but these can be overwritten by specifying them in the retrain configuration file.

    ### Key Details:
    1. **Default Configuration**:
    - By default, the script loads the "01_training_config.yaml" file from the agent's folder.
    - If no specific agent file is specified in the retrain configuration, the script will
        automatically load the latest model (i.e., the one with the most training steps).

    2. **Custom Configuration**:
    - You can override default parameters by specifying them in the retrain configuration file.
    - Refer to the example configuration file:
        `example_retrain_config.yaml` in `src/examples/retrain/example_retrain_config.yaml`.

    ### Usage:
    - Update the `config_path` parameter to point to your retrain configuration file.
    - Ensure the retrain configuration file specifies the correct folder for the agent's model files or to the actual model file itself.
    """
    config = ut.loadConfig(os.path.dirname(config_path), os.path.basename(config_path))
    if "mode" in config:
        config.pop("mode")
    retrain_agent(**config)


# * Basic train and retrain
def train_agent(
    # storing and logging directories and names
    model_dir: str = "out/models",
    log_dir: str = "out/logs",
    log_name: str = "",
    # model
    RL_model_name: str = "PPO",
    # env
    env: Union[str, Type[envs.Basic_MANETEnv]] = "Static_MANETEnv",
    # whether to render env
    render: bool = False,
    # env parameters
    numbOfNodes: int = 5,
    numbOfJammers: Union[int, Tuple[int, int]] = 1,
    numbOfUsers: Union[int, Tuple[int, int]] = 10,
    seed: int = 9,
    networkConnectedAtStart: bool = False,
    jammersSpawnNextToUsers: bool = False,
    allow_early_ep_finish: bool = True,
    routing_func: Union[Callable, str] = MaxFlow.sequential_max_flow,
    step_size: int = 2,
    # attacker model
    attackerModel: Union[
        Type[at.AttackerModel], List[Type[at.AttackerModel]], str, List[str]
    ] = "Static_ClusterJammers",
    steps_till_jammer_active: int = 0,
    step_jammers_start_moving: int = 0,
    # byzantine
    numbOfByzantineNodes: int = 0,
    byzantine_drop_rate: float = 0.5,
    byzantine_attack_type: str = "greyhole",
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
    # obswrapper
    obs_wrapper: Union[str, Type[gym.ObservationWrapper]] = "VS_iCOW",
    # obs type (full / random / real) and state loss rate for random/real
    obs_type: str = "full",
    state_loss_rate: float = 0.0,
    # obs config
    obs_config: dict = None,
    # rew wrapper
    rew_wrapper: Union[str, Type[gym.RewardWrapper]] = "SquaredReward",
    # training params
    timesteps: int = 30_000_000,
    ep_length: int = 10_000,
    save_freq: int = 500_000,
    stat_size: int = 10,
    n_envs: int = 1,
) -> None:
    """
    This function trains a reinforcement learning agent in a MANET environment with configurations specified by the parameters. Logging and saving occur based on provided directories and frequencies.

    Args:
        model_dir (str): Directory to store the model.
        log_dir (str): Directory to store the logs.
        log_name (str): Name of the log file.

        RL_model_name (str): The RL model's name one wants to use.

        env (Union[str, Type[envs.Basic_MANETEnv]]): The MANET environment class.

        render (bool): Whether to render the environment.

        numbOfNodes (int): Number of nodes in the environment.
        numbOfJammers (int): Number of jammers in the environment.
        numbOfUsers (int): Number of users in the environment.
        seed (int): Random seed for reproducibility.
        networkConnectedAtStart (bool): Whether the network is connected at the start.
        jammersSpawnNextToUsers (bool): Whether jammers spawn next to users.
        routing_func (Union[Callable, str]): How to route the data in the network.
        step_size (int): The maximum step size for each nodes action
        allow_early_ep_finish (bool): Whether to allow early episode finish.

        attackerModel (attackerModel: Union[Type[ at.AttackerModel], List[Type[at.AttackerModel]], str, List[str]]): The attacker model to use.
        steps_till_jammer_active (int): Steps until jammers become active.
        step_jammers_start_moving (int, optional): Step from which on the jammers start movig to position themselves.

        numbOfByzantineNodes (int): Number of MANET nodes that behave as Byzantine attackers. Default: 0.
        byzantine_drop_rate (float): Fraction of traffic each Byzantine node drops. Default: 0.5.
        byzantine_attack_type (str): The type of Byzantine attack. Default: "greyhole".

        obs_wrapper (Union[str, Type[gym.ObservationWrapper]]): The observation wrapper to use.
        obs_config (dict): Configuration for the observation wrapper.
        rew_wrapper (Union[str, Type[gym.RewardWrapper]]): The reward wrapper to use.

        timesteps (int): Number of timesteps to train.
        ep_length (int): Length of each episode.
        save_freq (int): Frequency to save the model.
        stat_size (int): Size of the statistics.

    Returns:
        None

    ### How It Works:
    1. **Parameter Resolution**:
    - The function accepts parameters for environment setup, RL model configuration, logging, and training.
    - If certain parameters (e.g., `env`, `attackerModel`, `obs_wrapper`, `rew_wrapper`) are provided as strings, they are resolved to their respective class constructors using utility functions.

    2. **Validation**:
    - The `checkTrainingValues` function validates the input parameters to ensure they meet the expected criteria (e.g., valid environment class, positive timesteps).

    3. **Environment Setup**:
    - The environment is created and wrapped using the `createAndWrapEnv` utility function. This includes:
        - Setting up the MANET environment with the specified number of nodes, jammers, and users.
        - Applying observation and reward wrappers.
        - Configuring attacker models and other environment parameters.

    4. **Logging and Storage**:
    - The `setDirectories` utility function sets up directories for storing the model, logs, and TensorBoard files.
    - A custom callback (`CheckpointCallback`) is used to save the model periodically during training.

    5. **Agent Initialization**:
    - The RL agent is initialized using the specified RL model (e.g., PPO) and the wrapped environment.
    - The agent's statistics buffer size and random seed are also configured.

    6. **Configuration Storage**:
    - The training configuration is stored as a YAML file (`01_training_config.yaml`) in the model directory. This includes all parameters used for training, ensuring reproducibility.

    7. **Training**:
    - The agent is trained using the `learn` method, which runs for the specified number of timesteps.
    - Progress is logged to TensorBoard, and the model is saved at regular intervals.

    ### Examples:
    1. **Model Storage**:
    - If `model_dir="out/models"`, `log_name="test"`, `RL_model_name="PPO"`, `numbOfNodes=5`, `numbOfJammers=2`, `numbOfUsers=10`, and `ep_length=10000`:
        - The trained model will be stored in:
        ```
        out/models/<date>/PPO_5_2_10_10k_test/
        ```
        - The model files will include checkpoints saved at intervals (e.g., `rl_model_500000_steps.zip`).

    2. **Log Storage**:
    - If [log_dir="out/logs"], [log_name="test"], [RL_model_name="PPO"], [numbOfNodes=5], [numbOfJammers=2], [numbOfUsers=10], and [ep_length=10000]:
        - The logs will be stored in:
        ```
        out/logs/<date>/PPO_5_2_10_10k_test/
        ```
        - TensorBoard logs will also be saved in this directory for visualization.

    3. **Configuration Storage**:
    - The training configuration will be saved as:
        ```
        out/models/<date>/PPO_5_2_10_10k_test/01_training_config.yaml
        ```
    - This file contains all parameters used for training, ensuring reproducibility.
    """

    # Resolve class constructors if they are given as strings
    if isinstance(env, str):
        env: Type[envs.Basic_MANETEnv] = ut.getEnvironmentFromName(env)
    if isinstance(attackerModel, str):
        attackerModel: Union[Type[at.AttackerModel], List[Type[at.AttackerModel]]] = (
            ut.getAttackerModelFromName(attackerModel)
        )
    elif isinstance(attackerModel, list) and all(
        isinstance(item, str) for item in attackerModel
    ):
        attackerModel: Union[Type[at.AttackerModel], List[Type[at.AttackerModel]]] = [
            ut.getAttackerModelFromName(item) for item in attackerModel
        ]
    if isinstance(rew_wrapper, str):
        rew_wrapper: Type[gym.RewardWrapper] = ut.getRewardWrapperFromName(rew_wrapper)
    if isinstance(obs_wrapper, str):
        obs_wrapper: Type[gym.ObservationWrapper] = ut.getObservationWrapperFromName(
            obs_wrapper
        )
    if isinstance(routing_func, str):
        routing_func: Callable = ut.getRoutingFunction(routing_func)

    # Check the validity of the values first
    checkTrainingValues(
        model_dir=model_dir,
        log_dir=log_dir,
        log_name=log_name,
        RL_model_name=RL_model_name,
        env=env,
        render=render,
        numbOfNodes=numbOfNodes,
        numbOfJammers=numbOfJammers,
        numbOfUsers=numbOfUsers,
        seed=seed,
        networkConnectedAtStart=networkConnectedAtStart,
        jammersSpawnNextToUsers=jammersSpawnNextToUsers,
        allow_early_ep_finish=allow_early_ep_finish,
        attackerModel=attackerModel,
        steps_till_jammer_active=steps_till_jammer_active,
        obs_wrapper=obs_wrapper,
        obs_config=obs_config,
        rew_wrapper=rew_wrapper,
        timesteps=timesteps,
        ep_length=ep_length,
        save_freq=save_freq,
        stat_size=stat_size,
    )

    # Set rendermode (disabled when using parallel envs)
    render_mode = getRenderMode(render) if n_envs == 1 else None

    # Create and wrap the env — vectorised when n_envs > 1
    if n_envs > 1:
        wrapped_env = ut.make_vec_env(
            n_envs=n_envs,
            env_constructor=env,
            rew_wrapper=rew_wrapper,
            obs_wrapper=obs_wrapper,
            numbOfNodes=numbOfNodes,
            numbOfJammers=numbOfJammers,
            numbOfUsers=numbOfUsers,
            seed=seed,
            networkConnectedAtStart=networkConnectedAtStart,
            jammersSpawnNextToUsers=jammersSpawnNextToUsers,
            allow_early_ep_finish=allow_early_ep_finish,
            attackerModel=attackerModel,
            steps_till_jammer_active=steps_till_jammer_active,
            obs_config=obs_config,
            ep_length=ep_length,
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
            obs_type=obs_type,
            state_loss_rate=state_loss_rate,
        )
    else:
        wrapped_env = ut.createAndWrapEnv(
            env_constructor=env,
            render_mode=render_mode,
            numbOfNodes=numbOfNodes,
            numbOfJammers=numbOfJammers,
            numbOfUsers=numbOfUsers,
            seed=seed,
            networkConnectedAtStart=networkConnectedAtStart,
            jammersSpawnNextToUsers=jammersSpawnNextToUsers,
            allow_early_ep_finish=allow_early_ep_finish,
            attackerModel=attackerModel,
            steps_till_jammer_active=steps_till_jammer_active,
            step_jammers_start_moving=step_jammers_start_moving,
            obs_wrapper=obs_wrapper,
            obs_config=obs_config,
            rew_wrapper=rew_wrapper,
            ep_length=ep_length,
            obs_type=obs_type,
            state_loss_rate=state_loss_rate,
            routing_func=routing_func,
            step_size=step_size,
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

    # Set storing and logging directories and names
    log_path, tb_logname, model_path = ut.setDirectories(
        RL_model_name=RL_model_name,
        model_dir=model_dir,
        log_dir=log_dir,
        log_name=log_name,
        numbOfNodes=numbOfNodes,
        numbOfJammers=numbOfJammers,
        numbOfUsers=numbOfUsers,
        ep_length=ep_length,
    )

    # Custom callback ensures frequent saving of RL_model and episodic reset
    # of the environment to sample different states concerning the users
    # Divide save_freq by n_envs: CheckpointCallback counts calls (one per
    # parallel step), so each call advances n_envs environment steps at once.
    callback = CheckpointCallback(save_freq=max(1, save_freq // n_envs), save_path=model_path)

    # Set the agent
    agent = ut.get_model(RL_model_name, wrapped_env, stat_size, log_path, seed=seed)

    # Store config
    ut.document_as_nested_config(
        path=model_path,
        file_name="01_training_config.yaml",
        # Values to store
        model_path=model_path,
        log_path=log_path,
        tb_logname=tb_logname,
        RL_model_name=RL_model_name,
        env=env.__name__,
        render=render,
        numbOfNodes=numbOfNodes,
        numbOfJammers=numbOfJammers,
        numbOfUsers=numbOfUsers,
        seed=seed,
        networkConnectedAtStart=networkConnectedAtStart,
        jammersSpawnNextToUsers=jammersSpawnNextToUsers,
        allow_early_ep_finish=allow_early_ep_finish,
        attackerModel=ut.getAttackerModelNameFromInstance(attackerModel),
        steps_till_jammer_active=steps_till_jammer_active,
        step_jammers_start_moving=step_jammers_start_moving,
        obs_wrapper=obs_wrapper.__name__,
        obs_config=obs_config,
        rew_wrapper=rew_wrapper.__name__,
        timesteps=timesteps,
        ep_length=ep_length,
        save_freq=save_freq,
        stat_size=stat_size,
        n_envs=n_envs,
        routing_func=routing_func.__name__,
        step_size=step_size,
        numbOfByzantineNodes=numbOfByzantineNodes,
        byzantine_drop_rate=byzantine_drop_rate,
        byzantine_attack_type=byzantine_attack_type,
        byzantine_capacity_inflation_factor=byzantine_capacity_inflation_factor,
        byzantine_jamming_power_factor=byzantine_jamming_power_factor,
        byzantine_pos_offset=byzantine_pos_offset,
    )

    # Start learning
    agent.learn(
        total_timesteps=timesteps,
        progress_bar=True,
        reset_num_timesteps=True,
        callback=callback,
        tb_log_name=tb_logname,
    )


def retrain_agent(
    dir_of_agent_to_load: str,
    steps_of_agent_to_load: int = None,
    # storing and logging directories and names
    model_dir: str = None,
    log_dir: str = None,
    log_name: str = None,
    # env
    env: Union[str, Type[envs.Basic_MANETEnv]] = "Static_MANETEnv",
    # whether to render env
    render: bool = None,
    # env parameters
    numbOfJammers: Union[int, Tuple[int, int]] = None,
    seed: int = None,
    networkConnectedAtStart: bool = None,
    jammersSpawnNextToUsers: bool = None,
    allow_early_ep_finish: bool = None,
    routing_func: Union[Callable, str] = MaxFlow.sequential_max_flow,
    step_size: int = None,
    # attacker model
    attackerModel: Union[
        Type[at.AttackerModel], List[Type[at.AttackerModel]], str, List[str]
    ] = None,
    steps_till_jammer_active: int = None,
    step_jammers_start_moving: int = None,
    # obs config
    obs_config: dict = None,
    # rew wrapper
    rew_wrapper: Union[str, Type[gym.RewardWrapper]] = None,
    # training params
    timesteps: int = None,
    ep_length: int = None,
    save_freq: int = None,
    stat_size: int = None,
    reset: bool = False,
    # byzantine obs-manipulation params
    byzantine_obs_user_dir_offset=None,
    byzantine_obs_capacity_factor: float = None,
    byzantine_obs_demand_factor: float = None,
    byzantine_obs_interference_factor: float = None,
    byzantine_obs_noise_std: float = None,
    byzantine_obs_capacity_deflate_factor: float = None,
    byzantine_obs_demand_deflate_factor: float = None,
    byzantine_obs_replay_delay: int = None,
    byzantine_obs_magnitude_range=None,
    # obs type and state loss rate
    obs_type: str = None,
    state_loss_rate: float = None,
    # parallel envs
    n_envs: int = 1,
    # To handle newer configs
    model_path: str = None,
    log_path: str = None,
    tb_logname: str = None,
) -> None:
    """
    Retrains the agent with the specified parameters.

    Args:
        dir_of_agent_to_load (str): Directory of the model to load.
        steps_of_agent_to_load (int, optional): Steps of the model to load.

        model_dir (str, optional): Directory to save the model.
        log_dir (str, optional): Directory to save the logs.
        log_name (str, optional): Name of the log file.

        env (Union[str, Type[envs.Basic_MANETEnv]]): The MANET environment class.

        render (bool, optional): Whether to render the environment.

        numbOfJammers (int, optional): Number of jammers in the environment.
        seed (int, optional): Seed for random number generation.
        networkConnectedAtStart (bool, optional): Whether the network is connected at the start.
        jammersSpawnNextToUsers (bool, optional): Whether jammers spawn next to users.
        allow_early_ep_finish (bool, optional): Whether to allow early episode finish.
        routing_func (Union[Callable, str]): How to route the data in the network.
        step_size (int): Maximum step size of MANET node. Since, per default an action is in [-1,1]^2 it simply scales the action.

        attackerModel (Union[str, Type[at.AttackerModel]], optional): Attacker model to use.
        steps_till_jammer_active (int, optional): Steps until jammer becomes active.
        step_jammers_start_moving (int, optional): Step from which on the jammers start movig to position themselves.

        numbOfByzantineNodes (int, optional): Number of MANET nodes that behave as Byzantine attackers.
        byzantine_drop_rate (float, optional): Fraction of traffic each Byzantine node drops.
        byzantine_attack_type (str, optional): The type of Byzantine attack.

        obs_config (dict, optional): Observation configuration.
        rew_wrapper (Union[str, Type[gym.RewardWrapper]], optional): Reward wrapper to use.

        timesteps (int, optional): Number of timesteps for training.
        ep_length (int, optional): Length of each episode.
        save_freq (int, optional): Frequency of saving the model.
        stat_size (int, optional): Size of the statistics buffer.

    Returns:
        None

    Notes:
        This function retrains a reinforcement learning agent using a pre-trained model,
        allowing customization of various training parameters and configurations.
    """

    print("Loading conf...")
    file_name = (
        "01_training_config.yaml"
        if ut.hasSpecificConfigFile(dir_of_agent_to_load, "01_training_config.yaml")
        else ut.getMostRecentConfigFile(dir_of_agent_to_load)
    )
    config = ut.loadConfig(dir_of_agent_to_load, file_name)

    print("Successfully loaded config.")

    # Here we do the config config
    (
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
        _byzantine_capacity_inflation_factor,
        _byzantine_jamming_power_factor,
        _byzantine_pos_offset,
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
    ) = ut.loadParametersFromConfig(
        env=env,
        config=config,
        model_dir=model_dir,
        log_dir=log_dir,
        log_name=log_name,
        render=render,
        numbOfJammers=numbOfJammers,
        seed=seed,
        networkConnectedAtStart=networkConnectedAtStart,
        jammersSpawnNextToUsers=jammersSpawnNextToUsers,
        allow_early_ep_finish=allow_early_ep_finish,
        attackerModel=attackerModel,
        steps_till_jammer_active=steps_till_jammer_active,
        obs_config=obs_config,
        rew_wrapper=rew_wrapper,
        timesteps=timesteps,
        ep_length=ep_length,
        save_freq=save_freq,
        stat_size=stat_size,
        reset=reset,
        routing_func=routing_func,
        step_size=step_size,
        step_jammers_start_moving=step_jammers_start_moving,
    )

    # Resolve class constructors if they are given as strings
    if isinstance(attackerModel, str):
        attackerModel: Union[Type[at.AttackerModel], List[Type[at.AttackerModel]]] = (
            ut.getAttackerModelFromName(attackerModel)
        )
    elif isinstance(attackerModel, list) and all(
        isinstance(item, str) for item in attackerModel
    ):
        attackerModel: Union[Type[at.AttackerModel], List[Type[at.AttackerModel]]] = [
            ut.getAttackerModelFromName(item) for item in attackerModel
        ]
    if isinstance(rew_wrapper, str):
        rew_wrapper = ut.getRewardWrapperFromName(rew_wrapper)
    if isinstance(obs_wrapper, str):
        obs_wrapper = ut.getObservationWrapperFromName(obs_wrapper)
    if isinstance(routing_func, str):
        routing_func: Callable = ut.getRoutingFunction(routing_func)
    if isinstance(env, str):
        env: Type[envs.Basic_MANETEnv] = ut.getEnvironmentFromName(env)

    # Load model
    path_of_agent_to_load = ut.getPathOfAgent(
        dir_of_agent_to_load, steps_of_agent_to_load
    )

    # Check the validity of the values first
    checkTrainingValues(
        model_dir=model_path,
        log_dir=log_path,
        log_name=tb_logname,
        RL_model_name=RL_model_name,
        env=env,
        render=render,
        numbOfJammers=numbOfJammers,
        seed=seed,
        networkConnectedAtStart=networkConnectedAtStart,
        jammersSpawnNextToUsers=jammersSpawnNextToUsers,
        allow_early_ep_finish=allow_early_ep_finish,
        attackerModel=attackerModel,
        steps_till_jammer_active=steps_till_jammer_active,
        obs_wrapper=obs_wrapper,
        obs_config=obs_config,
        rew_wrapper=rew_wrapper,
        timesteps=timesteps,
        ep_length=ep_length,
        save_freq=save_freq,
        stat_size=stat_size,
    )

    # Create and wrap the env — vectorised when n_envs > 1
    render_mode = getRenderMode(render) if n_envs == 1 else None
    if n_envs > 1:
        wrapped_env = ut.make_vec_env(
            n_envs=n_envs,
            env_constructor=env,
            rew_wrapper=rew_wrapper,
            obs_wrapper=obs_wrapper,
            numbOfNodes=numbOfNodes,
            numbOfJammers=numbOfJammers,
            numbOfUsers=numbOfUsers,
            seed=seed,
            networkConnectedAtStart=networkConnectedAtStart,
            jammersSpawnNextToUsers=jammersSpawnNextToUsers,
            allow_early_ep_finish=allow_early_ep_finish,
            attackerModel=attackerModel,
            steps_till_jammer_active=steps_till_jammer_active,
            obs_config=obs_config,
            ep_length=ep_length,
            routing_func=routing_func,
            step_size=step_size,
            step_jammers_start_moving=step_jammers_start_moving,
            numbOfByzantineNodes=numbOfByzantineNodes,
            byzantine_drop_rate=byzantine_drop_rate,
            byzantine_attack_type=byzantine_attack_type,
            byzantine_obs_user_dir_offset=byzantine_obs_user_dir_offset,
            byzantine_obs_capacity_factor=byzantine_obs_capacity_factor,
            byzantine_obs_demand_factor=byzantine_obs_demand_factor,
            byzantine_obs_interference_factor=byzantine_obs_interference_factor,
            byzantine_obs_noise_std=byzantine_obs_noise_std,
            byzantine_obs_capacity_deflate_factor=byzantine_obs_capacity_deflate_factor,
            byzantine_obs_demand_deflate_factor=byzantine_obs_demand_deflate_factor,
            byzantine_obs_replay_delay=byzantine_obs_replay_delay,
            byzantine_obs_magnitude_range=byzantine_obs_magnitude_range,
            obs_type=obs_type,
            state_loss_rate=state_loss_rate,
        )
    else:
        wrapped_env = ut.createAndWrapEnv(
            env_constructor=env,
            render_mode=render_mode,
            numbOfNodes=numbOfNodes,
            numbOfJammers=numbOfJammers,
            numbOfUsers=numbOfUsers,
            seed=seed,
            networkConnectedAtStart=networkConnectedAtStart,
            jammersSpawnNextToUsers=jammersSpawnNextToUsers,
            allow_early_ep_finish=allow_early_ep_finish,
            attackerModel=attackerModel,
            steps_till_jammer_active=steps_till_jammer_active,
            step_jammers_start_moving=step_jammers_start_moving,
            obs_wrapper=obs_wrapper,
            obs_config=obs_config,
            rew_wrapper=rew_wrapper,
            ep_length=ep_length,
            routing_func=routing_func,
            step_size=step_size,
            numbOfByzantineNodes=numbOfByzantineNodes,
            byzantine_drop_rate=byzantine_drop_rate,
            byzantine_attack_type=byzantine_attack_type,
            byzantine_obs_user_dir_offset=byzantine_obs_user_dir_offset,
            byzantine_obs_capacity_factor=byzantine_obs_capacity_factor,
            byzantine_obs_demand_factor=byzantine_obs_demand_factor,
            byzantine_obs_interference_factor=byzantine_obs_interference_factor,
            byzantine_obs_noise_std=byzantine_obs_noise_std,
            byzantine_obs_capacity_deflate_factor=byzantine_obs_capacity_deflate_factor,
            byzantine_obs_demand_deflate_factor=byzantine_obs_demand_deflate_factor,
            byzantine_obs_replay_delay=byzantine_obs_replay_delay,
            byzantine_obs_magnitude_range=byzantine_obs_magnitude_range,
            obs_type=obs_type,
            state_loss_rate=state_loss_rate,
        )

    # Divide save_freq by n_envs: CheckpointCallback counts calls (one per
    # parallel step), so each call advances n_envs environment steps at once.
    callback = CheckpointCallback(save_freq=max(1, save_freq // n_envs), save_path=model_path)

    # Set the agent
    agent = ut.get_model(
        RL_model_name, wrapped_env, stat_size, log_path, seed=seed
    ).load(path=path_of_agent_to_load, env=wrapped_env, tensorboard_log=log_path)

    # To store retrain config with incremented number if one already exists.
    config_store_name = getRetrainConfigName(dir_of_agent_to_load=dir_of_agent_to_load)

    # Store config
    ut.document_as_nested_config(
        path=model_path,
        file_name=config_store_name,
        # Values to store
        dir_of_agent_to_load=dir_of_agent_to_load,
        steps_of_agent_to_load=ut.getStepsFromFileName(
            os.path.basename(path_of_agent_to_load)
        ),
        model_path=model_path,
        log_path=log_path,
        tb_logname=tb_logname,
        RL_model_name=RL_model_name,
        env=env.__name__,
        render=render,
        numbOfNodes=numbOfNodes,
        numbOfJammers=numbOfJammers,
        numbOfUsers=numbOfUsers,
        seed=seed,
        networkConnectedAtStart=networkConnectedAtStart,
        jammersSpawnNextToUsers=jammersSpawnNextToUsers,
        allow_early_ep_finish=allow_early_ep_finish,
        attackerModel=ut.getAttackerModelNameFromInstance(attackerModel),
        steps_till_jammer_active=steps_till_jammer_active,
        obs_wrapper=obs_wrapper.__name__,
        obs_config=obs_config,
        rew_wrapper=rew_wrapper.__name__,
        timesteps=timesteps,
        ep_length=ep_length,
        save_freq=save_freq,
        stat_size=stat_size,
        reset=reset,
        routing_func=routing_func.__name__,
        step_size=step_size,
        numbOfByzantineNodes=numbOfByzantineNodes,
        byzantine_drop_rate=byzantine_drop_rate,
        byzantine_attack_type=byzantine_attack_type,
    )

    # Start learning
    agent.learn(
        total_timesteps=timesteps,
        progress_bar=True,
        reset_num_timesteps=reset,
        callback=callback,
        tb_log_name=tb_logname,
    )


# * Helper Methods
def checkTrainingValues(
    # log name
    log_name: str = None,
    log_dir: str = None,
    model_dir: str = None,
    # model
    RL_model_name: str = None,
    # Env
    env: Type[envs.Basic_MANETEnv] = None,
    # whether to render env
    render: bool = None,
    # env parameters
    numbOfNodes: int = None,
    numbOfJammers: int = None,
    numbOfUsers: int = None,
    seed: int = None,
    networkConnectedAtStart: bool = None,
    jammersSpawnNextToUsers: bool = None,
    allow_early_ep_finish: bool = None,
    # attacker model
    attackerModel: Type[at.AttackerModel] = None,
    steps_till_jammer_active: int = None,
    # obswrapper
    obs_wrapper: Type[gym.ObservationWrapper] = None,
    # obs config
    obs_config: dict = None,
    # rew wrapper
    rew_wrapper: Type[gym.RewardWrapper] = None,
    # training params
    timesteps: int = None,
    ep_length: int = None,
    save_freq: int = None,
    stat_size: int = None,
) -> None:
    """
    Checks the validity of the input values for training the agent.

    Args:
        log_name (str): Name of the log file.
        RL_model_name (str): The RL model's name one wants to use.
        env (Type[envs.Basic_MANETEnv]): The MANET environment class.
        render (bool): Whether to render the environment.
        numbOfNodes (int): Number of nodes in the environment.
        numbOfJammers (int): Number of jammers in the environment.
        numbOfUsers (int): Number of users in the environment.
        seed (int): Random seed for reproducibility.
        networkConnectedAtStart (bool): Whether the network is connected at the start.
        jammersSpawnNextToUsers (bool): Whether jammers spawn next to users.
        allow_early_ep_finish (bool): Whether to allow early episode finish.
        attackerModel (Type[at.AttackerModel]): The attacker model to use.
        steps_till_jammer_active (int): Steps until jammers become active.
        obs_wrapper (Type[gym.ObservationWrapper]): The observation wrapper to use.
        obs_config (dict): Configuration for the observation wrapper.
        rew_wrapper (Type[gym.RewardWrapper]): The reward wrapper to use.
        timesteps (int): Number of timesteps to train.
        ep_length (int): Length of each episode.
        save_freq (int): Frequency to save the model.
        stat_size (int): Size of the statistics.

    Returns:
        None

    Notes:
        This function validates the input parameters required for training a reinforcement
        learning agent, ensuring all necessary values are provided and meet expected criteria.
    """

    # Check log and models dir and logname
    if log_name is not None:
        assert isinstance(log_name, str), "log_name must be a string."
    if log_dir is not None:
        assert isinstance(log_dir, str), "log_dir must be a string."
    if model_dir is not None:
        assert isinstance(model_dir, str), "model_dir must be a string."

    # Check model exists in utils
    if RL_model_name is not None:
        assert (
            RL_model_name in ut.get_available_models()
        ), f"RL_model '{RL_model_name}' is not available."

    # Check env is child of Basic_MANETEnv
    if env is not None:
        assert issubclass(
            env, envs.Basic_MANETEnv
        ), "env must be a subclass of Basic_MANETEnv."

    # Check render is of type bool
    if render is not None:
        assert isinstance(render, bool), "render must be a boolean."

    # Check numbers of components are int or range except nodes only int
    if numbOfNodes is not None:
        assert numbOfNodes >= 0, "Number of nodes must be greater than or equal to 0."
    if numbOfJammers is not None:
        if isinstance(numbOfJammers, tuple) or isinstance(numbOfJammers, list):
            assert (
                len(numbOfJammers) == 2
            ), "numbOfJammers tuple must have exactly two elements."
            assert all(
                isinstance(x, int) and x >= 0 for x in numbOfJammers
            ), "Both elements of numbOfJammers tuple must be integers >= 0."
        else:
            assert (
                isinstance(numbOfJammers, int) and numbOfJammers >= 0
            ), "numbOfJammers must be an integer >= 0 or a tuple of integers >= 0."

    if numbOfUsers is not None:
        if isinstance(numbOfUsers, tuple) or isinstance(numbOfUsers, list):
            assert (
                len(numbOfUsers) == 2
            ), "numbOfUsers tuple must have exactly two elements."
            assert all(
                isinstance(x, int) and x > 0 for x in numbOfUsers
            ), "Both elements of numbOfUsers tuple must be integers > 0."
        else:
            assert (
                isinstance(numbOfUsers, int) and numbOfUsers > 0
            ), "numbOfUsers must be an integer > 0 or a tuple of integers > 0."
    # Check seed is int
    if seed is not None:
        assert isinstance(seed, int), "seed must be an integer."

    # Check networkConnectedAtStart is bool
    if networkConnectedAtStart is not None:
        assert isinstance(
            networkConnectedAtStart, bool
        ), "networkConnectedAtStart must be a boolean."

    # Check jammersSpawnNextToUsers is bool
    if jammersSpawnNextToUsers is not None:
        assert isinstance(
            jammersSpawnNextToUsers, bool
        ), "jammersSpawnNextToUsers must be a boolean."

    # Check allow_early_ep_finish is bool
    if allow_early_ep_finish is not None:
        assert isinstance(
            allow_early_ep_finish, bool
        ), "allow_early_ep_finish must be a boolean."

    # Check obs_wrapper is child of gym.ObservationWrapper
    if obs_wrapper is not None:
        assert issubclass(
            obs_wrapper, gym.ObservationWrapper
        ), "obs_wrapper must be a subclass of gym.ObservationWrapper."

    # Check reward_wrapper is child of gym.RewardWrapper
    if rew_wrapper is not None:
        assert issubclass(
            rew_wrapper, gym.RewardWrapper
        ), "rew_wrapper must be a subclass of gym.RewardWrapper."

    # Check training params are valid and int
    if timesteps is not None:
        assert timesteps >= 0, "timesteps must be greater than or equal to 0."
    if ep_length is not None:
        assert ep_length >= 0, "ep_length must be greater than or equal to 0."
    if save_freq is not None:
        assert save_freq >= 0, "save_freq must be greater than or equal to 0."
    if steps_till_jammer_active is not None:
        assert (
            steps_till_jammer_active >= 0
        ), "steps_till_jammer_active must be greater than or equal to 0."
        assert (
            steps_till_jammer_active <= ep_length
        ), "steps_till_jammer_active must be less than or equal to ep_length."
    if stat_size is not None:
        assert isinstance(stat_size, int), "stat_size must be integer"
        assert stat_size > 0, "stat_size must be a natural number"


def getRenderMode(render: bool) -> str:
    """
    Determines the render mode based on the render flag.

    Args:
        render (bool): Whether to render the environment.

    Returns:
        str: The render mode.

    Notes:
        This function evaluates the render flag and returns an appropriate string
        representing the render mode for the environment configuration.
    """

    render_mode = "plot" if render else None
    return render_mode


def getRetrainConfigName(dir_of_agent_to_load: str) -> str:
    """
    Find the retrain config file with the highest number in dir_of_agent_to_load
    and return one with an incremented number.

    Args:
        dir_of_agent_to_load (str): Directory containing the agent config files.

    Returns:
        str: Name of the new retrain config file with an incremented number.
    """
    config_files = os.listdir(dir_of_agent_to_load)
    pattern = re.compile(r"(\d+)_.*_config\.yaml")

    max_number = 0
    for config_file in config_files:
        match = pattern.match(config_file)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number

    new_number = max_number + 1
    new_config_file = f"{new_number:02d}_retraining_config.yaml"
    return new_config_file
