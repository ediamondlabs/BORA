# OS
import os, re
import shutil
from pathlib import Path

# File type
import csv
import json

# Table for quick mean statistics display
from tabulate import tabulate

# Base libs
from collections import defaultdict
import numpy as np

# Class annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Type hinting
from typing import Callable, Union, Optional, Tuple, Type, List, Dict

# Gymnasium for RL-Env
import gymnasium as gym

# Stable baselines for training
from stable_baselines3.common.base_class import BaseAlgorithm

# Helper libs
import Utils.training_utils as ut
from Utils.batch_utils import process_string

# Own MANET Classes
import MANET.ObservationWrappers as ow
import MANET.Environments as envs
import MANET.RewardWrappers as rw
import MANET.Attackers as at
import MANET.ConditionStrategy as cs
import MANET.Agents as ag
import MANET.Components as net_comps
from MANET.Routing import MaxFlow


@dataclass
class EvalParams:
    """
    Evaluation parameters for MANET evaluation.

    Attributes:
        n_eval_eps (int): Number of episodes for which the evaluation is conducted. Default is 20.
        ep_length (int): Maximum number of steps per episode. Default is 1,000.
        jammer_ep_steps (int): Number of steps before jammers become active. Default is 0.
        jammers_preposition (bool): Whether jammers are prepositioned before evaluation starts. Default is False.
        seed (int): Random seed for reproducibility. Default is 42.
        deterministic (bool): Whether the agent's actions are deterministic. Default is True.
        obs_type (str): Type of observation used by the agent (e.g., "full", "partial"). Default is "full".
        state_loss_rate (Optional[float]): Rate of state loss in the environment. Default is None.
        routing_func (Callable): Routing function used for network communication. Default is MaxFlow.sequential_max_flow.
        numbOfJammers (Optional[Union[int, Tuple[int, int]]]): Number of jammers in the environment.
            If an interval is provided, the number is sampled uniformly at the start of each episode and remains constant during the episode. Default is None.
        numbOfUsers (Optional[Union[int, Tuple[int, int]]]): Number of users in the environment.
            If an interval is provided, the number is sampled uniformly at the start of each episode and remains constant during the episode. Default is None.
        render_mode (Optional[str]): Rendering mode for the environment (e.g., "human", "rgb_array"). Default is None.
        networkConnectedAtStart (bool): Whether the network is connected at the start of the evaluation. Default is False.
        jammersSpawnNextToUsers (bool): Whether jammers spawn near users. Default is False.
        attackerModel (Type[at.AttackerModel]): Model used by the attackers (e.g., Static_GreedyJammers). Default is at.Static_GreedyJammers.
        allow_early_ep_finish (bool): Whether episodes can end early if certain conditions are met. Default is False.
        step_size (int): Step size for node movement. Default is 2.
        step_jammers_start_moving (int): Step at which jammers start moving. Default is 0.
        metric_param (float): Weighting parameter for quick evaluation metrics. Default is 0.75.
        filterval (float): Divisor used to filter saved models. Only models stored at steps divisible by this value are evaluated. Default is 0.9.
        numbOfByzantineNodes (Optional[int]): Number of Byzantine nodes injected into the environment during evaluation.
            If None, no Byzantine nodes are added. Default is None.
        byzantine_drop_rate (float): Fraction of traffic a greyhole/sinkhole Byzantine node drops (0.0–1.0). Default is 0.5.
        byzantine_attack_type (str or List[str]): Attack type for Byzantine nodes. Either a single
            type applied to all Byzantine nodes, or a list of types assigned round-robin
            (e.g., ["greyhole", "selective_jamming"]). Default is "greyhole".
        byzantine_capacity_inflation_factor (float): Capacity multiplier advertised by sinkhole nodes. Default is 3.0.
        byzantine_jamming_power_factor (float): Interference multiplier for selective_jamming nodes. Default is 1.0.
        byzantine_pos_offset (Optional[List]): 2-D position offset [dx, dy] for position_spoofing nodes. Default is None.
        byzantine_obs_user_dir_offset (Optional[List]): 2-D direction offset [dx, dy] added to all direction fields broadcast by obs_user_dir nodes. Default is None.
        byzantine_obs_capacity_factor (float): Capacity scale factor for obs_capacity nodes. Default is 1.0.
        byzantine_obs_demand_factor (float): Demand scale factor for obs_demand nodes. Default is 1.0.
        byzantine_obs_interference_factor (float): Interference scale factor for obs_interference_lie nodes (0.0 hides all interference). Default is 1.0.
        byzantine_obs_noise_std (float): Std dev of Gaussian noise added to all broadcast observation fields by obs_noise nodes. Default is 0.0.
    """

    n_eval_eps: int = 20
    ep_length: int = 1_000
    jammer_ep_steps: int = 0
    jammers_preposition: bool = False
    seed: int = 42
    deterministic: bool = True
    obs_type: str = "full"
    state_loss_rate: Optional[float] = None
    routing_func: Callable = MaxFlow.sequential_max_flow
    numbOfJammers: Optional[Union[int, Tuple[int, int]]] = None
    numbOfUsers: Optional[Union[int, Tuple[int, int]]] = None
    render_mode: Optional[str] = None
    networkConnectedAtStart: bool = False
    jammersSpawnNextToUsers: bool = False
    attackerModel: Type[at.AttackerModel] = at.Static_GreedyJammers
    allow_early_ep_finish: bool = False
    step_size: int = 2
    step_jammers_start_moving: int = 0
    metric_param: float = 0.75
    filterval: float = 0.9
    numbOfByzantineNodes: Optional[int] = None
    byzantine_drop_rate: float = 0.5
    byzantine_attack_type: Union[str, List[str]] = "greyhole"
    byzantine_capacity_inflation_factor: float = 3.0
    byzantine_jamming_power_factor: float = 1.0
    byzantine_pos_offset: Optional[List] = None
    byzantine_obs_user_dir_offset: Optional[List] = None
    byzantine_obs_capacity_factor: float = 1.0
    byzantine_obs_demand_factor: float = 1.0
    byzantine_obs_interference_factor: float = 1.0
    byzantine_obs_noise_std: float = 0.0
    byzantine_obs_capacity_deflate_factor: float = 1.0
    byzantine_obs_demand_deflate_factor: float = 1.0
    byzantine_obs_replay_delay: int = 0
    byzantine_obs_magnitude_range: Optional[List] = None
    n_envs: int = 1

    def _convert_tuples_to_lists(self) -> None:
        """Convert tuple values to lists for attackerModel, numbOfUsers, and numbOfJammers."""
        if isinstance(self.attackerModel, tuple):
            self.attackerModel = list(self.attackerModel)

        if isinstance(self.numbOfUsers, tuple):
            self.numbOfUsers = list(self.numbOfUsers)

        if isinstance(self.numbOfJammers, tuple):
            self.numbOfJammers = list(self.numbOfJammers)

    @classmethod
    def from_dict(cls, params: Dict) -> "EvalParams":
        """Create EvalParams from a dictionary."""
        routing_func = params.get("routing_func", MaxFlow.sequential_max_flow)
        if isinstance(routing_func, str):
            routing_func = ut.getRoutingFunction(routing_func)
        attacker_model = params.get("attackerModel", at.Static_GreedyJammers)
        if isinstance(attacker_model, str) or isinstance(attacker_model, list):
            attacker_model = ut.getAttackerModelFromName(attacker_model)
        instance = cls(
            n_eval_eps=params.get("n_eval_eps", 20),
            ep_length=params.get("ep_length", 1_000),
            jammer_ep_steps=params.get("jammer_ep_steps", 0),
            seed=params.get("seed", 42),
            deterministic=params.get("deterministic", True),
            obs_type=params.get("obs_type", "full"),
            state_loss_rate=params.get("state_loss_rate"),
            routing_func=routing_func,
            numbOfJammers=params.get("numbOfJammers"),
            numbOfUsers=params.get("numbOfUsers"),
            render_mode=params.get("render_mode"),
            networkConnectedAtStart=params.get("networkConnectedAtStart", False),
            jammersSpawnNextToUsers=params.get("jammersSpawnNextToUsers", False),
            attackerModel=attacker_model,
            allow_early_ep_finish=params.get("allow_early_ep_finish", False),
            step_size=params.get("step_size", 2),
            step_jammers_start_moving=params.get("step_jammers_start_moving", 0),
            metric_param=params.get("metric_param", 0.9),
            filterval=int(params.get("filterval", 2e6)),
            jammers_preposition=params.get("jammers_preposition", False),
            numbOfByzantineNodes=params.get("numbOfByzantineNodes"),
            byzantine_drop_rate=params.get("byzantine_drop_rate", 0.5),
            byzantine_attack_type=params.get("byzantine_attack_type", "greyhole"),
            byzantine_capacity_inflation_factor=params.get("byzantine_capacity_inflation_factor", 3.0),
            byzantine_jamming_power_factor=params.get("byzantine_jamming_power_factor", 1.0),
            byzantine_pos_offset=params.get("byzantine_pos_offset"),
            byzantine_obs_user_dir_offset=params.get("byzantine_obs_user_dir_offset"),
            byzantine_obs_capacity_factor=params.get("byzantine_obs_capacity_factor", 1.0),
            byzantine_obs_demand_factor=params.get("byzantine_obs_demand_factor", 1.0),
            byzantine_obs_interference_factor=params.get("byzantine_obs_interference_factor", 1.0),
            byzantine_obs_noise_std=params.get("byzantine_obs_noise_std", 0.0),
            byzantine_obs_capacity_deflate_factor=params.get("byzantine_obs_capacity_deflate_factor", 1.0),
            byzantine_obs_demand_deflate_factor=params.get("byzantine_obs_demand_deflate_factor", 1.0),
            byzantine_obs_replay_delay=int(params.get("byzantine_obs_replay_delay", 0)),
            byzantine_obs_magnitude_range=params.get("byzantine_obs_magnitude_range"),
            n_envs=params.get("n_envs", 1),
        )
        instance._convert_tuples_to_lists()
        return instance

    def to_dict(self) -> Dict:
        """Convert EvalParams to a dictionary."""
        return {
            "n_eval_eps": self.n_eval_eps,
            "ep_length": self.ep_length,
            "jammer_ep_steps": self.jammer_ep_steps,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "obs_type": self.obs_type,
            "state_loss_rate": self.state_loss_rate,
            "routing_func": (
                self.routing_func.__name__
                if callable(self.routing_func)
                else self.routing_func
            ),
            "step_size": self.step_size,
            "numbOfJammers": self.numbOfJammers,
            "numbOfUsers": self.numbOfUsers,
            "render_mode": self.render_mode,
            "networkConnectedAtStart": self.networkConnectedAtStart,
            "jammersSpawnNextToUsers": self.jammersSpawnNextToUsers,
            "attackerModel": ut.getAttackerModelNameFromInstance(self.attackerModel),
            "allow_early_ep_finish": self.allow_early_ep_finish,
            "metric_param": self.metric_param,
            "filterval": int(self.filterval),
            "jammers_preposition": self.jammers_preposition,
            "step_jammers_start_moving": self.step_jammers_start_moving,
            "numbOfByzantineNodes": self.numbOfByzantineNodes,
            "byzantine_drop_rate": self.byzantine_drop_rate,
            "byzantine_attack_type": self.byzantine_attack_type,
            "byzantine_capacity_inflation_factor": self.byzantine_capacity_inflation_factor,
            "byzantine_jamming_power_factor": self.byzantine_jamming_power_factor,
            "byzantine_pos_offset": self.byzantine_pos_offset,
            "byzantine_obs_user_dir_offset": self.byzantine_obs_user_dir_offset,
            "byzantine_obs_capacity_factor": self.byzantine_obs_capacity_factor,
            "byzantine_obs_demand_factor": self.byzantine_obs_demand_factor,
            "byzantine_obs_interference_factor": self.byzantine_obs_interference_factor,
            "byzantine_obs_noise_std": self.byzantine_obs_noise_std,
            "byzantine_obs_capacity_deflate_factor": self.byzantine_obs_capacity_deflate_factor,
            "byzantine_obs_demand_deflate_factor": self.byzantine_obs_demand_deflate_factor,
            "byzantine_obs_replay_delay": self.byzantine_obs_replay_delay,
            "byzantine_obs_magnitude_range": self.byzantine_obs_magnitude_range,
            "n_envs": self.n_envs,
        }


@dataclass
class EvalConfig:
    """Configuration for MANET evaluation.

    Attributes:
        env_constr (Type[envs.Basic_MANETEnv]): Constructor to create a Basic_MANETEnv,
            which is the environment in which the agents and baselines are evaluated.
        agent_patterns (List[str]): List of patterns to locate agent models for evaluation.
        baselines_with_cs (List[Tuple[Type[ag.AgentBase], cs.ConditionStrategy]]):
            List of tuples containing baseline agents and their associated condition strategies.
        eval_params (EvalParams): Evaluation parameters for the MANET environment.
        eval_type (str): Type of evaluation to be performed (e.g., "grouped_eval", "filter_best_agent").
        log_dir (str): Directory where evaluation logs will be stored.
        log_results (bool): Whether to log the evaluation results.
    """

    env_constr: Type[envs.Basic_MANETEnv]
    agent_patterns: List[str]
    baselines_with_cs: List[Tuple[Type[ag.AgentBase], cs.ConditionStrategy]]
    eval_params: EvalParams
    eval_type: str = ("grouped_eval",)
    log_dir: str = "out/evaluations"
    log_results: bool = True

    @classmethod
    def from_yaml(cls, config_path: Path) -> "EvalConfig":
        """Create EvalConfig from a YAML file."""
        config = ut.load_eval_config(config_path.parent, config_path.name)
        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: Dict) -> "EvalConfig":
        """Create EvalConfig from a dictionary."""
        return cls(
            env_constr=config["env_constr"],
            agent_patterns=config["agent_patterns"],
            baselines_with_cs=config["baselines_with_cs"],
            eval_params=EvalParams.from_dict(config["eval_params"]),
            eval_type=config.get("eval_type", "grouped_eval"),
            log_dir=config.get("log_dir", "out/evaluations"),
            log_results=config.get("log_results", True),
        )

    def to_dict(self) -> Dict:
        """Convert EvalConfig to a dictionary."""
        return {
            "env_constr": (
                self.env_constr.__name__
                if hasattr(self.env_constr, "__name__")
                else self.env_constr
            ),
            "agent_patterns": self.agent_patterns,
            "baselines_with_cs": [
                {
                    "baseline": baseline.__name__,
                    "cs": cs.__class__.__name__,
                    "threshold": cs.threshold,
                }
                for baseline, cs in self.baselines_with_cs
            ],
            "eval_params": self.eval_params.to_dict(),
            "eval_type": self.eval_type,
            "log_dir": self.log_dir,
            "log_results": self.log_results,
        }


class AgentResolver:
    """Resolves and groups agent paths.

    Attributes:
        agent_patterns (List[str]): List of patterns to locate agent models.
        seed (int): Random seed for reproducibility. Need to initiate the RL model with a seed.
        agent_paths (List[str]): List of resolved agent file paths.
        grouped_agents (Dict[Tuple, List[str]]): Dictionary grouping agent paths by their configuration.
        eval_params (EvalParams): Evaluation parameters used for grouping and filtering agents.
    """

    def __init__(self, agent_patterns: List[str], eval_params: EvalParams):
        """Initialize AgentResolver.

        Args:
            agent_patterns (List[str]): List of patterns to locate agent models.
            eval_params (EvalParams): Evaluation parameters used for evaluating and filtering agents.
        """
        self.agent_patterns: List[str] = agent_patterns
        self.seed: int = eval_params.seed
        self.agent_paths: List[str] = []
        self.grouped_agents: Dict[Tuple, List[str]] = defaultdict(list)
        self.eval_params = eval_params
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the resolver by filtering and grouping agent paths."""
        self._filter_paths()
        print("Paths found with config file:")
        for path in self.agent_paths:
            print(f"- {path}")
        self._group_paths()

    def load_agent(self, path: str, env: gym.Env) -> BaseAlgorithm:
        """Load an RL agent from the specified path.

        Args:
            path (str): Path to the agent model file.
            env (gym.Env): The environment in which the agent will operate.

        Returns:
            BaseAlgorithm: The loaded RL agent.
        """
        ag_config = self.load_agent_config(agent_path=path)
        model_name = ag_config.get("RL_model_name")
        model = ut.get_model(model_name=model_name, env=env, seed=self.seed)
        agent = model.load(path=path, env=env, seed=self.seed)
        return agent

    def load_agent_config(self, agent_path: str) -> dict:
        """Load the configuration of an agent from its path.

        Args:
            agent_path (str): Path to the agent model file.

        Returns:
            dict: The configuration dictionary of the agent.
        """
        folder_path = os.path.dirname(agent_path)
        file_name = (
            "01_training_config.yaml"
            if ut.hasSpecificConfigFile(folder_path, "01_training_config.yaml")
            else ut.getMostRecentConfigFile(folder_path)
        )
        agent_config = ut.loadConfig(folder_path, file_name)
        return agent_config

    def give_agent_name(self, agent_path: str, base_obs_type: str) -> str:
        """Generate a unique agent name based on the base observation type and the agent's path.

        Args:
            agent_path (str): The file path of the agent.
            base_obs_type (str): The base observation type.

        Returns:
            str: The generated agent name.
        """
        path = Path(agent_path)
        day_month = path.parts[-3]
        agent_name = f"{day_month}-{process_string(base_obs_type, max_length=2)}-{process_string(str(path.parent.name), max_length=3)}"
        return agent_name

    def _filter_paths(self) -> None:
        """Filter agent paths to include only those that exist and have a training configurations."""
        existing_paths = ut.findAgentsMatchingPattern(*self.agent_patterns)
        self.agent_paths = self._filter_paths_without_config(existing_paths)

    def _group_paths(self) -> None:
        """Group agent paths by the number of nodes, jammers, and users."""
        for agent_path in self.agent_paths:
            dir_path = os.path.dirname(agent_path)
            file_name = ut.getMostRecentConfigFile(dir_path)
            config = ut.loadConfig(dir_path, file_name)
            numbOfNodes = config["numbOfNodes"]
            numbOfJammers = self.eval_params.numbOfJammers
            numbOfJammers = (
                tuple(numbOfJammers)
                if not isinstance(numbOfJammers, int)
                else numbOfJammers
            )
            numbOfUsers = self.eval_params.numbOfUsers if self.eval_params.numbOfUsers is not None else config["numbOfUsers"]
            numbOfUsers = (
                tuple(numbOfUsers) if not isinstance(numbOfUsers, int) else numbOfUsers
            )
            key = (numbOfNodes, numbOfJammers, numbOfUsers)
            self.grouped_agents[key].append(agent_path)

    def _filter_paths_without_config(self, paths: List[str]) -> List[str]:
        """Filter out paths without training/retraining configuration files.

        Args:
            paths (List[str]): List of agent file paths.

        Returns:
            List[str]: List of paths with training/retraining configuration files.
        """
        verified_paths = []
        for path in paths:
            config_dir = os.path.dirname(path)
            if ut.hasConfigFiles(config_dir):
                verified_paths.append(path)
            else:
                print(f"No config found in {config_dir}")
        return verified_paths

    def make_agent_folder_only(self, agent_paths: List[str]) -> List[str]:
        """Convert agent file paths to their parent folder paths.

        Args:
            agent_paths (List[str]): List of file paths to agent models.

        Returns:
            List[str]: List of folder paths containing the agent files.
        """
        folder_paths = []
        for path in agent_paths:
            act_path = Path(path)
            if act_path.is_file:
                act_path = act_path.parent
            folder_paths.append(str(act_path))
        return folder_paths


class BaselineResolver:
    """Resolves baseline configurations and provides methods to load them."""

    def __init__(
        self,
        baselines_with_cs: List[Tuple[Type[ag.AgentBase], cs.ConditionStrategy]],
        seed: int,
    ):
        """
        Args:
            baselines_with_cs (List[Tuple[Type[ag.AgentBase], cs.ConditionStrategy]]): List of tuples containing baseline agents and their condition strategies.
            seed (int): Random seed for reproducibility.
        """
        self.baselines_with_cs: List[
            Tuple[Type[ag.AgentBase], cs.ConditionStrategy]
        ] = baselines_with_cs
        self.seed: int = seed

    def load_baseline(
        self,
        baseline: Type[ag.AgentBase],
        condition_strategy: cs.ConditionStrategy,
        env: gym.Env,
    ) -> ag.AgentBase:
        """
        Load a baseline agent with its condition strategy.

        Args:
            baseline (Type[ag.AgentBase]): The baseline agent class.
            condition_strategy (Type): The condition strategy class.
            env (gym.Env): The environment to wrap the baseline agent.

        Returns:
            ag.AgentBase: The initialized baseline agent.
        """
        return baseline(env=env, condition_strategy=condition_strategy)

    def load_all_baselines(self, env: gym.Env) -> List[ag.AgentBase]:
        """Load all the baselines that were given or, in other words, return them initialized.

        Args:
            env (gym.Env): Environment in which the agents and baselines are evaluated.

        Returns:
            List[ag.AgentBase]: Initialized Baseline Agents
        """
        self.print_baselines()

        all_baselines = []
        for baseline_constr, cs_constr in self.baselines_with_cs:
            baseline = self.load_baseline(
                baseline=baseline_constr, condition_strategy=cs_constr, env=env
            )
            all_baselines.append(baseline)

        return all_baselines

    def print_baselines(self) -> None:
        """Print the found baselines for debugging purposes."""
        print("Found Baselines:")
        for baseline, condition_strategy in self.baselines_with_cs:
            print(
                f"- Baseline: {baseline.__name__}, Condition Strategy: {condition_strategy.__class__.__name__}"
            )


class EnvironmentFactory:
    """Responsible for creating the Environments and wrapping them, as specified in the config.

    Attributes:
        config (EvalConfig): Configuration for the evaluation, including environment
            constructor and evaluation parameters.
        params (EvalParams): Evaluation parameters extracted from the configuration.
    """

    def __init__(self, config: EvalConfig):
        """
        Initialize the EnvironmentFactory.

        Args:
            config (EvalConfig): Configuration on how the evaluation is supposed to be ran.
        """
        self.config: EvalConfig = config
        self.params: EvalParams = config.eval_params

    def create(
        self,
        num_nodes: int,
        num_jammers: Union[int, Tuple[int, int]],
        num_users: Union[int, Tuple[int, int]],
    ) -> envs.Basic_MANETEnv:
        """
        Creates the environment for a given number of nodes, jammers and users. This is not yet wrapped for the agents.

        Args:
            num_nodes (int): Number of nodes in the environment.
            num_jammers (Union[int, Tuple[int, int]]): Number of jammers in the environment.
                If a tuple is provided, the number is sampled uniformly at the start of each episode.
            num_users (Union[int, Tuple[int, int]]): Number of users in the environment.
                If a tuple is provided, the number is sampled uniformly at the start of each episode.

        Returns:
            envs.Basic_MANETEnv: The created MANET environment.
        """
        num_jammers = (
            num_jammers
            if self.params.numbOfJammers is None
            else self.params.numbOfJammers
        )
        num_users = (
            num_users if self.params.numbOfUsers is None else self.params.numbOfUsers
        )
        self._num_nodes = num_nodes
        self._num_jammers_spec = num_jammers
        self._num_users_spec = num_users
        return ut.createEnv(
            env_constructor=self.config.env_constr,
            numbOfNodes=num_nodes,
            numbOfJammers=num_jammers,
            numbOfUsers=num_users,
            render_mode=self.params.render_mode,
            seed=self.params.seed,
            networkConnectedAtStart=self.params.networkConnectedAtStart,
            jammersSpawnNextToUsers=self.params.jammersSpawnNextToUsers,
            attackerModel=self.params.attackerModel,
            allow_early_ep_finish=self.params.allow_early_ep_finish,
            ep_length=self.params.ep_length,
            steps_till_jammer_active=self.params.jammer_ep_steps,
            routing_func=self.params.routing_func,
            step_size=self.params.step_size,
            step_jammers_start_moving=self.params.step_jammers_start_moving,
            numbOfByzantineNodes=self.params.numbOfByzantineNodes or 0,
            byzantine_drop_rate=self.params.byzantine_drop_rate,
            byzantine_attack_type=self.params.byzantine_attack_type,
            byzantine_capacity_inflation_factor=self.params.byzantine_capacity_inflation_factor,
            byzantine_jamming_power_factor=self.params.byzantine_jamming_power_factor,
            byzantine_pos_offset=self.params.byzantine_pos_offset,
            byzantine_obs_user_dir_offset=self.params.byzantine_obs_user_dir_offset,
            byzantine_obs_capacity_factor=self.params.byzantine_obs_capacity_factor,
            byzantine_obs_demand_factor=self.params.byzantine_obs_demand_factor,
            byzantine_obs_interference_factor=self.params.byzantine_obs_interference_factor,
            byzantine_obs_noise_std=self.params.byzantine_obs_noise_std,
            byzantine_obs_capacity_deflate_factor=self.params.byzantine_obs_capacity_deflate_factor,
            byzantine_obs_demand_deflate_factor=self.params.byzantine_obs_demand_deflate_factor,
            byzantine_obs_replay_delay=self.params.byzantine_obs_replay_delay,
            byzantine_obs_magnitude_range=self.params.byzantine_obs_magnitude_range,
        )

    def wrap_for_agent_from_config(
        self, env: gym.Env, agent_config: dict, node_idx: int = None
    ) -> gym.Env:
        """
        Wraps the environment with agent-specific observation and reward wrappers.

        Args:
            env (gym.Env): The environment to wrap.
            agent_config (dict): Configuration for the agent.
            node_idx (int, optional): Index of the node for which the environment is wrapped.
                Defaults to None.

        Returns:
            gym.Env: The wrapped environment.
        """
        return ut.wrapEnvFromConfig(
            env=env,
            config=agent_config,
            rew_wrapper=rw.EvaluationReward,
            obs_type=self.params.obs_type,
            state_loss_rate=self.params.state_loss_rate,
            node_idx=node_idx,
        )

    def wrap_for_baseline(self, env: gym.Env) -> gym.Env:
        """
        Wraps the environment for baseline agents.

        Args:
            env (gym.Env): The environment to wrap.

        Returns:
            gym.Env: The wrapped environment for baseline agents.
        """
        return ut.wrapEnv(
            env=env,
            rew_wrapper=rw.EvaluationReward,
            obs_wrapper=ow.VS_COW,
        )

    def make_eval_vec_env(
        self,
        n_envs: int,
        agent_config: dict,
    ):
        """
        Create a SubprocVecEnv of n_envs parallel eval environments.

        Each subprocess environment is wrapped with EvalInfoWrapper so that
        throughput, ep_step, and n_jammers are available in the info dict on
        every step. Requires create() to have been called first (to populate
        _num_nodes, _num_jammers_spec, _num_users_spec).
        """
        import functools
        from stable_baselines3.common.vec_env import SubprocVecEnv

        p = self.params
        fns = [
            functools.partial(
                _eval_env_factory,
                env_constr=self.config.env_constr,
                agent_config=agent_config,
                obs_type=p.obs_type,
                state_loss_rate=p.state_loss_rate,
                numbOfNodes=self._num_nodes,
                numbOfJammers=self._num_jammers_spec,
                numbOfUsers=self._num_users_spec,
                render_mode=None,
                seed=(p.seed + i) if p.seed is not None else i,
                networkConnectedAtStart=p.networkConnectedAtStart,
                jammersSpawnNextToUsers=p.jammersSpawnNextToUsers,
                attackerModel=p.attackerModel,
                allow_early_ep_finish=p.allow_early_ep_finish,
                ep_length=p.ep_length,
                steps_till_jammer_active=p.jammer_ep_steps,
                routing_func=p.routing_func,
                step_size=p.step_size,
                step_jammers_start_moving=p.step_jammers_start_moving,
                numbOfByzantineNodes=p.numbOfByzantineNodes or 0,
                byzantine_drop_rate=p.byzantine_drop_rate,
                byzantine_attack_type=p.byzantine_attack_type,
                byzantine_capacity_inflation_factor=p.byzantine_capacity_inflation_factor,
                byzantine_jamming_power_factor=p.byzantine_jamming_power_factor,
                byzantine_pos_offset=p.byzantine_pos_offset,
                byzantine_obs_user_dir_offset=p.byzantine_obs_user_dir_offset,
                byzantine_obs_capacity_factor=p.byzantine_obs_capacity_factor,
                byzantine_obs_demand_factor=p.byzantine_obs_demand_factor,
                byzantine_obs_interference_factor=p.byzantine_obs_interference_factor,
                byzantine_obs_noise_std=p.byzantine_obs_noise_std,
                byzantine_obs_capacity_deflate_factor=p.byzantine_obs_capacity_deflate_factor,
                byzantine_obs_demand_deflate_factor=p.byzantine_obs_demand_deflate_factor,
                byzantine_obs_replay_delay=p.byzantine_obs_replay_delay,
                byzantine_obs_magnitude_range=p.byzantine_obs_magnitude_range,
            )
            for i in range(n_envs)
        ]
        return SubprocVecEnv(fns)


class EvaluationCoordinator:
    """
    Orchestrates agent evaluation across groups.

    This class manages the evaluation of agents and baselines in a MANET environment.
    It handles grouped evaluations, filtering of stored models, and logging of results.

    Attributes:
        config (EvalConfig): Configuration for the evaluation, including environment constructor and evaluation parameters.
        env_factory (EnvironmentFactory): Factory for creating and wrapping environments.
        agent_resolver (AgentResolver): Resolves and groups agent paths for evaluation.
        baseline_resolver (BaselineResolver): Resolves baseline configurations and loads them.
        logging_service (LoggingService): Manages logging of metrics and states.
        eval_type (str): Type of evaluation to be performed (e.g., "grouped_eval", "filter_models").
        eval_params (EvalParams): Evaluation parameters extracted from the configuration.
        values_to_log (List[str]): List of metrics to log during evaluation.
    """

    def __init__(
        self,
        eval_type: str,
        config: EvalConfig,
        agent_resolver: AgentResolver,
        baseline_resolver: BaselineResolver,
        logging_service: "LoggingService",
        values_to_log: List[str],
    ) -> None:
        """
        Initialize the EvaluationCoordinator.

        Args:
            eval_type (str): Type of evaluation to be performed.
            config (EvalConfig): Configuration for the evaluation.
            agent_resolver (AgentResolver): Resolves and groups agent paths.
            baseline_resolver (BaselineResolver): Resolves and loads baseline configurations.
            logging_service (LoggingService): Manages logging of metrics and states.
            values_to_log (List[str]): List of metrics to log during evaluation.
        """
        self.config = config
        self.env_factory = EnvironmentFactory(config=config)
        self.agent_resolver = agent_resolver
        self.baseline_resolver = baseline_resolver
        self.logging_service = logging_service
        self.eval_type = eval_type
        self.eval_params = config.eval_params
        self.values_to_log = values_to_log

    def run(self) -> None:
        """
        Run the evaluation based on the specified evaluation type.

        If the evaluation type is "filter_models", it filters stored models.
        Otherwise, it performs a grouped evaluation or filter the best agent between different rl agents, not just stored models of the same from different steps as with "filter_models".
        """
        if self.eval_type == "filter_models":
            self._filter_stored_model_grouped()
        else:
            self._evaluate_grouped()

    def _evaluate_grouped(self) -> None:
        """
        Perform grouped evaluation of agents and baselines (groupd by number of nodes, jammers and users)

        This method creates environments, wraps them for baselines, and evaluates agents and baselines using the specified evaluation type.
        """
        grouped_agents = self.agent_resolver.grouped_agents

        for key, agent_paths in grouped_agents.items():
            unwrapped_env = self.env_factory.create(*key)
            log_folder = (
                self._create_log_folder(*key)
                if self.eval_type != "filter_best_agent"
                else self.config.log_dir
            )

            baseline_env = self.env_factory.wrap_for_baseline(env=unwrapped_env)
            baselines = self.baseline_resolver.load_all_baselines(env=baseline_env)

            self._store_eval_config(store_folder=log_folder, numbOfNodes=key[0])
            self.logging_service.initialize(log_folder=log_folder)

            # Create the AgentEvaluator
            agent_evaluator = AgentEvaluator(
                unwrapped_env=unwrapped_env,
                baselines=baselines,
                baseline_env=baseline_env,
                agent_paths=agent_paths,
                agent_resolver=self.agent_resolver,
                env_factory=self.env_factory,
                eval_params=self.eval_params,
                eval_config=self.config,
                logging_service=self.logging_service,
                values_to_log=self.values_to_log,
            )

            # Use EvaluationHandler to run the selected evaluation type
            EvaluationHandler.run_evaluation(
                eval_type=self.eval_type,
                obs_type=self.eval_params.obs_type,
                agent_evaluator=agent_evaluator,
            )

            self.logging_service.close()

    def _filter_stored_model_grouped(self) -> None:
        """
        Filter stored models based on evaluation parameters.

        This method identifies agent models stored at specific intervals, evaluates
        them, and logs the results.
        """
        grouped_agents = self.agent_resolver.grouped_agents

        for key, agent_paths in grouped_agents.items():
            unwrapped_env = self.env_factory.create(*key)

            agent_folders = self.agent_resolver.make_agent_folder_only(
                agent_paths=agent_paths
            )

            for folder in agent_folders:
                folder_path = Path(folder)

                log_folder = str(
                    folder_path.parents[0].with_name(
                        folder_path.parents[0].name + "_filtered"
                    )
                )
                self.config.log_dir = log_folder

                model_pattern = str(folder_path / "*")
                model_paths = ut.findAllAgentFilesWithInterval(
                    model_pattern, filterval=self.eval_params.filterval
                )

                baseline_env = self.env_factory.wrap_for_baseline(env=unwrapped_env)
                baselines = self.baseline_resolver.load_all_baselines(env=baseline_env)

                self._store_eval_config(store_folder=log_folder, numbOfNodes=key[0])
                self.logging_service.initialize(log_folder=log_folder)

                # Create the AgentEvaluator
                agent_evaluator = AgentEvaluator(
                    unwrapped_env=unwrapped_env,
                    baselines=baselines,
                    baseline_env=baseline_env,
                    agent_paths=model_paths,
                    agent_resolver=self.agent_resolver,
                    env_factory=self.env_factory,
                    eval_params=self.eval_params,
                    eval_config=self.config,
                    logging_service=self.logging_service,
                    values_to_log=self.values_to_log,
                )

                # Use EvaluationHandler to run the selected evaluation type
                EvaluationHandler.run_evaluation(
                    eval_type="filter_best_agent",
                    obs_type=self.eval_params.obs_type,
                    agent_evaluator=agent_evaluator,
                )

                self.logging_service.close()

    def _store_eval_config(self, store_folder: str, numbOfNodes: int) -> None:
        """
        Store the evaluation configuration in the specified folder.

        Args:
            store_folder (str): Path to the folder where the configuration will be stored.
            numbOfNodes (int): Number of nodes in the environment.
        """
        if self.logging_service.enabled:
            eval_dict = self.config.to_dict()
            eval_dict["eval_params"]["numbOfNodes"] = numbOfNodes
            ut.document_config(
                path=store_folder,
                file_name="20_evaluation_config",
                **eval_dict,
                evaluated_agents=self.agent_resolver.agent_paths,
            )

    def _create_log_folder(
        self,
        numb_of_nodes: int,
        numb_of_jammers: int,
        numb_of_users: int,
    ) -> str:
        """
        Create a log folder path based on evaluation parameters and configuration.

        Args:
            numb_of_nodes (int): Number of nodes in the environment.
            numb_of_jammers (int): Number of jammers in the environment.
            numb_of_users (int): Number of users in the environment.

        Returns:
            str: Path to the created log folder.
        """
        # Process attacker and environment names
        attacker_model = self.eval_params.attackerModel
        attacker_name = ut.getAttackerModelNameFromInstance(
            self.eval_params.attackerModel
        )
        if isinstance(attacker_model, list):
            for i, name in enumerate(sorted(attacker_name)):
                attacker_name[i] = process_string(name, max_length=2)
            attacker_name = "-".join(attacker_name)
        else:
            attacker_name = process_string(
                attacker_name,
                max_length=2,
            )
        env_name = process_string(self.config.env_constr.__name__, max_length=2)

        # Construct the base log path
        log_path = Path(self.config.log_dir) / attacker_name / env_name

        # Determine evaluation type string
        eval_type_str = (
            f"{self.eval_params.obs_type}_{self.eval_params.state_loss_rate}"
            if self.eval_params.obs_type in ["random", "real"]
            else self.eval_params.obs_type
        )

        # Construct the log folder name
        log_name = f"{numb_of_nodes}_{numb_of_jammers}_{numb_of_users}_{eval_type_str}"

        # Combine paths to create the full log folder path
        log_folder = log_path / log_name

        return log_folder


# * Value Funcs
class ValueFunction(ABC):
    """
    Abstract base class for value functions.

    Value functions are used to calculate specific metrics or values
    based on the environment's state or actions. All subclasses must
    implement the `calculate` method.

    Methods:
        calculate(unwrapped_env, **kwargs): Abstract method to calculate the value.
    """

    @abstractmethod
    def calculate(self, unwrapped_env: envs.Basic_MANETEnv, **kwargs) -> float:
        """
        Calculate the value based on the environment's state or actions.

        Args:
            unwrapped_env (envs.Basic_MANETEnv): The unwrapped MANET environment.

        Returns:
            float: The calculated value.
        """
        pass


class DistanceTravelled(ValueFunction):
    """
    Calculates the total distance travelled by nodes in the environment.

    This value function computes the sum of the Euclidean distances
    travelled by all nodes based on their actions.
    """

    def calculate(
        self, unwrapped_env: envs.Basic_MANETEnv, action: np.ndarray, **kwargs
    ) -> float:
        """
        Calculate the total distance travelled by nodes.

        Args:
            unwrapped_env (envs.Basic_MANETEnv): The unwrapped MANET environment.
            action (np.ndarray): The action array representing node movements.

        Returns:
            float: The total distance travelled by all nodes.
        """
        return np.sum(
            np.linalg.norm(
                action * unwrapped_env.node_step_size,
                axis=1,
            )
        )


class ThroughputInMbits(ValueFunction):
    """
    Calculates the network throughput in megabits per second.

    This value function computes the total throughput of the network
    by accessing the environment's network component.
    """

    def calculate(self, unwrapped_env: envs.Basic_MANETEnv, **kwargs) -> float:
        """
        Calculate the network throughput in megabits per second.

        Args:
            unwrapped_env (envs.Basic_MANETEnv): The unwrapped MANET environment.

        Returns:
            float: The network throughput in megabits per second.
        """
        return unwrapped_env.network.getThroughput() / 1e6


class EvalInfoWrapper(gym.Wrapper):
    """Injects throughput, ep_step, and n_jammers into the step/reset info dict.

    Used by one_evaluation_run_vec so SubprocVecEnv workers expose these metrics
    to the main-process collection loop via the standard info dict.  Also calls
    initializeComps() on reset so SubprocVecEnv auto-resets are equivalent to
    the explicit initializeComps() call in the sequential eval path.
    """

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        u = self.env.unwrapped
        info["throughput"] = u.network.getThroughput() / 1e6
        info["ep_step"] = u.ep_step
        info["n_jammers"] = u.numbOfJammers
        return obs, reward, done, trunc, info

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        try:
            self.env.get_wrapper_attr("initializeComps")()
        except AttributeError:
            pass
        obs = self.env.get_wrapper_attr("observation")()
        u = self.env.unwrapped
        return obs, {"ep_step": u.ep_step, "n_jammers": u.numbOfJammers}


def _eval_env_factory(env_constr, agent_config: dict, obs_type: str, state_loss_rate, **env_kwargs) -> gym.Env:
    """Top-level picklable factory for evaluation SubprocVecEnv workers.

    Accepts all ut.createEnv keyword args forwarded from make_eval_vec_env
    via functools.partial, wraps the result for agent inference, and adds
    EvalInfoWrapper on top so per-step metrics are available in the info dict.
    """
    # obs_type in {"real", "gossip1", "reliable"} require a per-node node_idx for
    # decentralised execution.  SubprocVecEnv workers run a full multi-agent
    # simulation with no single node_idx, so fall back to "full" which works with
    # batched shared-policy prediction.
    effective_obs_type = "full" if obs_type in ("real", "gossip1", "reliable") else obs_type
    env = ut.createEnv(env_constructor=env_constr, **env_kwargs)
    wrapped = ut.wrapEnvFromConfig(
        env=env,
        config=agent_config,
        rew_wrapper=rw.EvaluationReward,
        obs_type=effective_obs_type,
        state_loss_rate=state_loss_rate,
        node_idx=None,
    )
    return EvalInfoWrapper(wrapped)


# * Evluation Classes
class AgentEvaluator:
    """
    Handles evaluation of agents and baselines in single or multi-agent scenarios.

    This class evaluates reinforcement learning (RL) agents and baseline agents in a MANET environment.
    It supports both single-agent and multi-agent setups in decentralized execution, logging metrics, and identifying the best-performing agents.

    Attributes:
        VALUE_FUNCTIONS (Dict[str, ValueFunction]): Dictionary mapping metric names to their corresponding value functions.
        unwrapped_env (envs.Basic_MANETEnv): The unwrapped MANET environment used for evaluation.
        baselines (List[ag.AgentBase]): List of baseline agents to evaluate.
        baseline_env (gym.Env): Environment wrapped for baseline evaluation.
        agent_paths (List[str]): List of paths to RL agent models.
        agent_resolver (AgentResolver): Resolves and loads RL agents.
        env_factory (EnvironmentFactory): Factory for creating and wrapping environments.
        eval_params (EvalParams): Evaluation parameters for the MANET environment.
        eval_config (EvalConfig): Configuration for the evaluation process.
        logging_service (LoggingService): Service for logging metrics and states.
        values_to_log (List[str]): List of metrics to log during evaluation.
    """

    VALUE_FUNCTIONS: Dict[str, ValueFunction] = {
        "distance_travelled": DistanceTravelled(),
        "throughput": ThroughputInMbits(),
    }

    def __init__(
        self,
        unwrapped_env: envs.Basic_MANETEnv,
        baselines: List[ag.AgentBase],
        baseline_env: gym.Env,
        agent_paths: List[str],
        agent_resolver: AgentResolver,
        env_factory: EnvironmentFactory,
        eval_params: EvalParams,
        eval_config: EvalConfig,
        logging_service: "LoggingService",
        values_to_log: List[str],
    ):
        """
        Initialize the AgentEvaluator.

        Args:
            unwrapped_env (envs.Basic_MANETEnv): The unwrapped MANET environment.
            baselines (List[ag.AgentBase]): List of baseline agents to evaluate.
            baseline_env (gym.Env): Environment wrapped for baseline evaluation.
            agent_paths (List[str]): List of paths to RL agent models.
            agent_resolver (AgentResolver): Resolves and loads RL agents.
            env_factory (EnvironmentFactory): Factory for creating and wrapping environments.
            eval_params (EvalParams): Evaluation parameters for the MANET environment.
            eval_config (EvalConfig): Configuration for the evaluation process.
            logging_service (LoggingService): Service for logging metrics and states.
            values_to_log (List[str]): List of metrics to log during evaluation.
        """
        self.unwrapped_env = unwrapped_env
        self.baselines = baselines
        self.baseline_env = baseline_env
        self.agent_paths = agent_paths
        self.agent_resolver = agent_resolver
        self.env_factory = env_factory
        self.eval_params = eval_params
        self.eval_config = eval_config
        self.logging_service = logging_service
        self.values_to_log = values_to_log

    def evaluate_list_of_rl_agents_against_baseline(
        self,
        is_single_agent: bool = True,
        eval_agents: bool = True,
        eval_baselines: bool = True,
        copy_best_agent: bool = False,
    ) -> None:
        """
        Evaluate RL agents and baselines, and optionally save the best-performing agent.

        Args:
            is_single_agent (bool): Whether to evaluate in a single-agent setup.
            eval_agents (bool): Whether to evaluate RL agents.
            eval_baselines (bool): Whether to evaluate baseline agents.
            copy_best_agent (bool): Whether to copy the best-performing RL agent to a seperate folder, the log_dir specified.
        """
        best_agent_path, best_value = "", 0

        if eval_baselines:
            self.evaluate_baselines()

        if eval_agents:
            best_agent_path, best_value = self.evaluate_rl_agents(
                is_single_agent=is_single_agent,
            )

            if copy_best_agent:
                self._save_best_agent(
                    best_agent_path=best_agent_path, log_dir=self.eval_config.log_dir
                )

    def evaluate_rl_agents(
        self,
        is_single_agent: bool = True,
    ) -> Tuple[str, float]:
        """
        Evaluate RL agents and return the best-performing agent.

        Args:
            is_single_agent (bool): Whether to evaluate in a single-agent setup.

        Returns:
            Tuple[str, float]: Path to the best-performing agent and its evaluation metric.
        """
        path_mean_pairs: List[Tuple[str, float]] = []
        for ag_path in self.agent_paths:
            agent_config = self.agent_resolver.load_agent_config(agent_path=ag_path)
            agent_name = self.agent_resolver.give_agent_name(
                agent_path=ag_path, base_obs_type=self.eval_params.obs_type
            )

            if self.eval_params.n_envs > 1:
                # Vectorised path: load a single agent for batched prediction.
                agent_env, agent = self.setup_single_agent(
                    ag_path=ag_path, agent_config=agent_config
                )
                quick_stats = self.one_evaluation_run_vec(
                    agent_name=agent_name,
                    agent=agent,
                    agent_config=agent_config,
                )
            else:
                agent_envs, agents = self.setup_agents(
                    ag_path=ag_path,
                    agent_config=agent_config,
                    is_single_agent=is_single_agent,
                )
                quick_stats = self.one_evaluation_run(
                    agent_name=agent_name,
                    agents=agents,
                    agent_envs=agent_envs,
                    is_single_agent=is_single_agent,
                )

            self.logging_service.set_state_logger_enabled(enabled=False)

            value = self._quick_stat_metric(stats=quick_stats)

            path_mean_pairs.append((ag_path, value))

        best_agent_path, best_value = max(path_mean_pairs, key=lambda x: x[1])

        return best_agent_path, best_value

    def evaluate_baselines(self) -> None:
        """
        Evaluate baseline agents and log their performance.
        """
        for bsline in self.baselines:
            bsline_name = process_string(bsline.getName(), max_length=5)
            self.one_evaluation_run(
                agent_name=bsline_name,
                agents=bsline,
                agent_envs=self.baseline_env,
                is_single_agent=True,
            )

    def setup_agents(
        self, ag_path: str, agent_config: dict, is_single_agent: bool = True
    ) -> Union[
        Tuple[List[gym.Env], List[BaseAlgorithm]], Tuple[gym.Env, BaseAlgorithm]
    ]:
        """
        Set up agents and their environments.

        Args:
            ag_path (str): Path to the agent model.
            agent_config (dict): Configuration for the agent.
            is_single_agent (bool): Whether to set up a single-agent environment.

        Returns:
            Union[Tuple[List[gym.Env], List[BaseAlgorithm]], Tuple[gym.Env, BaseAlgorithm]]:
            Environments and agents for evaluation.
        """
        if is_single_agent:
            return self.setup_single_agent(ag_path=ag_path, agent_config=agent_config)
        else:
            return self.setup_multiple_agents(
                ag_path=ag_path, agent_config=agent_config
            )

    def setup_single_agent(
        self, ag_path: str, agent_config: dict
    ) -> Tuple[gym.Env, BaseAlgorithm]:
        """
        Set up a single agent and its environment.

        Args:
            ag_path (str): Path to the agent model.
            agent_config (dict): Configuration for the agent.

        Returns:
            Tuple[gym.Env, BaseAlgorithm]: Environment and agent for evaluation.
        """
        agent_env = self.env_factory.wrap_for_agent_from_config(
            env=self.unwrapped_env, agent_config=agent_config
        )
        agent = self.agent_resolver.load_agent(path=ag_path, env=agent_env)

        return agent_env, agent

    def setup_multiple_agents(
        self, ag_path: str, agent_config: dict
    ) -> Tuple[List[gym.Env], List[BaseAlgorithm]]:
        """
        Set up multiple agents and their environments.

        Args:
            ag_path (str): Path to the agent model.
            agent_config (dict): Configuration for the agents.

        Returns:
            Tuple[List[gym.Env], List[BaseAlgorithm]]: Environments and agents for evaluation.
        """
        agents = []
        agent_envs = []

        for node_idx in range(self.unwrapped_env.numbOfNodes):
            agent_env = self.env_factory.wrap_for_agent_from_config(
                env=self.unwrapped_env, agent_config=agent_config, node_idx=node_idx
            )
            agent = self.agent_resolver.load_agent(path=ag_path, env=agent_env)
            agents.append(agent)
            agent_envs.append(agent_env)

        return agent_envs, agents

    def one_evaluation_run(
        self,
        agent_name: str,
        agents: Union[BaseAlgorithm, List[BaseAlgorithm]],
        agent_envs: Union[gym.Env, List[gym.Env]],
        is_single_agent: bool = False,
    ) -> dict:
        """
        Perform a single evaluation run for an agent or baseline.

        Args:
            agent_name (str): Name of the agent or baseline being evaluated.
            agents (Union[BaseAlgorithm, List[BaseAlgorithm]]): The agent(s) to evaluate.
            agent_envs (Union[gym.Env, List[gym.Env]]): The environment(s) for evaluation.
            is_single_agent (bool): Whether the evaluation is for a single agent.

        Returns:
            dict: A dictionary containing evaluation statistics for the run.
        """
        # Initialize statistics tracker
        stats = EpisodeStats(jammer_ep_step=self.eval_params.jammer_ep_steps)

        self.unwrapped_env.reset(seed=self.eval_params.seed)

        for episode in range(self.eval_params.n_eval_eps):
            done, trunc = False, False

            self._ensure_obs_wrappers_are_init(
                agent_envs=agent_envs, is_single_agent=is_single_agent
            )
            self._ensure_baselines_are_init(
                agents=agents, is_single_agent=is_single_agent
            )

            if self.eval_params.jammers_preposition:
                self._preposition_jammers(
                    episode=episode,
                    stats=stats,
                    agent_name=agent_name,
                    agents=agents,
                    agent_envs=agent_envs,
                    is_single_agent=is_single_agent,
                )

            # Delay for nicer graph, doesn't impact evaluation
            if is_single_agent:
                if isinstance(agents, ag.AgentBase):
                    for i in range(episode % 9):
                        done, trunc = self._execute_single_step(
                            agents=agents,
                            agent_envs=agent_envs,
                            is_single_agent=is_single_agent,
                            stats=stats,
                            agent_name=agent_name,
                            episode=episode,
                            use_zero_action=True,
                        )

            while not (done or trunc):
                done, trunc = self._execute_single_step(
                    agents=agents,
                    agent_envs=agent_envs,
                    is_single_agent=is_single_agent,
                    stats=stats,
                    agent_name=agent_name,
                    episode=episode,
                    use_zero_action=False,
                )

        quick_stats = stats.get_overall_stats()
        self._print_throughput(
            agent_name=agent_name,
            overall_mean_throughput=quick_stats["overall"],
            before_jammer_mean_throughput=quick_stats["before_jammers"],
            at_point_of_jamming=quick_stats.get("at_activation", 0.0),
            after_jammer_mean_throughput=quick_stats["after_jammers"],
        )
        return quick_stats

    def one_evaluation_run_vec(
        self,
        agent_name: str,
        agent,
        agent_config: dict,
    ) -> dict:
        """
        Vectorised evaluation run using n_envs parallel SubprocVecEnv workers.

        Runs self.eval_params.n_eval_eps episodes total across n_envs parallel
        environments, collecting per-step throughput for compare_byzantine.py
        and per-episode aggregate stats for the overall metrics table.

        Limitations vs the sequential path:
        - jammers_preposition is not supported (raises ValueError if True).
        - Per-step state logging and distance_travelled metric are skipped.
        - Only the single-agent (is_single_agent=True) setup is supported.

        Args:
            agent_name (str): Display name of the agent being evaluated.
            agent: Loaded SB3 model; predict() is called on batched obs.
            agent_config (dict): Agent training config (for env wrapping).

        Returns:
            dict: Overall stats dict identical to one_evaluation_run output.
        """
        if self.eval_params.jammers_preposition:
            raise ValueError("n_envs > 1 does not support jammers_preposition=True.")

        n_envs = self.eval_params.n_envs
        stats = EpisodeStats(jammer_ep_step=self.eval_params.jammer_ep_steps)
        stats._init_vec_state(n_envs)

        self.logging_service.set_state_logger_enabled(enabled=False)

        vec_env = self.env_factory.make_eval_vec_env(n_envs=n_envs, agent_config=agent_config)

        n_target = self.eval_params.n_eval_eps
        completed = 0
        # Per-env buffers of (ep_step, throughput) — flushed to LoggingService on episode end
        # so each row gets a globally unique episode number.
        ep_buffers: List[List] = [[] for _ in range(n_envs)]

        try:
            obs = vec_env.reset()
            while completed < n_target:
                actions, _ = agent.predict(obs, deterministic=self.eval_params.deterministic)
                step_result = vec_env.step(actions)
                obs, dones, infos = step_result[0], step_result[2], step_result[3]

                for i in range(n_envs):
                    if completed >= n_target:
                        break
                    info = infos[i]
                    throughput = info["throughput"]
                    ep_step = info["ep_step"]
                    n_jammers = info["n_jammers"]

                    stats.update_for_env(i, throughput, ep_step, n_jammers)
                    ep_buffers[i].append((ep_step, throughput))

                    if dones[i]:
                        ep_idx = completed
                        for log_step, log_tput in ep_buffers[i]:
                            self.logging_service.log_metric(
                                agent_name=agent_name,
                                ep_step=log_step,
                                episode=ep_idx,
                                metric_name="throughput",
                                value=log_tput,
                            )
                        ep_buffers[i] = []
                        stats.end_episode_for_env(i)
                        completed += 1
        finally:
            vec_env.close()

        quick_stats = stats.get_overall_stats()
        self._print_throughput(
            agent_name=agent_name,
            overall_mean_throughput=quick_stats["overall"],
            before_jammer_mean_throughput=quick_stats["before_jammers"],
            at_point_of_jamming=quick_stats.get("at_activation", 0.0),
            after_jammer_mean_throughput=quick_stats["after_jammers"],
        )
        return quick_stats

    def _ensure_baselines_are_init(
        self,
        agents: Union[BaseAlgorithm, List[BaseAlgorithm]],
        is_single_agent: bool = False,
    ) -> None:
        """
        Ensure that baseline agents are properly initialized.

        Args:
            agents (Union[BaseAlgorithm, List[BaseAlgorithm]]): The baseline agent(s).
            is_single_agent (bool): Whether the evaluation is for a single agent.
        """
        if is_single_agent:
            agent = agents
            if isinstance(agent, ag.AgentBase):
                agent: ag.AgentBase
                agent.reset_movement_operators()

    def _execute_single_step(
        self,
        agents: ag.AgentBase,
        agent_envs: gym.Env,
        is_single_agent: bool,
        stats: "EpisodeStats",
        agent_name: str,
        episode: int,
        use_zero_action: bool = False,
    ) -> Tuple[bool, bool]:
        """
        Execute a single step in the environment and handle updates.

        Args:
            agents (ag.AgentBase): The agent(s) performing the step.
            agent_envs (gym.Env): The environment(s) for the step.
            is_single_agent (bool): Whether the evaluation is for a single agent.
            stats ("EpisodeStats"): The statistics tracker for the evaluation.
            agent_name (str): Name of the agent being evaluated.
            episode (int): The current episode number.
            use_zero_action (bool): Whether to use a zero action (no movement).

        Returns:
            Tuple[bool, bool]: A tuple indicating whether the episode is done or truncated.
        """
        self._check_log_state(episode=episode)

        if use_zero_action:
            # Use zero action (no movement)
            action = np.zeros(shape=(self.unwrapped_env.numbOfNodes, 2))
            _, reward, done, trunc, _ = self.unwrapped_env.step(action)
            self._ensure_agents_update(
                agent_envs=agent_envs, is_single_agent=is_single_agent
            )
        else:
            # Get action from agent
            action, reward, done, trunc = self._step_agent(
                agents=agents,
                agent_envs=agent_envs,
                is_single_agent=is_single_agent,
            )

        # Calculate throughput and update statistics
        throughput = ThroughputInMbits().calculate(unwrapped_env=self.unwrapped_env)
        stats.update(
            throughput,
            ep_step=self.unwrapped_env.ep_step,
            numbOfJammers=self.unwrapped_env.numbOfJammers,
        )

        # Log metrics
        self._log_metrics(action=action, agent_name=agent_name, episode=episode)

        # Handle episode termination
        if done or trunc:
            stats.end_episode()
            self.unwrapped_env.reset()

        return done, trunc

    def _step_agent(
        self, agents: ag.AgentBase, agent_envs: gym.Env, is_single_agent: bool
    ) -> Tuple[np.ndarray, float, bool, bool]:
        """
        Perform a step in the environment for the given agent(s).

        This method determines whether to execute a single-agent or multi-agent step
        based on the `is_single_agent` flag.

        Args:
            agents (ag.AgentBase): The agent(s) performing the step.
            agent_envs (gym.Env): The environment(s) for the step.
            is_single_agent (bool): Whether the evaluation is for a single agent.

        Returns:
            Tuple[np.ndarray, float, bool, bool]: A tuple containing:
                - The action(s) taken by the agent(s).
                - The reward(s) received.
                - A boolean indicating whether the episode is done.
                - A boolean indicating whether the episode is truncated.
        """
        if is_single_agent:
            action, reward, done, trunc = self._step_single_agent(
                agent=agents, agent_env=agent_envs
            )
        else:
            action, reward, done, trunc = self._step_multi_agent(
                agents=agents, agent_envs=agent_envs
            )

        return action, reward, done, trunc

    def _step_single_agent(
        self, agent: ag.AgentBase, agent_env: gym.Env
    ) -> Tuple[np.ndarray, float, bool, bool]:
        """
        Perform a single step for a single agent in the environment.

        Args:
            agent (ag.AgentBase): The agent performing the step.
            agent_env (gym.Env): The environment for the agent.

        Returns:
            Tuple[np.ndarray, float, bool, bool]: A tuple containing:
                - The action taken by the agent.
                - The reward received.
                - A boolean indicating whether the episode is done.
                - A boolean indicating whether the episode is truncated.
        """
        obs = agent_env.get_wrapper_attr("observation")()
        action, _ = agent.predict(obs, deterministic=self.eval_params.deterministic)
        _, reward, done, trunc, _ = self.unwrapped_env.step(action)
        return action, reward, done, trunc

    def _step_multi_agent(
        self,
        agents: List[BaseAlgorithm],
        agent_envs: List[gym.Env],
    ) -> Tuple[np.ndarray, float, bool, bool]:
        """
        Perform a single step for multiple agents in the environment.

        Args:
            agents (List[BaseAlgorithm]): The list of agents performing the step.
            agent_envs (List[gym.Env]): The list of environments for the agents.

        Returns:
            Tuple[np.ndarray, float, bool, bool]: A tuple containing:
                - The actions taken by the agents.
                - The reward received.
                - A boolean indicating whether the episode is done.
                - A boolean indicating whether the episode is truncated.
        """
        observations = [env.get_wrapper_attr("observation")() for env in agent_envs]
        action = np.zeros((self.unwrapped_env.numbOfNodes, 2))

        for i in range(self.unwrapped_env.numbOfNodes):
            agent: BaseAlgorithm = agents[i]
            act, _ = agent.predict(
                observations[i], deterministic=self.eval_params.deterministic
            )
            action[i][:] = act[i][:]

        _, reward, done, trunc, _ = self.unwrapped_env.step(action)

        return action, reward, done, trunc

    def _ensure_obs_wrappers_are_init(
        self,
        agent_envs: Union[gym.Env, List[gym.Env]],
        is_single_agent: bool = False,
    ) -> None:
        """
        Ensure that observation wrappers are initialized for the given environment(s).

        This method initializes the observation wrappers for either a single-agent
        or multi-agent setup.

        Args:
            agent_envs (Union[gym.Env, List[gym.Env]]): The environment(s) for the agent(s).
            is_single_agent (bool): Whether the evaluation is for a single agent.
        """
        if is_single_agent:
            agent_env = agent_envs
            agent_env.get_wrapper_attr("initializeComps")()
        else:
            for agent_env in agent_envs:
                agent_env: gym.ObservationWrapper
                agent_env.get_wrapper_attr("initializeComps")()

    def _ensure_agents_update(
        self,
        agent_envs: Union[gym.Env, List[gym.Env]],
        is_single_agent: bool = False,
    ) -> None:
        """
        Ensure that agents observations are updated in the given environment(s).

        This method updates the agents' observations from other components for either a single-agent
        or multi-agent setup.

        Args:
            agent_envs (Union[gym.Env, List[gym.Env]]): The environment(s) for the agent(s).
            is_single_agent (bool): Whether the evaluation is for a single agent.
        """
        if is_single_agent:
            agent_env = agent_envs
            agent_env.get_wrapper_attr("updateComps")()
        else:
            for agent_env in agent_envs:
                agent_env: gym.ObservationWrapper
                agent_env.get_wrapper_attr("updateComps")()

    def _preposition_jammers(
        self,
        episode: int,
        stats: "EpisodeStats",
        agent_name: str,
        agents: Union[BaseAlgorithm, List[BaseAlgorithm]],
        agent_envs: Union[gym.Env, List[gym.Env]],
        is_single_agent: bool = False,
    ) -> None:
        """
        Preposition jammers in the environment before evaluation starts.

        This method ensures that jammers are prepositioned by executing steps
        with zero actions until the jammer activation step is reached.

        Args:
            episode (int): The current episode number.
            stats ("EpisodeStats"): The statistics tracker for the evaluation.
            agent_name (str): Name of the agent being evaluated.
            agents (Union[BaseAlgorithm, List[BaseAlgorithm]]): The agent(s) to evaluate.
            agent_envs (Union[gym.Env, List[gym.Env]]): The environment(s) for evaluation.
            is_single_agent (bool): Whether the evaluation is for a single agent.
        """
        while self.unwrapped_env.ep_step < self.eval_params.jammer_ep_steps:
            done, trunc = self._execute_single_step(
                agents=agents,
                agent_envs=agent_envs,
                is_single_agent=is_single_agent,
                stats=stats,
                agent_name=agent_name,
                episode=episode,
                use_zero_action=True,
            )

    def _quick_stat_metric(self, stats: dict) -> float:
        """
        Calculate a quick evaluation metric based on throughput statistics.

        This method computes a weighted metric using throughput values before
        and after jammer activation.

        Args:
            stats (dict): A dictionary containing throughput statistics.

        Returns:
            float: The calculated evaluation metric.
        """
        alpha = self.eval_params.metric_param

        before_jammers = stats["before_jammers"]
        after_jammers = stats["after_jammers"]

        metric = (1 - alpha) * before_jammers + alpha * after_jammers

        return metric

    # Helper methods
    def _log_metrics(
        self,
        action: np.ndarray,
        agent_name: str,
        episode: int,
    ) -> None:
        """
        Log evaluation metrics for a specific step.

        This method calculates and logs metrics such as throughput or distance traveled
        for the given agent and action.

        Args:
            action (np.ndarray): The action taken by the agent(s) during the step.
            agent_name (str): The name of the agent being evaluated.
            episode (int): The current episode number.
        """
        for value_name in self.values_to_log:
            value_func = self.VALUE_FUNCTIONS.get(value_name)
            if value_func:
                value = value_func.calculate(
                    unwrapped_env=self.unwrapped_env, action=action
                )
                self.logging_service.log_metric(
                    agent_name=agent_name,
                    ep_step=self.unwrapped_env.ep_step,
                    episode=episode,
                    metric_name=value_name,
                    value=value,
                )

        # Log obs_user_dir detection error if any Byzantine nodes are active.
        # Set by ObservationWrappers.get_selected_node_observations each step.
        detection_errors = getattr(self.unwrapped_env, "_detection_errors", {})
        if detection_errors:
            mean_err = float(np.mean(list(detection_errors.values())))
            self.logging_service.log_metric(
                agent_name=agent_name,
                ep_step=self.unwrapped_env.ep_step,
                episode=episode,
                metric_name="detection_error",
                value=mean_err,
            )

    def _check_log_state(self, episode: int) -> None:
        """
        Log the environment state if specific conditions are met.

        This method logs the state of the environment at the start of an episode
        or just before jammer activation, depending on the evaluation parameters.

        Args:
            episode (int): The current episode number.
        """
        ep_step = self.unwrapped_env.ep_step
        jammer_ep_steps = self.eval_params.jammer_ep_steps
        jammers_preposition = self.eval_params.jammers_preposition

        if (ep_step == 0 and not jammers_preposition) or (
            ep_step == jammer_ep_steps - 1 and jammers_preposition
        ):
            self.logging_service.log_state(
                episode,
                self.unwrapped_env.nodes,
                self.unwrapped_env.jammers,
                self.unwrapped_env.users,
            )

    def _print_throughput(
        self,
        agent_name: str,
        overall_mean_throughput: int,
        before_jammer_mean_throughput: int,
        at_point_of_jamming: int,
        after_jammer_mean_throughput: int,
    ) -> None:
        """
        Display throughput statistics in a formatted table.

        This method uses the `tabulate` library to print throughput statistics
        for different phases of the evaluation.

        Args:
            agent_name (str): The name of the agent being evaluated.
            overall_mean_throughput (int): The overall mean throughput.
            before_jammer_mean_throughput (int): The mean throughput before jammer activation.
            at_point_of_jamming (int): The throughput at the point of jammer activation.
            after_jammer_mean_throughput (int): The mean throughput after jammer activation.
        """
        # Prepare table data
        headers = ["Phase", "Throughput (Mbit/s)"]
        table_data = [
            ["Overall", f"{overall_mean_throughput:.2f}"],
            ["Before jammers", f"{before_jammer_mean_throughput:.2f}"],
            ["At activation", f"{at_point_of_jamming:.2f}"],
            ["After jammers", f"{after_jammer_mean_throughput:.2f}"],
        ]

        # Print title and table
        print(f"\nThroughput Results for: {agent_name}")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print()  # Add empty line after table

    def _save_best_agent(self, best_agent_path: str, log_dir: str) -> None:
        """
        Save the best-performing agent and its configuration.

        This method copies the best agent's model and its configuration file
        to the specified log directory.

        Args:
            best_agent_path (str): Path to the best agent's model file.
            log_dir (str): Directory where the agent and its configuration will be saved.
        """
        act_path = Path(best_agent_path)
        part1 = act_path.parents[1].name  # Example: '27_04'
        parent_name = act_path.parent.name
        match = re.search(r"(PPO_.*)", parent_name)
        part2 = match.group(1) if match else parent_name
        result = f"{part1}-{part2}"

        store_folder = Path(log_dir) / result

        ut.create_dirs(store_folder)
        shutil.copy(best_agent_path, store_folder)

        # Load config
        folder_path = os.path.dirname(best_agent_path)
        file_name = (
            "01_training_config.yaml"
            if ut.hasSpecificConfigFile(folder_path, "01_training_config.yaml")
            else ut.getMostRecentConfigFile(folder_path)
        )
        # Copy config
        shutil.copy(os.path.join(folder_path, file_name), store_folder)


class EvaluationHandler:
    """
    Handles different types of evaluations for agents and baselines.

    Provides static methods to perform various evaluation types,
    such as grouped evaluation, real grouped evaluation, and filtering the best agent.
    """

    @staticmethod
    def grouped_eval(agent_evaluator: AgentEvaluator, is_single_agent: bool):
        """
        Perform grouped evaluation of agents and baselines.

        Evaluates agents and baselines grouped by the number of nodes,
        jammers, and users in the environment.

        Args:
            agent_evaluator (AgentEvaluator): The evaluator responsible for running the evaluation.
        """
        print("Running grouped evaluation...")
        agent_evaluator.evaluate_list_of_rl_agents_against_baseline(
            is_single_agent=is_single_agent,
            eval_agents=True,
            eval_baselines=True,
            copy_best_agent=False,
        )

    @staticmethod
    def best_agent_eval(agent_evaluator: AgentEvaluator, is_single_agent: bool):
        """
        Evaluate and filter the best agent among different RL agents.

        Identifies the best-performing agent across different RL agents
        and copies the best agent's model for further use.

        Args:
            agent_evaluator (AgentEvaluator): The evaluator responsible for running the evaluation.
        """
        print("Running best agent evaluation...")
        agent_evaluator.evaluate_list_of_rl_agents_against_baseline(
            is_single_agent=is_single_agent,
            eval_agents=True,
            eval_baselines=False,
            copy_best_agent=True,
        )


    @staticmethod
    def run_evaluation(eval_type: str, obs_type: str, agent_evaluator: AgentEvaluator):
        """
        Run the appropriate evaluation based on the specified type.

        This method selects the evaluation method to execute based on the evaluation type and observation type.

        Args:
            eval_type (str): The type of evaluation to perform. Supported types are:
                - "grouped_eval": Grouped evaluation of agents and baselines.
                - "filter_best_agent": Filter the best agent among different RL agents.
            obs_type (str): The type of observation used by the agents (e.g., "real").
            agent_evaluator (AgentEvaluator): The evaluator responsible for running the evaluation.

        Raises:
            ValueError: If an unknown evaluation type is provided.
        """
        eval_methods = {
            "grouped_eval": EvaluationHandler.grouped_eval,
            "filter_best_agent": EvaluationHandler.best_agent_eval,
        }

        if eval_type not in eval_methods:
            raise ValueError(f"Unknown evaluation type: {eval_type}")

        eval_method: Callable = eval_methods[eval_type]

        # Important to wrap rl agents adequately
        is_single_agent = obs_type not in ("real", "gossip1", "reliable")

        # Call the selected evaluation method
        eval_method(agent_evaluator=agent_evaluator, is_single_agent=is_single_agent)


class WelfordStats:
    """
    Implements Welford's online algorithm for computing running statistics (mean, variance, std).
    Can track multiple statistics simultaneously.
    """

    def __init__(self):
        # Dictionary to store counts for different stats
        self.counts = defaultdict(int)
        # Dictionary to store means for different stats
        self.means = defaultdict(float)
        # Dictionary to store M2 aggregates for variance calculation
        self.m2s = defaultdict(float)

    def update(self, stat_name: str, value: float) -> None:
        """
        Update the statistics with a new value.

        Args:
            stat_name (str): Name of the statistic to update
            value (float): New value to incorporate
        """
        self.counts[stat_name] += 1
        count = self.counts[stat_name]

        # Update mean and M2 using Welford's algorithm
        delta = value - self.means[stat_name]
        self.means[stat_name] += delta / count
        delta2 = value - self.means[stat_name]
        self.m2s[stat_name] += delta * delta2

    def get_mean(self, stat_name: str) -> float:
        """Get current mean for the specified statistic."""
        return self.means[stat_name] if self.counts[stat_name] > 0 else 0.0

    def get_variance(self, stat_name: str) -> float:
        """Get current variance for the specified statistic."""
        return (
            self.m2s[stat_name] / self.counts[stat_name]
            if self.counts[stat_name] > 1
            else 0.0
        )

    def get_std(self, stat_name: str) -> float:
        """Get current standard deviation for the specified statistic."""
        return np.sqrt(self.get_variance(stat_name))

    def get_count(self, stat_name: str) -> int:
        """Get the count of samples for the specified statistic."""
        return self.counts[stat_name]

    def reset(self, stat_name: str = None) -> None:
        """
        Reset statistics for a specific stat or all stats.

        Args:
            stat_name (str, optional): Name of statistic to reset. If None, resets all stats.
        """
        if stat_name is None:
            self.counts.clear()
            self.means.clear()
            self.m2s.clear()
        else:
            if stat_name in self.counts:
                del self.counts[stat_name]
                del self.means[stat_name]
                del self.m2s[stat_name]


class EpisodeStats:
    """
    Tracks statistics for multiple episodes using Welford's algorithm.

    This class maintains separate statistics for different phases of an episode:
    - Overall statistics across the entire episode.
    - Statistics before jammers are activated.
    - Statistics after jammers are activated.
    - Statistics at the exact point of jammer activation.

    Attributes:
        jammer_ep_steps (int): The step at which jammers become active in the environment.
        current_ep_stats (WelfordStats): Tracks statistics for the current episode.
        overall_stats (WelfordStats): Tracks aggregated statistics across all episodes.
        OVERALL (str): Key for overall statistics.
        BEFORE_JAMMERS (str): Key for statistics before jammers are active.
        AFTER_JAMMERS (str): Key for statistics after jammers are active.
        AT_ACTIVATION (str): Key for statistics at the point of jammer activation.
        activation_recorded_for_episode (bool): Flag to ensure activation statistics are recorded only once per episode.
    """

    def __init__(self, jammer_ep_step: int):
        """
        Initialize the EpisodeStats class.

        Args:
            jammer_ep_step (int): The step at which jammers become active in the environment.
        """
        self.jammer_ep_steps = jammer_ep_step
        self.current_ep_stats = WelfordStats()
        self.overall_stats = WelfordStats()
        self.OVERALL = "overall"
        self.BEFORE_JAMMERS = "before_jammers"
        self.AFTER_JAMMERS = "after_jammers"
        self.AT_ACTIVATION = "at_activation"
        self.activation_recorded_for_episode = False

    def update(self, value: float, ep_step: int, numbOfJammers: int) -> None:
        """
        Update statistics with a new throughput value.

        Args:
            value (float): Throughput value to add.
            ep_step (int): Current step in the episode.
            numbOfJammers (int): Number of jammers in the environment.
        """
        self.current_ep_stats.update(self.OVERALL, value)

        # Record activation point statistics if applicable
        if ep_step == self.jammer_ep_steps and not self.activation_recorded_for_episode:
            self.current_ep_stats.update(self.AT_ACTIVATION, value)
            self.activation_recorded_for_episode = True

        # Determine whether to update before or after jammer statistics
        stat_key = (
            self.BEFORE_JAMMERS
            if self.is_before_jammer_active(ep_step, numbOfJammers)
            else self.AFTER_JAMMERS
        )
        self.current_ep_stats.update(stat_key, value)

    def end_episode(self) -> None:
        """
        End the current episode and incorporate its statistics into the overall stats.

        This method aggregates the current episode's statistics into the overall statistics and resets the current episode's statistics for the next episode.
        """
        self.overall_stats.update(
            self.OVERALL, self.current_ep_stats.get_mean(self.OVERALL)
        )
        self.overall_stats.update(
            self.BEFORE_JAMMERS, self.current_ep_stats.get_mean(self.BEFORE_JAMMERS)
        )
        self.overall_stats.update(
            self.AFTER_JAMMERS, self.current_ep_stats.get_mean(self.AFTER_JAMMERS)
        )

        if self.activation_recorded_for_episode:
            self.overall_stats.update(
                self.AT_ACTIVATION, self.current_ep_stats.get_mean(self.AT_ACTIVATION)
            )

        self.current_ep_stats = WelfordStats()
        self.activation_recorded_for_episode = False

    def get_current_episode_stats(self) -> dict:
        """
        Retrieve the statistics for the current episode.

        Returns:
            dict: A dictionary containing the current episode's statistics for overall, before jammers, after jammers, and (if available) at activation.
        """
        stats = {
            "overall": self.current_ep_stats.get_mean(self.OVERALL),
            "before_jammers": self.current_ep_stats.get_mean(self.BEFORE_JAMMERS),
            "after_jammers": self.current_ep_stats.get_mean(self.AFTER_JAMMERS),
        }

        if self.activation_recorded_for_episode:
            stats["at_activation"] = self.current_ep_stats.get_mean(self.AT_ACTIVATION)

        return stats

    def get_overall_stats(self) -> dict:
        """
        Retrieve the aggregated statistics across all episodes.

        Returns:
            dict: A dictionary containing the overall statistics for overall, before jammers, after jammers, and (if available) at activation.
        """
        stats = {
            "overall": self.overall_stats.get_mean(self.OVERALL),
            "before_jammers": self.overall_stats.get_mean(self.BEFORE_JAMMERS),
            "after_jammers": self.overall_stats.get_mean(self.AFTER_JAMMERS),
        }

        if self.overall_stats.get_count(self.AT_ACTIVATION) > 0:
            stats["at_activation"] = self.overall_stats.get_mean(self.AT_ACTIVATION)

        return stats

    def reset(self) -> None:
        """
        Reset all statistics.

        This method clears both the current episode's statistics and the overall statistics.
        """
        self.current_ep_stats = WelfordStats()
        self.overall_stats = WelfordStats()
        self.activation_recorded_for_episode = False

    def is_before_jammer_active(self, ep_step: int, numbOfJammers: int) -> bool:
        """
        Determine whether the current step is before jammers are active.

        Args:
            ep_step (int): Current step in the episode.
            numbOfJammers (int): Number of jammers in the environment.

        Returns:
            bool: True if the current step is before jammers are active or if there are no jammers, False otherwise.
        """
        return ep_step < self.jammer_ep_steps or numbOfJammers == 0

    # ------------------------------------------------------------------
    # Vectorised-eval path: per-env equivalents of update / end_episode
    # ------------------------------------------------------------------

    def _init_vec_state(self, n_envs: int) -> None:
        """Initialise per-env state for vectorised evaluation."""
        self._per_env_stats: List["WelfordStats"] = [WelfordStats() for _ in range(n_envs)]
        self._per_env_activation_recorded: List[bool] = [False] * n_envs

    def update_for_env(self, env_idx: int, value: float, ep_step: int, numbOfJammers: int) -> None:
        """Update per-env stats with a new throughput value (vectorised path)."""
        s = self._per_env_stats[env_idx]
        s.update(self.OVERALL, value)

        if ep_step == self.jammer_ep_steps and not self._per_env_activation_recorded[env_idx]:
            s.update(self.AT_ACTIVATION, value)
            self._per_env_activation_recorded[env_idx] = True

        stat_key = (
            self.BEFORE_JAMMERS
            if self.is_before_jammer_active(ep_step, numbOfJammers)
            else self.AFTER_JAMMERS
        )
        s.update(stat_key, value)

    def end_episode_for_env(self, env_idx: int) -> None:
        """Commit per-env episode stats to overall_stats and reset (vectorised path)."""
        s = self._per_env_stats[env_idx]
        self.overall_stats.update(self.OVERALL, s.get_mean(self.OVERALL))
        self.overall_stats.update(self.BEFORE_JAMMERS, s.get_mean(self.BEFORE_JAMMERS))
        self.overall_stats.update(self.AFTER_JAMMERS, s.get_mean(self.AFTER_JAMMERS))

        if self._per_env_activation_recorded[env_idx]:
            self.overall_stats.update(self.AT_ACTIVATION, s.get_mean(self.AT_ACTIVATION))

        self._per_env_stats[env_idx] = WelfordStats()
        self._per_env_activation_recorded[env_idx] = False


# * Logging
class LoggingService:
    """Manages logging of metrics and states."""

    def __init__(
        self,
        log_dir: str,
        enabled: bool = True,
        enable_eval_logger: bool = True,
        enable_state_logger: bool = True,
        log_every_n_steps: int = 10,
    ):
        """
        Initialize the LoggingService.

        Args:
            log_dir (str): Directory where logs will be stored.
            enabled (bool): If False, no logging will occur.
            enable_eval_logger (bool): If True, enable evaluation logging.
            enable_state_logger (bool): If True, enable state logging.
        """
        self.log_dir: str = log_dir
        self.enabled: bool = enabled
        self.enable_eval_logger: bool = enable_eval_logger and enabled
        self.enable_state_logger: bool = enable_state_logger and enabled
        self.log_every_n_steps: int = max(1, log_every_n_steps)
        self.eval_logger: Optional["BufferedEvaluationLogger"] = None
        self.state_logger: Optional["BufferedStateLogger"] = None

        # Store original states for reset
        self._original_eval_logger_state = self.enable_eval_logger
        self._original_state_logger_state = self.enable_state_logger

        if self.enabled:
            ut.create_dirs(self.log_dir)

    def initialize(self, log_folder: str) -> None:
        """Setup loggers for a specific folder and reset to original states."""
        if not self.enabled:
            return

        # Reset to original states before initializing
        self.enable_eval_logger = self._original_eval_logger_state
        self.enable_state_logger = self._original_state_logger_state

        ut.create_dirs(self.log_dir)
        if self.enable_eval_logger:
            self.eval_logger = BufferedEvaluationLogger(log_folder, log_every_n_steps=self.log_every_n_steps)
        if self.enable_state_logger:
            self.state_logger = BufferedStateLogger(log_folder)

    def set_eval_logger_enabled(self, enabled: bool) -> None:
        """
        Enable or disable the evaluation logger temporarily.

        Args:
            enabled (bool): If True, enables the evaluation logger.
        """
        self.enable_eval_logger = enabled and self.enabled

    def set_state_logger_enabled(self, enabled: bool) -> None:
        """
        Enable or disable the state logger temporarily.

        Args:
            enabled (bool): If True, enables the state logger.
        """
        self.enable_state_logger = enabled and self.enabled

    def log_metric(
        self,
        agent_name: str,
        ep_step: int,
        episode: int,
        metric_name: str,
        value: float,
    ) -> None:
        """Log an evaluation metric."""
        if self.enabled and self.enable_eval_logger and self.eval_logger:
            self.eval_logger.log_data(agent_name, ep_step, episode, metric_name, value)

    def log_state(
        self,
        episode: int,
        nodes: np.ndarray[net_comps.MANETNode],
        jammers: np.ndarray[net_comps.Jammer],
        users: np.ndarray[net_comps.User],
    ) -> None:
        """Log environment state."""
        if self.enabled and self.enable_state_logger and self.state_logger:
            self.state_logger.log_state(episode, nodes, jammers, users)

    def close(self) -> None:
        """Flush and close loggers."""
        if self.eval_logger:
            self.eval_logger.close()
        if self.state_logger:
            self.state_logger.close()


class BufferedEvaluationLogger:
    """
    A class to log evaluation data for MANET agents using buffered writes to a CSV file.
    """

    def __init__(self, log_folder: str, buffer_size: int = 500, log_every_n_steps: int = 10) -> None:
        """
        Initialize the BufferedEvaluationLogger class.

        Args:
            log_folder (str): The name of the log directory.
            buffer_size (int): The size of the buffer before flushing to file.
            log_every_n_steps (int): Only log every nth step to reduce CSV size.
        """
        self.log_folder = log_folder
        self.buffer_size = buffer_size
        self.log_every_n_steps: int = max(1, log_every_n_steps)
        self.buffers = {}
        self.header_written = set()

    def create_csv_file(self, agent_name: str) -> None:
        """
        Create a CSV file with headers for the given agent.

        Args:
            agent_name (str): The name of the agent.
        """
        agent_file = self._getAgentFilePath(agent_name)

        if agent_file not in self.header_written:
            ut.create_dirs(os.path.dirname(agent_file))
            with open(agent_file, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["ep_step", "episode", "value_name", "value"]
                )  # Write header
            self.header_written.add(agent_file)

    def log_data(
        self, agent_name: str, ep_step: int, episode: int, value_name: str, value: float
    ) -> None:
        """
        Add log data to the buffer.

        Args:
            agent_name (str): The name of the agent.
            ep_step (int): Current timestep in the episode (timesteps modulo ep_length)
            episode (int): The episode number.
            value_name (str): The name of the value to log.
            value (float): The value to log.
        """
        if ep_step % self.log_every_n_steps != 0:
            return

        if agent_name not in self.buffers:
            self.buffers[agent_name] = []
            self.create_csv_file(agent_name)  # Create CSV file with headers

        self.buffers[agent_name].append((ep_step, episode, value_name, value))

        # Flush buffer if it reaches the buffer size
        if len(self.buffers[agent_name]) >= self.buffer_size:
            self._flush(agent_name)

    def _flush(self, agent_name=None) -> None:
        """
        Write the buffered data to the corresponding CSV files.
        """

        agent_file = self._getAgentFilePath(agent_name)

        # Append to CSV file
        with open(agent_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.buffers[agent_name])

        # Clear the buffer for this agent after flushing
        self.buffers[agent_name].clear()

    def _flushAll(self) -> None:
        """
        Flush remaining data in buffer.
        """
        agents_to_flush = list(self.buffers.keys())
        for agent_name in agents_to_flush:

            self._flush(agent_name=agent_name)

    def _getAgentFilePath(self, agent_name: str) -> str:
        """
        To uniformly store the log files.

        Args:
            agent_name (str): name of agent (e.g. PPO_2_1_3_20_k_capacities)

        Returns:
            str: path of agent log csv file where data is stored
        """
        agentFilePath: str = os.path.join(self.log_folder, f"{agent_name}.csv")

        return agentFilePath

    def close(self) -> None:
        """
        Ensure all buffered data is flushed before exiting.
        """
        self._flushAll()


class BufferedStateLogger:
    """
    A class to log the state of MANET agents (nodes, jammers, users) using buffered writes to a JSON file.
    """

    def __init__(self, log_folder: str, buffer_size: int = 1) -> None:
        """
        Initialize the BufferedStateLogger class.

        Args:
            log_folder (str): The name of the log directory.
            buffer_size (int): The size of the buffer before flushing to file.
        """
        self.log_folder = log_folder
        self.buffer_size = buffer_size
        self.buffers = []
        self.file_name = "state_logs"
        self.header_written = False

    def create_json_file(self) -> None:
        """
        Create a JSON file for logging.
        """
        file_path = self._getFilePath()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, mode="w") as f:
            json.dump([], f)  # Initialize with an empty list

    def log_state(
        self,
        episode: int,
        nodes: np.ndarray[net_comps.MANETNode],
        jammers: np.ndarray[net_comps.Jammer],
        users: np.ndarray[net_comps.User],
    ) -> None:
        """
        Add the state data to the buffer.

        Args:
            episode (int): The episode number.
            nodes (np.ndarray[MANETNode]): np.ndarray of MANET nodes.
            jammers (np.ndarray[Jammer]): np.ndarray of jammers.
            users (np.ndarray[User]): np.ndarray of users.
        """
        if not self.header_written:
            self.create_json_file()
            self.header_written = True

        state = {
            "episode": episode,
            "nodes": [node.to_dict() for node in nodes],
            "jammers": [jammer.to_dict() for jammer in jammers],
            "users": [user.to_dict() for user in users],
        }

        self.buffers.append(state)

        # Flush buffer if it reaches the buffer size
        if len(self.buffers) >= self.buffer_size:
            self._flush()

    def _flush(self) -> None:
        """
        Write the buffered data to the JSON file.
        """
        file_path = self._getFilePath()

        with open(file_path, mode="r+") as f:
            data = json.load(f)
            data.extend(self.buffers)
            f.seek(0)
            json.dump(data, f, indent=4)

        # Clear the buffer after flushing
        self.buffers.clear()

    def _getFilePath(self) -> str:
        """
        Get the file path for the JSON log file.

        Returns:
            str: The full path of the JSON file.
        """
        return os.path.join(self.log_folder, f"{self.file_name}.json")

    def close(self) -> None:
        """
        Ensure all buffered data is flushed before exiting.
        """
        if self.buffers:
            self._flush()

    @staticmethod
    def load_state(file_path: str) -> list[dict]:
        """
        Load the state data from a JSON file.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            list[dict]: A list of state data for each episode.
        """
        with open(file_path, mode="r") as f:
            data = json.load(f)

        # Create user lookup from the first episode/state entry
        user_lookup = {
            user["id"]: net_comps.User.from_dict(user) for user in data[0]["users"]
        }

        for state in data:
            state["nodes"] = [
                net_comps.MANETNode.from_dict(node) for node in state["nodes"]
            ]
            state["jammers"] = [
                net_comps.Jammer.from_dict(jammer) for jammer in state["jammers"]
            ]
            state["users"] = [
                net_comps.User.from_dict(user, user_lookup) for user in state["users"]
            ]

        return data


def evaluate(
    env_constr: Type[envs.Basic_MANETEnv],
    agent_patterns: list[str],
    eval_params: dict,
    baselines_with_cs: list = [],
    eval_type: str = "grouped_eval",
    log_dir: str = "out/evaluations",
    log_results: bool = True,
    log_state: bool = True,
    log_eval: bool = True,
    values_to_log: List[str] = ["throughput"],
    log_every_n_steps: int = 10,
) -> None:
    """
    Perform evaluation of RL agents and baselines in a MANET environment.

    This function sets up the evaluation configuration, resolves agents and baselines,
    initializes logging, and coordinates the evaluation process.

    Args:
        env_constr (Type[envs.Basic_MANETEnv]): Constructor for the MANET environment.
        agent_patterns (list[str]): List of patterns to locate agent models.
        eval_params (dict): Dictionary of evaluation parameters.
        baselines_with_cs (list): List of tuples containing baseline agents and their condition strategies.
        eval_type (str): Type of evaluation to perform (e.g., "grouped_eval", "filter_best_agent").
        log_dir (str): Directory where evaluation logs will be stored.
        log_results (bool): Whether to log evaluation results.
        log_state (bool): Whether to log the environment state during evaluation.
        log_eval (bool): Whether to log evaluation metrics.
        values_to_log (List[str]): List of metrics to log during evaluation (e.g., "throughput", "distance_travelled").
    """
    eval_config = EvalConfig(
        env_constr=env_constr,
        agent_patterns=agent_patterns,
        baselines_with_cs=baselines_with_cs,
        eval_params=EvalParams.from_dict(eval_params),
        eval_type=eval_type,
        log_dir=log_dir,
        log_results=log_results,
    )
    eval_params_data = eval_config.eval_params

    agent_resolver = AgentResolver(
        eval_config.agent_patterns, eval_params=eval_params_data
    )
    baseline_resolver = BaselineResolver(
        baselines_with_cs=baselines_with_cs, seed=eval_params_data.seed
    )
    logging_service = LoggingService(
        log_dir=log_dir,
        enabled=log_results,
        enable_eval_logger=log_eval,
        enable_state_logger=log_state,
        log_every_n_steps=log_every_n_steps,
    )

    eval_coordinator = EvaluationCoordinator(
        config=eval_config,
        agent_resolver=agent_resolver,
        baseline_resolver=baseline_resolver,
        logging_service=logging_service,
        eval_type=eval_type,
        values_to_log=values_to_log,
    )

    eval_coordinator.run()


def evaluate_from_config(path_to_config: str) -> None:
    """
    Perform evaluation using a configuration file.

    This function loads the evaluation configuration from a YAML file,
    validates the mode, and calls the `evaluate` function with the loaded configuration.

    Args:
        path_to_config (str): Path to the YAML configuration file.
    """
    path_to_config: Path = Path(path_to_config)
    # Load the configuration
    config = ut.load_eval_config(path_to_config.parent, path_to_config.name)
    mode = config.pop("mode")
    if mode != "evaluate":
        print(f"Mode should be 'evaluate', not '{mode}'")
        return

    # Perform evaluation
    evaluate(**config)
