# Maths
import numpy as np

# More robust dictionaries

# Type hinting
from typing import Callable, Union, Optional, Tuple, List, Type

# To define abstract classes
from abc import ABC

# Open AI Gymnasium
import gymnasium as gym
from gymnasium import spaces

# Used for visualization
import queue
import threading
from time import sleep
from .GUI import MANETGUI

# Own classes
from .Network import Network
from .Components import MANETNode, ByzantineNode, Jammer, User
from . import Attackers as at

# Utility functions
from Utils import MANET_utils as ut

"""
# Environments.py
All the MANET environments model an environment in which users (e.g. Smartphones) use a MANET as backbone to communicate with each other 
while one or several malicious attackers try to interrupt their communication by jamming MANET nodes.
Additionally, some MANET nodes can behave as Byzantine nodes, silently dropping a fraction of the traffic routed through them.

__Model assumptions__
The MANET nodes' positions are gaussian distributed with the playground's center as the mean. The users occur in clusters, in which
they are also gaussian distributed with random means. Every step the users choose randomly a communication partner from the other cluster.
The jammers randomly choose a node to jam. The nodes' goal is to achieve max throughput for the users.

The MANET Environments are custom gymnasium environments with the purpose to train a model on it.

__Environments__
There are three different Environments which differ in how dynamic they are in terms of movement and network traffic. All base on the [Basic_MANETEnv](#MANET.Environments.Basic_MANETEnv) which includes the core functionality. The environments themselves follow the Gymnasium Environment, however the observation and reward are not implemented in it. Hence, to use any of the environments they must be wrapped with an [ObservationWrapper](ObservationWrappers.md) and a [RewardWrapper](RewardWrappers.md) (the order is not important). 

Main functionalities:
- Component initialization
    - Component distribution
    - Component creation
- Network simulation with [Network](Network.md)
- Node movement
- Attacker movement
- Traffic modelling

Special parameters:
    - networkConnectedAtStart (bool): One can choose whether the Nodes and Users should be distributed such that the Nodes are a connected to each other.
    - jammersSpawnNextToUsers (bool): Whether Jammers a iniated next to a User
    - steps_till_jammers_active (int): How many steps the Jammers have to find a feasible position until they are activated.
    - allow_early_ep_finish (bool): Whether an epsiode can finish before it has run through the specified `ep_length` either through achieving the maximum possible throughput or not making any advances.
    - numbOfByzantineNodes (int): Number of MANET nodes that behave as Byzantine attackers (default: 0).
    - byzantine_drop_rate (float): Fraction of traffic each Byzantine node drops (default: 0.5).
    - byzantine_attack_type (str): Type of Byzantine attack, e.g. "greyhole" (default: "greyhole").

In the `Basic_MANETEnv` however, the User to not yet move and the traffic (sending destinatio, offered load) is constant for each episode. 

Available MANET Environments:
- [Static_MANETEnv](#MANET.Environments.Static_MANETEnv): This is basically the `Basic_MANETEnv`, but the `Basic_MANETEnv should not be used since it serves as a blue print. Additionaly, since traffic is static the `done` condition here is modified, to stop the episode if the maximum throughput was reached. To account for this correctly in terms of `reward`, one should implement an extra reward in the custom `Reward Wrapper` for when the `done` condition is reached. The `done` flag in all `Gymnasium` Environments is used to signify the a goal to have been reached. Hence, this makes sense.

- [Less_Static_MANETEnv](#MANET.Environments.Less_Static_MANETEnv):  In the Less_Static_MANETEnv Users start moving slowly and in clusters (according to the reference point group mobility model). The sending behaviour stays constant.

- [Dynamic_MANETEnv](#MANET.Environments.Dynamic_MANETEnv): The Dynamic_MANETEnv` inherites from the `Less_Static_MANETEnv` and additionaly models traffic to be more realistic. Here, the destinations change randomly according to a simplified Markov Chain and the offered load of the users changes accoring to and actual Markov Chain.
"""


class Basic_MANETEnv(ABC, gym.Env):
    """
    Basic_MANETEnv is the blueprint for all environments.

    Attributes:
        metadata (dict): Metadata for the environment.
        render_mode (str): Mode for rendering the environment.

        action_space (gym.spaces.Box): Action space for the environment.
        observation_space (gym.spaces.Box): Observation space for the environment.

        SIZE (int): Size of the playground.
        numbOfNodes (int): Number of MANET nodesduring the current episode.
        numbOfJammers (int): Number of jammers during the current episode.
        numbOfJammersRange (Union[int, Tuple[int,int]]): Can be either an integer for a constant number of jammers, or an interval, from which for each episode the number of jammers in that episode will be uniformly sampled.
        numbOfUsers (int): Number of users during the current episode.
        numbOfUsersRange (Union[int, Tuple[int,int]]): Can be either an integer for a constant number of users, or am interval, from which for each episode the number of users in that episode will be uniformly sampled.
        nodeSignalPow_mW (int): Signal power of the nodes in mW.
        userSignalPow_mW (int): Signal power of the users in mW.
        jammerSignalPow_mW (int): Signal power of the jammers in mW.

        ep_length (float): Length of an episode.
        ep_step (int): Current step in the episode.
        seed (int): Seed for random number generation.

        networkConnectedAtStart (bool): Whether the network is connected at the start.
        jammersSpawnNextToUsers (bool): Whether jammers spawn next to users.
        steps_till_jammers_active (int): Steps until jammers become active.
        step_jammers_start_moving (int): Steps until the jammers are allowed to start moving.
        allow_early_ep_finish (bool): Whether to allow early episode finish.

        network (Network): The network object.
        routing_func (Callable): Routes the offered loads given the network graph and a set of source, sink and demand tuples.

        numbOfByzantineNodes (int): Number of nodes that behave as Byzantine attackers.
        byzantine_drop_rate (float): Fraction of traffic each Byzantine node drops.
        byzantine_attack_type (str): The type of Byzantine attack (e.g. "greyhole").

        node_positions (np.ndarray): Positions of the nodes.
        nodes (list): List of MANET nodes.
        node_step_size (int): Step size for node movement.
        lastAction (np.ndarray): The previous action executed

        jammer_positions (np.ndarray): Positions of the jammers.
        jammers (list): List of jammers.
        attackerModel (class): Model for the attackers for the current episode.
        attacker (AttackerModel): Attacker model object.

        user_positions (np.ndarray): Positions of the users.
        users (list): List of users.
        userClusters (list): List of user clusters.
        user_to_idx (dict): Mapping from user to index.
        user_to_cluster_idx (dict): Mapping from user to cluster index.
        clusterSizes (np.ndarray): Sizes of the clusters.

        max_log_size (int): Maximum size of the log.
        check_interval (int): Interval for checking conditions.
        lastThroughputs (list): List of last throughputs.
        lastDistances (list): List of last distances.

        gui_thread (threading.Thread): Thread for the GUI.
        gui_closed (bool): Whether the GUI is closed.
        gui_queue (queue.Queue): Queue for the GUI.

        attacker_random_gen (np.random.Generator): Random generator controlling the smapling of the attacker models, if a list of attacker models was given.
        numb_of_random_gen (np.random.Generator): Random generator used if given an interval of users or jammers to sample uniformly from it.
        node_random_gen (np.random.Generator): Random generator controlling random distribution of the MANET nodes
        jammer_random_gen (np.random.Generator): Random generator controlling random distribution of the jammers
        user_random_gen (np.random.Generator): Random generator controlling random distribution of the users and the dynamic behaviour during the episode
        network_random_gen (np.random.Generator): Random generator controlling all randomicity involved in the network model.

        MAX_OFFERED_LOAD (int): the maximum offered load
        TRANSMITTER_PROB (float): probability for a user to be a source node or not
    """

    metadata = {"render_modes": ["plot"]}

    # Playground size
    SIZE = 100

    nodeSignalPow_mW = 200  # in mW
    userSignalPow_mW = 10  # in mW
    jammerSignalPow_mW = 300  # in mW

    # Max offered load
    MAX_OFFERED_LOAD: int = 20e6  # 20 Mb

    # Transmitter
    TRANSMITTER_PROB = 0.2

    def __init__(
        self,
        numbOfNodes: int,
        numbOfJammers: Union[int, Tuple[int, int]],
        numbOfUsers: Union[int, Tuple[int, int]],
        render_mode: str = None,
        seed: int = None,
        networkConnectedAtStart: bool = False,
        jammersSpawnNextToUsers: bool = False,
        attackerModel: Union[
            at.AttackerModel, List[at.AttackerModel]
        ] = at.Static_GreedyJammers,
        allow_early_ep_finish: bool = False,
    ):
        """
        Initialize a Basic_MANETEnv environment.

        Args:
            numbOfNodes (int): Number of MANET nodes.
            numbOfJammers (Union[int, Tuple[int, int]]): Number of jammers as an integer or an interval from which the number is uniformly randomly sampled for each episode.
            numbOfUsers (Union[int, Tuple[int, int]]): Number of users as an integer or an interval from which the number is uniformly randomly sampled for each episode.
            render_mode (str, optional): Use "plot" to get the GUI, else use None. Default is None.
            seed (int, optional): Seed for random number generation. Default is None.
            networkConnectedAtStart (bool, optional): Whether the network is connected at the start. Default is False.
            jammersSpawnNextToUsers (bool, optional): Whether jammers spawn next to users. Default is False.
            attackerModel (at.AttackerModel, optional): Model for the attackers. Default is at.Static_GreedyJammers.
            allow_early_ep_finish (bool, optional): Whether to allow early episode finish. Default is False.
        """

        # Initialize gym env
        super().__init__()

        # Set render mode
        self.render_mode = render_mode

        # Number of nodes, jammers, and users
        self.numbOfNodes = numbOfNodes
        self.numbOfJammers = (
            numbOfJammers if isinstance(numbOfJammers, int) else numbOfJammers[0]
        )
        self.numbOfUsers = (
            numbOfUsers if isinstance(numbOfUsers, int) else numbOfUsers[0]
        )

        self.numbOfJammersRange = numbOfJammers
        self.numbOfUsersRange = numbOfUsers

        # Action and observation space
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.numbOfNodes, 2), dtype=np.float32
        )
        self.observation_space = None

        # Training params
        self.ep_length = np.inf
        self.ep_step = None

        self.networkConnectedAtStart = networkConnectedAtStart
        self.jammersSpawnNextToUsers = jammersSpawnNextToUsers
        self.steps_till_jammers_active = 0
        self.step_jammers_start_moving = 0
        self.allow_early_ep_finish = allow_early_ep_finish

        # Network
        self.network = None
        self.routing_func: Callable = None

        # Network components
        self.node_positions = None
        self.nodes: np.ndarray[MANETNode] = None
        self.node_step_size = 2

        # Byzantine nodes
        self.numbOfByzantineNodes = 0
        self.byzantine_drop_rate = 0.5
        self.byzantine_attack_type = "greyhole"
        self.byzantine_capacity_inflation_factor = 3.0
        self.byzantine_jamming_power_factor = 1.0
        self.byzantine_pos_offset = None
        self.byzantine_obs_user_dir_offset = None
        self.byzantine_obs_capacity_factor = 1.0
        self.byzantine_obs_demand_factor = 1.0
        self.byzantine_obs_interference_factor = 1.0
        self.byzantine_obs_noise_std = 0.0
        self.byzantine_obs_capacity_deflate_factor = 1.0
        self.byzantine_obs_demand_deflate_factor = 1.0
        self.byzantine_obs_replay_delay = 0
        self.byzantine_obs_magnitude_range = None
        self.node_to_idx = None
        self.lastAction = np.zeros(shape=(self.numbOfNodes, 2))

        self.jammer_positions = None
        self.jammers: np.ndarray[Jammer] = None
        self.attackerModel = attackerModel
        self.attacker = None

        self.user_positions = None
        self.users: np.ndarray[User] = None
        self.user_to_idx = None
        self.userClusters = None
        self.clusterSizes = None
        self.user_to_cluster_idx = None

        # Log throughputs and distance
        self.max_log_size = 500
        self.check_interval = 500

        self.lastThroughputs = None
        self.lastDistances = None

        # Initialize all above values
        self.seed = seed
        self.attacker_random_gen = np.random.default_rng(seed=None)
        self.numb_of_random_gen = np.random.default_rng(seed=None)
        self.node_random_gen = np.random.default_rng(seed=None)
        self.jammer_random_gen = np.random.default_rng(seed=None)
        self.user_random_gen = np.random.default_rng(seed=None)
        self.network_random_gen = np.random.default_rng(seed=None)
        self.reset(seed=self.seed)

        # * Initialize the Gui in a thread
        self.gui_thread = None
        self.gui_closed = False

        if self.render_mode == "plot":
            self.gui_queue = queue.Queue()
            self.gui_queue.put(
                (
                    self.node_positions,
                    self.jammer_positions,
                    self.user_positions,
                )
            )
            self.gui_thread = threading.Thread(target=self.start_gui)
            self.gui_thread.start()

    # * Main Components
    def step(self, action) -> tuple:
        """
        Execute one time step within the environment. One step includes:

            1. Moving Nodes (Users and Jammers)
            2. (Simulate desination and offered load choice)
            3. Update the network and send the demand
            4. Log the throughputs and movement to potentially cut episode short

        Args:
            action (np.ndarray): Actions to be taken by the nodes.

        Returns:
            observation (np.ndarray): The current observation of the environment.
            reward (float): The reward obtained after taking the action.
            terminated (bool): Whether the episode has ended.
            truncated (bool): Whether the episode was truncated.
            info (dict): Additional information about the environment.
        """
        # Store last action
        self.lastAction = action

        # Increase episode step
        self.ep_step += 1

        # Move the nodes, users, and jammers
        self.moveNodes(action)
        self.moveUsers()
        self.moveJammers()

        # Model sending behaviour and offered load
        self.simulateTrafficPartnerChoice()
        self.simulateOfferedLoadChoice()

        # Update the network
        self.network.updateNetwork()
        self.network.routeOfferedLoads()

        # To finish episode early if no movement or change in throughput is achieved over some time
        # Hence, these values need to be logged to check such occurence
        self.logLastThroughputsAndMovement(action, self.network.getThroughput())

        reward = self.reward()

        observation = self.observation()
        info = self.info()

        terminated = self.done()
        truncated = self.truncated()

        self.render()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None) -> tuple:
        """
        Reset the environment to an initial state though new state.

        Args:
            seed (int, optional): Seed for random number generation. Defaults to None.
            options (dict, optional): Additional options for resetting the environment. Defaults to None.

        Returns:
            observation (np.ndarray): The initial observation of the environment.
            info (dict): Additional information about the environment.
        """

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        if seed is not None:
            self.node_random_gen = np.random.default_rng(seed=seed)
            self.jammer_random_gen = np.random.default_rng(seed=(seed + 1))
            self.user_random_gen = np.random.default_rng(seed=(seed + 2))
            self.network_random_gen = np.random.default_rng(seed=(seed + 3))
            self.numb_of_random_gen = np.random.default_rng(seed=(seed + 4))
            self.attacker_random_gen = np.random.default_rng(seed=(seed + 5))

        # Reset episode step
        self.ep_step = 0
        # Initialize numb of components
        self.numbOfJammers = resolve_numb_of(
            self.numbOfJammersRange, self.numb_of_random_gen
        )
        self.numbOfUsers = resolve_numb_of(
            self.numbOfUsersRange, self.numb_of_random_gen
        )

        # Randomize Byzantine attack magnitude each episode if a range is set
        if self.byzantine_obs_magnitude_range is not None:
            lo, hi = self.byzantine_obs_magnitude_range
            magnitude = self.node_random_gen.uniform(lo, hi)
            angle = self.node_random_gen.uniform(0, 2 * np.pi)
            self.byzantine_obs_user_dir_offset = [
                float(magnitude * np.cos(angle)),
                float(magnitude * np.sin(angle)),
            ]

        # Initialize components
        self.initializeComponents()

        # Initialize attacker model
        self.initialize_attacker(self.seed)

        # Initialize network
        self.network = Network(
            self.nodes,
            self.jammers,
            self.users,
            np_random=self.network_random_gen,
        )

        if self.routing_func is not None:
            self.network.set_routing_func(self.routing_func)

        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.user_to_idx = {user: idx for idx, user in enumerate(self.users)}
        # Demand and destination
        self.initializeSendersAndReceivers()
        self.initializeDemand()

        # Signal GUI reset if rendering
        if self.render_mode == "plot" and hasattr(self, "gui_queue"):
            self.gui_queue.put(
                (
                    "reset",
                    self.node_positions,
                    self.jammer_positions,
                    self.user_positions,
                )
            )

        # Truncated values
        self.lastThroughputs = []
        self.lastDistances = []

        observation = self.observation()
        info = self.info()

        return observation, info

    def observation(self) -> np.ndarray:
        """
        Get the current observation of the environment. Handled by the Observation Wrappers

        Returns:
            np.ndarray: The current observation of the environment.
        """
        return np.zeros((self.numbOfNodes,))

    def done(self) -> bool:
        """
        Check if the episode is done. The done signal indicates success to the agent.

        Returns:
            bool: True if the episode is done, False otherwise.
        """

        return False

    def truncated(self) -> bool:
        """
        Check if the episode is truncated. The truncated signal indicates an early episode exit due to failure of achieving a certain goal
        or failure of making progress. In our case it is triggered when the epsiode is finished and if the `allowEarlyEpFinish` flag was set to true, the
        it is also triggered, if the MANET nodes cease to move or fail to achieve any significant change in throughput over a certain time.

        Returns:
            bool: True if the episode is truncated, False otherwise.
        """
        episode_over = self.ep_step >= self.ep_length

        if (
            self.ep_step % self.check_interval == 0
            and self.allow_early_ep_finish
            and self.ep_step >= self.max_log_size
            and self.ep_step >= self.steps_till_jammers_active
        ):
            noThroughputChange = self.noThroughputChange()
            noMovement = self.noMovement()

            episode_over = episode_over or noThroughputChange or noMovement

        return episode_over

    def info(self) -> dict:
        """
        Get additional information about the environment.

        Returns:
            dict: Additional information.
        """
        info = {}
        return info

    def render(self) -> None:
        """
        Render the environment.
        """
        if self.render_mode == "plot":
            if self.gui_closed:
                self.render_mode = None
                self.stop_gui_thread()
            else:
                edges = list(self.network.getGraph().edges)
                throughput = self.network.getThroughput()
                self.gui_queue.put(
                    (
                        "update",
                        self.node_positions,
                        self.jammer_positions,
                        self.user_positions,
                        edges,
                        throughput,
                    )
                )
                while not self.gui_queue.empty() and self.render_mode == "plot":
                    sleep(0.001)

    def reward(self) -> float:
        """
        Calculate the reward for the current state of the environment. Handled by the reward wrappers

        Returns:
            float: The calculated reward.
        """
        return 0

    # * Important setters
    def set_ep_length(self, ep_length) -> None:
        """
        Set the episode length.

        Args:
            ep_length (int): The length of the episode.
        """
        self.ep_length = ep_length

    def set_steps_till_activate_jammers(self, steps: int) -> None:
        """
        Set the number of steps until jammers become active.

        Args:
            steps (int): The number of steps until jammers become active.
        """
        self.steps_till_jammers_active = steps

    def set_step_jammers_start_moving(self, steps: int) -> None:
        """
        Set the number of steps until jammers become active.

        Args:
            steps (int): The number of steps until jammers become active.
        """
        self.step_jammers_start_moving = steps

    def set_node_step_size(self, step_size: int) -> None:
        """
        Setter for parameter step_size, which determines the scale for the actions. The actions usually are in [0,1]^2
        and the default step_size is 2.

        Args:
            step_size (int): Scale for the node actions.
        """
        self.node_step_size = step_size

    def set_routing_function(self, routing_func: Callable) -> None:
        """Set the network's routing function, which determines how the data in the network is routed.

        Args:
            routing_func (Callable): Function that determines how the data in the network is routed.
        """
        self.routing_func = routing_func

    def set_byzantine_params(
        self,
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
    ) -> None:
        """Set the Byzantine node parameters used when creating ByzantineNode instances.

        Args:
            numbOfByzantineNodes (int): Number of nodes that behave as Byzantine attackers.
            byzantine_drop_rate (float): Fraction of traffic each Byzantine node drops (greyhole/sinkhole).
            byzantine_attack_type (str or List[str]): The type of Byzantine attack, or a list of attack
                types assigned round-robin to Byzantine nodes. Valid values per node:
                "greyhole", "sinkhole", "selective_jamming", "position_spoofing",
                "obs_user_dir", "obs_capacity", "obs_demand", "obs_interference_lie", "obs_noise",
                "obs_capacity_deflate", "obs_demand_deflate", "obs_replay".
                Example: ["obs_demand", "obs_interference_lie"] gives alternating attack types.
            byzantine_capacity_inflation_factor (float): Multiplier on reported link capacity for sinkhole.
            byzantine_jamming_power_factor (float): Multiplier on signalPower_mW emitted as interference
                for selective_jamming nodes.
            byzantine_pos_offset (array-like or None): 2-D offset added to the reported position for
                position_spoofing nodes. None defaults to [0, 0].
            byzantine_obs_user_dir_offset (array-like or None): 2-D offset added to reported closest-user
                direction vectors for obs_user_dir nodes.
            byzantine_obs_capacity_factor (float): Multiplier on reported in/out capacities for obs_capacity.
            byzantine_obs_demand_factor (float): Multiplier on reported sender/receiver demand for obs_demand.
            byzantine_obs_interference_factor (float): Multiplier on reported received interference for
                obs_interference_lie nodes (0.0 hides all interference).
            byzantine_obs_noise_std (float): Std dev of zero-mean Gaussian noise added to the full broadcast
                state slice for obs_noise nodes.
            byzantine_obs_capacity_deflate_factor (float): Multiplier (<1) on reported in/out capacities
                for obs_capacity_deflate nodes — repels honest nodes from the Byzantine position.
            byzantine_obs_demand_deflate_factor (float): Multiplier (<1) on reported sender/receiver demand
                for obs_demand_deflate nodes — causes honest nodes to abandon high-traffic areas.
            byzantine_obs_replay_delay (int): Number of steps the Byzantine node replays its own stale
                observation for obs_replay nodes (0 = disabled).
        """
        self.numbOfByzantineNodes = numbOfByzantineNodes
        self.byzantine_drop_rate = byzantine_drop_rate
        self.byzantine_attack_type = byzantine_attack_type
        self.byzantine_capacity_inflation_factor = byzantine_capacity_inflation_factor
        self.byzantine_jamming_power_factor = byzantine_jamming_power_factor
        self.byzantine_pos_offset = byzantine_pos_offset
        self.byzantine_obs_user_dir_offset = byzantine_obs_user_dir_offset
        self.byzantine_obs_capacity_factor = byzantine_obs_capacity_factor
        self.byzantine_obs_demand_factor = byzantine_obs_demand_factor
        self.byzantine_obs_interference_factor = byzantine_obs_interference_factor
        self.byzantine_obs_noise_std = byzantine_obs_noise_std
        self.byzantine_obs_capacity_deflate_factor = byzantine_obs_capacity_deflate_factor
        self.byzantine_obs_demand_deflate_factor = byzantine_obs_demand_deflate_factor
        self.byzantine_obs_replay_delay = byzantine_obs_replay_delay
        self.byzantine_obs_magnitude_range = byzantine_obs_magnitude_range

    def set_max_log_size(self, max_log_size: int) -> None:
        """
        Set the maximum size of the log list.

        Args:
            max_log_size (int): The maximum size of the log.
        """
        self.max_log_size = max_log_size

    def set_check_interval(self, check_interval: int) -> None:
        """
        Set the interval at which to check and trim the logs.

        Args:
            check_interval (int): The interval at which to check and trim the logs.
        """
        self.check_interval = check_interval

    # * Initialize components
    def initializeComponents(self) -> None:
        """
        Initialize the components of the network. First they are distributed, or in other words, their
        positions are determined and then their objects are initialized. Nodes, jammers and users are
        distributed randomly in \\([0,1]^2\\).

        For nodes and users, two distribution methods are considered based on system connectivity:

        - If connectivity is ensured, nodes and users are distributed with the nodes placed at user clusters' center using a Monte Carlo algorithm ensuring that all nodes are connected.
        - If connectivity is not ensured, nodes and users are simply distributed randomly.
        """
        if self.networkConnectedAtStart:
            # Distribute the nodes and users such that the nodes are all connected.
            self.distributeComponentsConnected()
        else:
            # Distribute without special regard for connectivity
            self.distributeComponents()
        self.createComponents()

    # Distribute in connected manner
    def distributeComponentsConnected(self) -> None:
        """
        Nodes and users are distributed such that they are connected, with the nodes placed
        at the user clusters' center using a Monte Carlo algorithm ensuring that all nodes are
        connected.

        Jammers are placed either by:

        - Randomly selecting a spot in \\([0,1]^2\\) and distributing them normally around it, reflecting a coordinated jamming attack.
        - Selecting a user's position and adding a relatively small random uniform displacement vector. If multiple jammers are present, they are ensured not to choose the same user's position.
        """
        self.distributeNodesAndUsersConnected()
        self.distributeJammers()

    def distributeNodesAndUsersConnected(self) -> None:
        """
        Distributes the nodes and users so that the nodes are already connected.

        The number of clusters \\( x_c \\) is randomly chosen according to an exponential distribution \\(X_c \\sim \\text{Exp}(\\lambda) \\) with \\(\\alpha \\in [0,1]\\) and \\(\\lambda = 10 \\cdot \\alpha \\), such that \\(x_c \\in \\left[2, |U|\\right]\\).

        Given \\( x_c \\), the number of users per cluster follows a Dirichlet distribution \\(\\text{Dir}((\\hat{\\beta}_1 , \\dots, \\hat{\\beta}_{x_c})) \\) with \\(\\beta \\in [0,1]\\) and \\(\\hat{\\beta}_1 = \\dots = \\hat{\\beta}_{x_c} = 10 \\cdot \\beta\\). A resolving algorithm ensures each cluster includes at least one user and there are exactly as many users distributed over the cluster as total users.

        Cluster centers are uniformly randomly chosen inside \\([0,1]^2\\). Nodes are placed at each cluster center, starting with the largest cluster. If \\(|M| > x_c\\), a bridging algorithm assigns node demand based on pairwise cluster distances, with remaining nodes placed between cluster pairs starting with the highest demand.

        Connectivity is checked to ensure all nodes form a joint graph; if not, the process restarts. Then users are normally distributed around their cluster centers.
        """
        numbOfClusters = self.determineNumbOfClusters()
        self.clusterSizes = self.randChooseClusterSizes(numbOfClusters)
        clusterCenters, self.node_positions = self.randDistrClusterCentersAndNodes(
            numbOfClusters
        )
        self.user_positions = self.distributeUsersOverClusters(clusterCenters)

    def randDistrClusterCentersAndNodes(self, numbOfClusters) -> tuple:
        """
        - Generate random cluster centers, given the number of Clusters \\(x_c\\)
        - Place the nodes at the clusters' center.
            - If \\(|M| > x_c\\), a bridging algorithm assigns node demand based on pairwise cluster distances, with remaining nodes placed between cluster pairs starting with the highest demand.
            - Else \\(|M| \\leq x_c \\), start from the largest cluster and place a node a its center
        - Check for connectivity.
            - If the nodes are not connected / forming a joint graph, restart.
            - Else, this process is finished.

        Args:
            numbOfClusters (int): Number of clusters to generate.

        Returns:
            tuple: (clusterCenters, node_positions)
                - clusterCenters (np.ndarray): Cluster centers to be used for distribution of users.
                - node_positions (np.ndarray): Positions of all nodes.
        """
        connected = False

        while not connected:
            # Randomly sample cluster centers within the area defined by SIZE
            clusterCenters = self.randDistrClusterCenters(numbOfClusters)

            # Generate node positions around the cluster centers
            node_positions = self.distributeNodesConnected(
                clusterCenters, numbOfClusters
            )

            # Check if node positions result in a connected network
            nodes = [
                MANETNode(f"Node {i}", pos) for i, pos in enumerate(node_positions)
            ]
            connected = ut.are_nodes_connected(nodes)

        return clusterCenters, np.array(node_positions, dtype=np.float32)

    def distributeNodesConnected(self, clusterCenters, numbOfClusters) -> np.ndarray:
        """
        Assign positions to nodes based on cluster centers:

        - If \\(|M| > x_c\\), a bridging algorithm assigns node demand based on pairwise cluster distances, with remaining nodes placed between cluster pairs starting with the highest demand.
        - Else \\(|M| \\leq x_c \\), start from the largest cluster and place a node a its center

        Args:
            clusterCenters (np.ndarray): Array of cluster center positions.
            numbOfClusters (int): Number of clusters.

        Returns:
            np.ndarray: Array of node positions.
        """
        if self.numbOfNodes <= numbOfClusters:
            # If there are fewer or equal nodes than clusters, just assign node positions to cluster centers
            return clusterCenters[: self.numbOfNodes]
        else:
            # Otherwise, place the remaining nodes between the cluster centers
            remaining_nodes = self.numbOfNodes - numbOfClusters
            remaining_positions = self.placeRemainingNodes(
                clusterCenters, remaining_nodes
            )
            return np.concatenate((clusterCenters, remaining_positions))

    def placeRemainingNodes(self, clusterCenters, remaining_nodes) -> np.ndarray:
        """
        Place remaining nodes between cluster centers to create bridges and ensure network connectivity.

        Args:
            clusterCenters (np.ndarray): Array of cluster center positions.
            remaining_nodes (int): Number of remaining nodes to place.

        Returns:
            np.ndarray: Positions of the remaining nodes.

        The bridging algorithm works as follows:

        1. Calculate the distances between each pair of cluster centers.
        2. Assign a demand for nodes between each pair of cluster centers based on the distance, with larger distances having higher demand.
        3. Sort the pairs by their demand in descending order.
        4. Distribute the remaining nodes to fulfill the demand, starting with the largest demand.
        5. Place nodes evenly between each pair of cluster centers according to the allocated demand.
        """
        num_clusters = len(clusterCenters)
        dist_threshold = self.SIZE / 8  # Distance threshold to create bridge nodes

        # Calculate distances between each pair of cluster centers
        requests = []
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                dist = np.linalg.norm(clusterCenters[i] - clusterCenters[j])
                request = int(dist / dist_threshold)
                requests.append((request, i, j))

        # Sort requests by distance
        requests.sort(reverse=True)

        # Distribute remaining nodes between cluster centers
        remaining_positions = []
        nodes_to_place = remaining_nodes
        bucket_allocation = [0] * len(requests)

        for request, i, j in requests:
            if nodes_to_place == 0:
                break

            # Distribute nodes between cluster centers proportionally to the request
            num_nodes = min(request, nodes_to_place)
            nodes_to_place -= num_nodes
            bucket_allocation[i] += num_nodes

        # Evenly distribute any leftover nodes
        while nodes_to_place > 0:
            for idx in range(len(bucket_allocation)):
                if nodes_to_place == 0:
                    break
                bucket_allocation[idx] += 1
                nodes_to_place -= 1

        # Place the nodes between the cluster centers according to the bucket allocation
        for idx, num_nodes in enumerate(bucket_allocation):
            if num_nodes > 0:
                _, i, j = requests[idx]
                node_i_pos = clusterCenters[i]
                node_j_pos = clusterCenters[j]
                for k in range(num_nodes):
                    t = (k + 1) / (num_nodes + 1)
                    new_pos = (1 - t) * node_i_pos + t * node_j_pos
                    remaining_positions.append(new_pos)

        # Combine cluster center positions and remaining node positions
        return remaining_positions

    # Distribute without special regard for connectivity
    def distributeComponents(self) -> None:
        """
        Distribute the components of the network without regard for node connectivity.

        Distribution Details:

        - Cluster Centers: Uniformly randomly chosen across the playground.
        - Users: Normally distributed around their cluster centers.
        - Nodes: Normally distributed around the center of the playground.
        - Jammers are placed either by:

            1. Randomly selecting a spot in \\([0, \\text{SIZE}]^2\\) and distributing them normally around it, reflecting a coordinated jamming attack.
            2. Selecting a user's position and adding a relatively small random uniform displacement vector. If multiple jammers are present, they are ensured not to choose the same user's position.

        The positions are clipped to ensure they fall within the playground boundaries.
        """
        self.distributeNodes()
        self.distributeUsers()
        self.distributeJammers()

    def distributeNodes(self) -> None:
        """
        Distribute the nodes on the playground.

        The nodes' positions are generated using a normal (Gaussian) distribution centered at the middle of the playground.
        The positions are then clipped to ensure they fall within the playground boundaries.

        The normal distribution is defined as:

        - Mean (\\(\\mu\\)): SIZE / 2 (center of the playground)
        - Standard deviation (\\(\\sigma\\)): SIZE / 2

        The positions are clipped to the range [0, SIZE].
        """
        self.node_positions = self.node_random_gen.normal(
            self.SIZE / 2, self.SIZE / 2, (self.numbOfNodes, 2)
        )
        np.clip(
            a=self.node_positions,
            a_min=0,
            a_max=self.SIZE,
            out=self.node_positions,
        )

    def distributeUsers(self) -> None:
        """
        Distribute the users on the playground.

        The users are distributed into clusters. The number of clusters is determined randomly using an exponential distribution.
        The sizes of the clusters are determined using a Dirichlet distribution, ensuring that the total number of users is maintained.
        The cluster centers are uniformly distributed across the playground, and the users are normally distributed around these centers.

        The exponential distribution for the number of clusters is defined as:

        - Scale \\(\\lambda = 10 \\cdot \\alpha \\)

        The Dirichlet distribution for cluster sizes is defined as \\(\\text{Dir}((\\hat{\\beta}_1 , \\dots, \\hat{\\beta}_{x_c})) \\) with \\(\\beta \\in [0,1]\\) and \\(\\hat{\\beta}_1 = \\dots = \\hat{\\beta}_{x_c} = 10 \\cdot \\beta\\)

        The normal distribution for user positions within clusters is defined as:

        - Mean (\\(\\mu\\)): Cluster center
        - Covariance matrix (\\(\\Sigma\\)): Diagonal matrix with SIZE / 10 as the diagonal elements

        The positions are clipped to playground's size.
        """
        numbOfClusters = self.determineNumbOfClusters()
        self.clusterSizes = self.randChooseClusterSizes(numbOfClusters)
        clusterCenters = self.randDistrClusterCenters(numbOfClusters)
        self.user_positions = self.distributeUsersOverClusters(clusterCenters)

    def determineNumbOfClusters(self, alpha=0.6) -> int:
        """
        Determine the number of clusters.

        Args:
            alpha (float, optional): Controls how much the random choice of number of clusters varies. Default is 0.25.

        Returns:
            int: Randomly generated number of clusters.

        The number of clusters is determined using an exponential distribution with the following parameters:

        - Minimum number of clusters: 2
        - Maximum number of clusters: numbOfUsers
        - Scale \\( \\lambda = 10 \\cdot \\alpha \\)
        """
        min_clusters = 2
        max_clusters = self.numbOfUsers
        # Determine number of clusters
        scale = alpha * 10

        # Guard against division by zero
        if scale <= 0:
            # If scale is zero or negative, just return the minimum number of clusters
            return min_clusters

        numbOfClusters = np.clip(
            self.user_random_gen.geometric(p=1 / scale),
            a_min=min_clusters,
            a_max=max_clusters,
        )
        return int(numbOfClusters)

    def randChooseClusterSizes(self, numbOfClusters, beta=0.1) -> list:
        """
        Randomly chooses a cluster size, influenced by beta.

        Args:
            numbOfClusters (int): Number of clusters to be created.
            beta (float, optional): Influences how strongly the clusters can vary in size. Default is 0.1.

        Returns:
            list: Sorted list of the cluster sizes.

        The cluster sizes are determined using a Dirichlet distribution with the following parameters:

        - Concentration parameter \\(\\alpha \\) = [beta * 10] * numbOfClusters

        The sizes are then rounded to the nearest integer and adjusted to ensure no cluster has zero size and the total number of users is maintained.
        """
        # Generate the initial cluster sizes using Dirichlet distribution
        distribution = np.array([beta * 10] * numbOfClusters)
        cluster_sizes = self.user_random_gen.dirichlet(distribution) * self.numbOfUsers
        cluster_sizes = np.round(cluster_sizes).astype(int)

        # Ensure no clusters have zero size
        cluster_sizes = np.clip(cluster_sizes, 1, None)

        # If any cluster sizes are zero, adjust by redistributing excess
        total_users = np.sum(cluster_sizes)
        diff = self.numbOfUsers - total_users

        while diff != 0:
            # Adjust the sizes to ensure the sum equals self.numbOfUsers
            indices = np.argsort(-cluster_sizes)  # Get indices for descending sort
            for i in range(abs(diff)):
                cluster_sizes[indices[i % numbOfClusters]] += np.sign(diff)

            # Ensure no clusters have zero size
            cluster_sizes = np.clip(cluster_sizes, 1, None)

            # If any cluster sizes are zero, adjust by redistributing excess
            total_users = np.sum(cluster_sizes)
            diff = self.numbOfUsers - total_users

        assert (
            np.sum(cluster_sizes) == self.numbOfUsers
        ), "Cluster sizes do not sum to self.numbOfUsers"
        assert np.all(cluster_sizes > 0), "Cluster sizes contain zero values"

        return np.sort(cluster_sizes)[::-1]  # Return sizes in descending order

    def randDistrClusterCenters(self, numbOfClusters) -> np.ndarray:
        """
        Generate random cluster centers.

        Args:
            numbOfClusters (int): Number of clusters to generate.

        Returns:
            np.ndarray: Array of cluster center positions.

        The cluster centers are uniformly distributed across the playground.
        The positions are clipped to the playground's size.
        """
        clusterCenters = np.clip(
            self.user_random_gen.uniform(0, self.SIZE, size=(numbOfClusters, 2)),
            0,
            self.SIZE,
        )

        return clusterCenters

    def distributeUsersOverClusters(self, clusterCenters) -> np.ndarray:
        """
        Distribute the users on the playground.

        Args:
            clusterCenters (np.ndarray): Array of cluster center positions.

        Returns:
            np.ndarray: Array of user positions.

        The users are normally distributed around the cluster centers.
        The normal distribution for user positions within clusters is defined as:

        - Mean (\\(\\mu\\)): Cluster center
        - Covariance matrix (\\(\\Sigma\\)): Diagonal matrix with SIZE / 4 as the diagonal elements

        The positions are clipped to the range [0, SIZE].
        """
        user_positions = []

        cov = np.diag([self.SIZE / 4, self.SIZE / 4])
        for i, center in enumerate(clusterCenters):
            mean = center
            cluster_user_positions = self.user_random_gen.multivariate_normal(
                mean=mean, cov=cov, size=self.clusterSizes[i]
            )
            cluster_user_positions = np.clip(cluster_user_positions, 0, self.SIZE)
            user_positions.extend(cluster_user_positions)

        return np.array(user_positions, dtype=np.float32)

    def distributeJammers(self) -> None:
        """
        Randomly place the jammers on the playground.

        The jammers can be placed in two ways:

        1. If jammersSpawnNextToUsers is True, the jammers are placed next to randomly selected users with a small random displacement.
        2. If jammersSpawnNextToUsers is False, the jammers are normally distributed around a randomly selected point on the playground.

        The normal distribution for jammer positions is defined as:

        - Mean (\\(\\mu\\)): Randomly selected point on the playground
        - Covariance matrix (\\(\\Sigma\\)): Diagonal matrix with SIZE / 4 as the diagonal elements

        The positions are clipped to the range [0, SIZE].
        """
        if self.jammersSpawnNextToUsers:
            user_positions = self.user_positions.copy().tolist()

            jammer_positions = []
            for _ in range(self.numbOfJammers):
                index = self.jammer_random_gen.integers(0, len(user_positions))
                selected_pos = user_positions.pop(index)
                jammer_pos = selected_pos + self.jammer_random_gen.uniform(-5, 5, 2)

                jammer_positions.append(jammer_pos)

                if len(user_positions) <= 0:
                    user_positions = self.user_positions.copy().tolist()

            self.jammer_positions = np.array(jammer_positions, np.float32)
        else:
            mean = self.jammer_random_gen.uniform(low=0, high=self.SIZE, size=2)
            cov = np.diag([self.SIZE / 4, self.SIZE / 4])

            self.jammer_positions = self.jammer_random_gen.multivariate_normal(
                mean=mean, cov=cov, size=self.numbOfJammers
            )

        np.clip(
            a=self.jammer_positions,
            a_min=0,
            a_max=self.SIZE,
            out=self.jammer_positions,
        )

    def initialize_attacker(self, seed: Optional[int] = None) -> None:
        """
        Initialize the attacker model based on the provided attackerModel attribute.

        Args:
            seed (Optional[int]): Seed for the attacker model's random generator.

        """
        if isinstance(self.attackerModel, (list, tuple)):
            curr_attacker: Type[at.AttackerModel] = self.attacker_random_gen.choice(
                self.attackerModel
            )
        else:
            curr_attacker: Type[at.AttackerModel] = self.attackerModel

        self.attacker = curr_attacker(
            nodes=self.nodes,
            jammers=self.jammers,
            users=self.users,
            clusterSizes=self.clusterSizes,
        )
        self.attacker.set_seed(seed)

    # Create components instances
    def createComponents(self) -> None:
        """
        Create the components of the network.
        """
        self.createNodes()
        self.createJammers()
        self.createUsers()
        self.createUserClusters()

    def createNodes(self) -> None:
        """
        Create the MANET nodes. If `numbOfByzantineNodes > 0`, the last N nodes
        are created as `ByzantineNode` instances instead of regular `MANETNode`s.
        The attack-specific parameters (`drop_rate`, `capacity_inflation_factor`,
        `jamming_power_factor`, `pos_offset`) are taken from the environment's
        Byzantine configuration set via `set_byzantine_params()`.

        If `byzantine_attack_type` is a list, attack types are assigned round-robin
        to the Byzantine nodes (e.g. ["greyhole", "selective_jamming"] with 4 Byzantine
        nodes gives greyhole, selective_jamming, greyhole, selective_jamming).
        """
        n_total = len(self.node_positions)
        n_byz = self.numbOfByzantineNodes
        pos_offset = (
            np.array(self.byzantine_pos_offset, dtype=float)
            if self.byzantine_pos_offset is not None
            else None
        )
        # Normalize attack_type to a list for uniform indexing
        attack_types = (
            self.byzantine_attack_type
            if isinstance(self.byzantine_attack_type, list)
            else [self.byzantine_attack_type]
        )
        # If more attack types are listed than there are Byzantine nodes, every
        # Byzantine node receives the full list (simultaneous multi-attack).
        # Otherwise attack types are assigned round-robin across nodes.
        multi_attack = len(attack_types) > n_byz
        nodes = []
        byz_idx = 0
        for i, pos in enumerate(self.node_positions):
            if i >= n_total - n_byz:
                node_attack_type = attack_types if multi_attack else attack_types[byz_idx % len(attack_types)]
                obs_user_dir_offset = (
                    np.array(self.byzantine_obs_user_dir_offset, dtype=float)
                    if self.byzantine_obs_user_dir_offset is not None
                    else None
                )
                nodes.append(
                    ByzantineNode(
                        id=f"Node {i}",
                        pos=pos,
                        signalPower_mW=self.nodeSignalPow_mW,
                        attack_type=node_attack_type,
                        drop_rate=self.byzantine_drop_rate,
                        capacity_inflation_factor=self.byzantine_capacity_inflation_factor,
                        jamming_power_factor=self.byzantine_jamming_power_factor,
                        pos_offset=pos_offset,
                        obs_user_dir_offset=obs_user_dir_offset,
                        obs_capacity_factor=self.byzantine_obs_capacity_factor,
                        obs_demand_factor=self.byzantine_obs_demand_factor,
                        obs_interference_factor=self.byzantine_obs_interference_factor,
                        obs_noise_std=self.byzantine_obs_noise_std,
                        obs_capacity_deflate_factor=self.byzantine_obs_capacity_deflate_factor,
                        obs_demand_deflate_factor=self.byzantine_obs_demand_deflate_factor,
                        obs_replay_delay=self.byzantine_obs_replay_delay,
                    )
                )
                byz_idx += 1
            else:
                nodes.append(
                    MANETNode(
                        id=f"Node {i}",
                        pos=pos,
                        signalPower_mW=self.nodeSignalPow_mW,
                    )
                )
        self.nodes: np.ndarray[MANETNode] = np.array(nodes)

    def createJammers(self) -> None:
        """
        Create the jammers.
        """
        self.jammers: np.ndarray[Jammer] = np.array(
            [
                Jammer(
                    id=f"Jammer {i}",
                    pos=pos,
                    signalPower_mW=self.jammerSignalPow_mW,
                )
                for i, pos in enumerate(self.jammer_positions)
            ]
        )

    def createUsers(self) -> None:
        """
        Create the users.
        """
        self.users: np.ndarray[User] = np.array(
            [
                User(
                    id=f"User {i}",
                    pos=pos,
                    signalPower_mW=self.userSignalPow_mW,
                )
                for i, pos in enumerate(self.user_positions)
            ]
        )

    def createUserClusters(self) -> None:
        """
        Groups the users into clusters and creates dict to quickly map them to
        their cluster.
        """
        self.userClusters = ut.findUsersPerCluster(self.users, self.clusterSizes)

        self.user_to_cluster_idx = {
            user: cluster_idx
            for cluster_idx, cluster in enumerate(self.userClusters)
            for user in cluster
        }

    # * Move the components
    def moveNodes(self, action) -> None:
        """
        Move the nodes in the network.

        Args:
            action (np.ndarray): Actions to be taken by the nodes.
        """
        self.node_positions += self.node_step_size * action

        np.clip(
            a=self.node_positions,
            a_min=0,
            a_max=self.SIZE,
            out=self.node_positions,
        )

    def moveUsers(self) -> None:
        """
        Move users.
        """
        pass

    def moveJammers(self) -> None:
        """
        Move jammers.
        """
        if self.numbOfJammers > 0 and self.step_jammers_start_moving <= self.ep_step:
            if self.ep_step >= self.steps_till_jammers_active:
                self.attacker.activateJammers()

            moves = self.attacker.getMoves()
            self.jammer_positions += moves
            np.clip(
                a=self.jammer_positions,
                a_min=0,
                a_max=self.SIZE,
                out=self.jammer_positions,
            )

    # * Initialize Demand and Destination
    def initializeDemand(self) -> None:
        """
        Set constant user and offeredLoad for each episode.
        """
        constant_demand = self.MAX_OFFERED_LOAD
        for user in self.users:
            user: User
            user.setOfferedLoad(constant_demand)

    def initializeSendersAndReceivers(self) -> None:
        """
        Set the destinations randomly, ensuring:

        1. At least one user is transmitting
        2. Senders and receivers are disjoint sets
        3. Respect intra-cluster destination choice
        """
        # Use Bernoulli distribution to determine transmitting users
        transmitting_flags = self.user_random_gen.binomial(
            1, self.TRANSMITTER_PROB, self.numbOfUsers
        )

        # Ensure at least one user is transmitting
        if not transmitting_flags.any():  # Check if all flags are 0
            random_user = self.user_random_gen.integers(0, self.numbOfUsers)
            transmitting_flags[random_user] = 1

        for user, is_transmitting in zip(self.users, transmitting_flags):
            user: User
            if is_transmitting:
                possible_dests = self.filterForIntraDestChoice(user=user)
                user.dest = (
                    self.user_random_gen.choice(possible_dests)
                    if possible_dests.size > 0
                    else None
                )
            else:
                user.dest = None

    def filterForIntraDestChoice(
        self, user: "User", possible_destinations: np.ndarray = None
    ) -> np.ndarray:
        """
        Filters and returns users not in the same cluster as the given user.

        Args:
            user (User): The user for whom the filtering is done.
            possible_destinations (np.ndarray): array of possible destinations. If None, uses all users.

        Returns:
            np.ndarray: Array of users not in the same cluster.
        """
        if possible_destinations is None:
            possible_destinations = self.users

        user_cluster_idx = self.user_to_cluster_idx[user]
        allowed = np.array(
            [
                u
                for u in possible_destinations
                if self.user_to_cluster_idx[u] != user_cluster_idx
            ]
        )
        return allowed

    def simulateTrafficPartnerChoice(self) -> None:
        """
        Simulate the choice of communication partners for the users.
        """
        pass

    def simulateOfferedLoadChoice(self) -> None:
        """
        Simulate the offered load each user sends.
        """
        pass

    # * GUI stuff
    def start_gui(self) -> None:
        """
        Start the GUI for visualizing the environment.
        """
        gui = MANETGUI(self.gui_queue, self)
        gui.mainloop()
        self.render_mode = None
        self.gui_closed = True

    def stop_gui_thread(self) -> None:
        """
        Stop the GUI thread.
        """
        self.gui_thread.join()  # Wait for the GUI thread to finish
        self.gui_thread = None  # Reset the GUI thread attribute

    def copy(self) -> "Basic_MANETEnv":
        """
        Create a deep copy of the environment instance, excluding generator objects.

        Returns:
            Basic_MANETEnv: A deep copy of the environment instance.
        """
        new_instance = Basic_MANETEnv(
            self.numbOfNodes,
            self.numbOfJammers,
            self.numbOfUsers,
            self.render_mode,
            seed=self.seed,
            networkConnectedAtStart=self.networkConnectedAtStart,
            jammersSpawnNextToUsers=self.jammersSpawnNextToUsers,
            attackerModel=self.attackerModel,
        )

        # Manually copy attributes
        new_instance.__dict__.update(
            {
                k: (v if not hasattr(v, "__iter__") else list(v))
                for k, v in self.__dict__.items()
                if not callable(v) and not isinstance(v, type(self.copy))
            }
        )

        return new_instance

    def __del__(self) -> None:
        """
        Ensure the stop_gui_thread method is called when the environment is closed or deleted.
        """
        if self.render_mode == "plot":
            self.gui_queue.put(("close",))
            sleep(1)
            self.stop_gui_thread()

    # * Truncated condition collection
    def logLastThroughputsAndMovement(
        self,
        action: np.ndarray,
        throughput: float,
    ) -> None:
        """
        Log the last throughputs and movement distances.

        Args:
            action (np.ndarray): Actions taken by the nodes.
            throughput (float): The throughput achieved in the current step.
        """
        # Log throughputs
        self.lastThroughputs.append(throughput)
        if (
            len(self.lastThroughputs) > self.max_log_size
            and self.ep_step % self.check_interval == 0
        ):
            self.trim_log(self.lastThroughputs, self.max_log_size)

        # Log movement distances
        distance = np.sum(np.linalg.norm(action, axis=1))
        self.lastDistances.append(distance)
        if (
            len(self.lastDistances) > self.max_log_size
            and self.ep_step % self.check_interval == 0
        ):
            self.trim_log(self.lastDistances, self.max_log_size)

    def trim_log(self, log: list, max_size: int) -> None:
        """
        Trim the log to the maximum size.

        Args:
            log (list): The log to be trimmed.
            max_size (int): The maximum size of the log.
        """
        excess = len(log) - max_size
        if excess > 0:
            del log[:excess]

    def noThroughputChange(self) -> bool:
        """
        Check if there is no significant change in throughput.

        Returns:
            bool: True if the standard deviation of the last throughputs is below the threshold, False otherwise.
        """
        std_threshold = 2e5
        return np.std(self.lastThroughputs) <= std_threshold

    def noMovement(self) -> bool:
        """
        Check if there is no significant movement.

        Returns:
            bool: True if the standard deviation of the last distances is below the threshold, False otherwise.
        """
        threshold = 0.05 * self.node_step_size
        return np.mean(self.lastDistances) <= threshold

    # * Setters
    def setObservationSpace(self, observation_space) -> bool:
        """
        Set the observation space for the environment.

        Args:
            observation_space (gym.spaces.Space): The observation space to be set.
        """
        if isinstance(observation_space, gym.spaces.Space):
            self.observation_space = observation_space

    def setRenderMode(self, render_mode) -> bool:
        """
        Set the render mode for the environment.

        Args:
            render_mode (str): The render mode to be set. Use "plot" to get the GUI, else use None.
        """
        if render_mode == "plot" or render_mode is None:
            self.render_mode = render_mode

    def setEpStep(self, ep_step: int) -> bool:
        """
        Set the current step in the episode.

        Args:
            ep_step (int): The current step in the episode.
        """
        if isinstance(ep_step, int) and ep_step <= self.ep_length:
            self.ep_step = ep_step

    def setNumbOfNodes(self, numbOfNodes) -> bool:
        """
        Set the number of MANET nodes.

        Args:
            numbOfNodes (int): The number of MANET nodes.
        """
        self.nodes = None
        self.node_positions = None

        self.numbOfNodes = numbOfNodes

    def setNumbOfUsers(self, numbOfUsers) -> bool:
        """
        Set the number of users.

        Args:
            numbOfUsers (int): The number of users.
        """
        self.users = None
        self.user_positions = None
        self.user_to_idx = None

        self.numbOfUsers = numbOfUsers

    def setNumbOfJammers(self, numbOfJammers) -> bool:
        """
        Set the number of jammers.

        Args:
            numbOfJammers (int): The number of jammers.
        """
        self.jammers = None
        self.jammer_positions = None

        self.numbOfJammers = numbOfJammers

    def setNodePositions(self, node_positions: np.ndarray) -> bool:
        """
        Set the positions of the nodes.

        Args:
            node_positions (np.ndarray): Array of node positions.
        """
        if node_positions.shape[0] == self.numbOfNodes:
            self.node_positions = node_positions
            self.createNodes()

    def setJammerPositions(self, jammer_positions: np.ndarray) -> bool:
        """
        Set the positions of the jammers.

        Args:
            jammer_positions (np.ndarray): Array of jammer positions.
        """
        if jammer_positions.shape[0] == self.numbOfJammers:
            self.jammer_positions = jammer_positions
            self.createJammers()

    def setUserPositions(self, user_positions: np.ndarray) -> bool:
        """
        Set the positions of the users.

        Args:
            user_positions (np.ndarray): Array of user positions.
        """
        if user_positions.shape[0] == self.numbOfUsers:
            self.user_positions = user_positions
            self.createUsers()

    def setClusterSizes(self, clusterSizes) -> bool:
        """
        Set the sizes of the clusters.

        Args:
            clusterSizes (np.ndarray): Array of cluster sizes.
        """
        if np.sum(clusterSizes) == self.numbOfUsers and np.all(
            isinstance(size, int) for size in clusterSizes
        ):
            self.clusterSizes = clusterSizes


class Static_MANETEnv(Basic_MANETEnv):
    """
    In the Static_MANETEnv Users are completely static. There, users also choose their
    communication partner for a whole episode at the start of it and have a constant offered load.

    Attributes:
        metadata (dict): Metadata for the environment.
        render_mode (str): Mode for rendering the environment.

        action_space (gym.spaces.Box): Action space for the environment.
        observation_space (gym.spaces.Box): Observation space for the environment.

        SIZE (int): Size of the playground.
        numbOfNodes (int): Number of MANET nodesduring the current episode.
        numbOfJammers (int): Number of jammers during the current episode.
        numbOfJammersRange (Union[int, Tuple[int,int]]): Can be either an integer for a constant number of jammers, or an interval, from which for each episode the number of jammers in that episode will be uniformly sampled.
        numbOfUsers (int): Number of users during the current episode.
        numbOfUsersRange (Union[int, Tuple[int,int]]): Can be either an integer for a constant number of users, or am interval, from which for each episode the number of users in that episode will be uniformly sampled.
        nodeSignalPow_mW (int): Signal power of the nodes in mW.
        userSignalPow_mW (int): Signal power of the users in mW.
        jammerSignalPow_mW (int): Signal power of the jammers in mW.

        ep_length (float): Length of an episode.
        ep_step (int): Current step in the episode.
        seed (int): Seed for random number generation.

        networkConnectedAtStart (bool): Whether the network is connected at the start.
        jammersSpawnNextToUsers (bool): Whether jammers spawn next to users.
        steps_till_jammers_active (int): Steps until jammers become active.
        step_jammers_start_moving (int): Steps until the jammers are allowed to start moving.
        allow_early_ep_finish (bool): Whether to allow early episode finish.

        network (Network): The network object.
        routing_func (Callable): Routes the offered loads given the network graph and a set of source, sink and demand tuples.
        ep_network_max_throughput (float): Maximum throughput of the network in the episode.

        numbOfByzantineNodes (int): Number of nodes that behave as Byzantine attackers.
        byzantine_drop_rate (float): Fraction of traffic each Byzantine node drops.
        byzantine_attack_type (str): The type of Byzantine attack (e.g. "greyhole").

        node_positions (np.ndarray): Positions of the nodes.
        nodes (list): List of MANET nodes.
        node_step_size (int): Step size for node movement.
        lastAction (np.ndarray): The previous action executed

        jammer_positions (np.ndarray): Positions of the jammers.
        jammers (list): List of jammers.
        attackerModel (class): Model for the attackers for the current episode.
        attacker (AttackerModel): Attacker model object.

        user_positions (np.ndarray): Positions of the users.
        users (list): List of users.
        userClusters (list): List of user clusters.
        user_to_idx (dict): Mapping from user to index.
        user_to_cluster_idx (dict): Mapping from user to cluster index.
        clusterSizes (np.ndarray): Sizes of the clusters.

        max_log_size (int): Maximum size of the log.
        check_interval (int): Interval for checking conditions.
        lastThroughputs (list): List of last throughputs.
        lastDistances (list): List of last distances.

        gui_thread (threading.Thread): Thread for the GUI.
        gui_closed (bool): Whether the GUI is closed.
        gui_queue (queue.Queue): Queue for the GUI.

        attacker_random_gen (np.random.Generator): Random generator controlling the smapling of the attacker models, if a list of attacker models was given.
        numb_of_random_gen (np.random.Generator): Random generator used if given an interval of users or jammers to sample uniformly from it.
        node_random_gen (np.random.Generator): Random generator controlling random distribution of the MANET nodes
        jammer_random_gen (np.random.Generator): Random generator controlling random distribution of the jammers
        user_random_gen (np.random.Generator): Random generator controlling random distribution of the users and the dynamic behaviour during the episode
        network_random_gen (np.random.Generator): Random generator controlling all randomicity involved in the network model.

        MAX_OFFERED_LOAD (int): the maximum offered load
        TRANSMITTER_PROB (float): probability for a user to be a source node or not
    """

    def __init__(
        self,
        numbOfNodes: int,
        numbOfJammers: int,
        numbOfUsers: int,
        render_mode: str = None,
        seed: int = None,
        networkConnectedAtStart: bool = False,
        jammersSpawnNextToUsers: bool = False,
        attackerModel: at.AttackerModel = at.Static_GreedyJammers,
        allow_early_ep_finish: bool = True,
    ) -> None:
        """
        Initialize a Static_MANETEnv environment.

        Args:
            numbOfNodes (int): Number of MANET nodes.
            numbOfJammers (int): Number of jammers.
            numbOfUsers (int): Number of users.
            render_mode (str, optional): Use "plot" to get the GUI, else use None. Default is None.
            seed (int, optional): Seed for random number generation. Default is None.
            networkConnectedAtStart (bool, optional): Whether the network is connected at the start. Default is False.
            jammersSpawnNextToUsers (bool, optional): Whether jammers spawn next to users. Default is False.
            attackerModel (at.AttackerModel, optional): Model for the attackers. Default is at.Static_GreedyJammers.
            allow_early_ep_finish (bool, optional): Whether to allow early episode finish. Default is True.
        """
        super().__init__(
            numbOfNodes,
            numbOfJammers,
            numbOfUsers,
            render_mode,
            seed,
            networkConnectedAtStart,
            jammersSpawnNextToUsers,
            attackerModel,
            allow_early_ep_finish,
        )

        self.ep_network_max_throughput = 0

    # * Override
    def reset(self, seed=None, options=None) -> tuple:
        """
        Reset the environment to an initial state.

        Args:
            seed (int, optional): Seed for random number generation. Defaults to None.
            options (dict, optional): Additional options for resetting the environment. Defaults to None.

        Returns:
            observation (np.ndarray): The initial observation of the environment.
            info (dict): Additional information about the environment.
        """
        # Call reset from Basic_MANETEnv
        observation, info = super().reset(seed=seed, options=options)

        self.ep_network_max_throughput = self.network.getMaxThroughput()

        return observation, info

    # * Override
    def done(self) -> bool:
        """
        Check if the episode is done.

        Returns:
            bool: True if the episode is done, False otherwise.
        """
        if self.allow_early_ep_finish:
            if self.steps_till_jammers_active < self.ep_step or self.numbOfJammers == 0:
                epsilon = 1e-6
                throughput = self.network.getThroughput()
                max_throughput_achieved = throughput >= (
                    self.ep_network_max_throughput - epsilon
                )

                if max_throughput_achieved:
                    return True

        return False

    def copy(self) -> "Static_MANETEnv":
        """
        Create a deep copy of the environment instance, excluding generator objects.

        Returns:
            Static_MANETEnv: A deep copy of the environment instance.
        """
        new_instance = Static_MANETEnv(
            self.numbOfNodes,
            self.numbOfJammers,
            self.numbOfUsers,
            self.render_mode,
            seed=self.seed,
            networkConnectedAtStart=self.networkConnectedAtStart,
            jammersSpawnNextToUsers=self.jammersSpawnNextToUsers,
            attackerModel=self.attackerModel,
            allow_early_ep_finish=self.allow_early_ep_finish,
        )

        # Manually copy attributes
        new_instance.__dict__.update(
            {
                k: (v if not hasattr(v, "__iter__") else list(v))
                for k, v in self.__dict__.items()
                if not callable(v) and not isinstance(v, type(self.copy))
            }
        )

        return new_instance


class Less_Static_MANETEnv(Basic_MANETEnv):
    """
    In the Less_Static_MANETEnv Users start moving slowly and in clusters (according to the reference point group mobility model).
    The sending behaviour stays constant.

    Attributes:
        metadata (dict): Metadata for the environment.
        render_mode (str): Mode for rendering the environment.

        action_space (gym.spaces.Box): Action space for the environment.
        observation_space (gym.spaces.Box): Observation space for the environment.

        SIZE (int): Size of the playground.
        numbOfNodes (int): Number of MANET nodesduring the current episode.
        numbOfJammers (int): Number of jammers during the current episode.
        numbOfJammersRange (Union[int, Tuple[int,int]]): Can be either an integer for a constant number of jammers, or an interval, from which for each episode the number of jammers in that episode will be uniformly sampled.
        numbOfUsers (int): Number of users during the current episode.
        numbOfUsersRange (Union[int, Tuple[int,int]]): Can be either an integer for a constant number of users, or am interval, from which for each episode the number of users in that episode will be uniformly sampled.
        nodeSignalPow_mW (int): Signal power of the nodes in mW.
        userSignalPow_mW (int): Signal power of the users in mW.
        jammerSignalPow_mW (int): Signal power of the jammers in mW.

        ep_length (float): Length of an episode.
        ep_step (int): Current step in the episode.
        seed (int): Seed for random number generation.

        networkConnectedAtStart (bool): Whether the network is connected at the start.
        jammersSpawnNextToUsers (bool): Whether jammers spawn next to users.
        steps_till_jammers_active (int): Steps until jammers become active.
        step_jammers_start_moving (int): Steps until the jammers are allowed to start moving.
        allow_early_ep_finish (bool): Whether to allow early episode finish.

        network (Network): The network object.
        routing_func (Callable): Routes the offered loads given the network graph and a set of source, sink and demand tuples.

        numbOfByzantineNodes (int): Number of nodes that behave as Byzantine attackers.
        byzantine_drop_rate (float): Fraction of traffic each Byzantine node drops.
        byzantine_attack_type (str): The type of Byzantine attack (e.g. "greyhole").

        node_positions (np.ndarray): Positions of the nodes.
        nodes (list): List of MANET nodes.
        node_step_size (int): Step size for node movement.
        lastAction (np.ndarray): The previous action executed

        jammer_positions (np.ndarray): Positions of the jammers.
        jammers (list): List of jammers.
        attackerModel (class): Model for the attackers for the current episode.
        attacker (AttackerModel): Attacker model object.

        user_positions (np.ndarray): Positions of the users.
        users (list): List of users.
        userClusters (list): List of user clusters.
        user_to_idx (dict): Mapping from user to index.
        user_to_cluster_idx (dict): Mapping from user to cluster index.
        clusterSizes (np.ndarray): Sizes of the clusters.
        user_movementGen (generator): Generator for user movement.

        max_log_size (int): Maximum size of the log.
        check_interval (int): Interval for checking conditions.
        lastThroughputs (list): List of last throughputs.
        lastDistances (list): List of last distances.

        gui_thread (threading.Thread): Thread for the GUI.
        gui_closed (bool): Whether the GUI is closed.
        gui_queue (queue.Queue): Queue for the GUI.

        attacker_random_gen (np.random.Generator): Random generator controlling the smapling of the attacker models, if a list of attacker models was given.
        numb_of_random_gen (np.random.Generator): Random generator used if given an interval of users or jammers to sample uniformly from it.
        node_random_gen (np.random.Generator): Random generator controlling random distribution of the MANET nodes
        jammer_random_gen (np.random.Generator): Random generator controlling random distribution of the jammers
        user_random_gen (np.random.Generator): Random generator controlling random distribution of the users and the dynamic behaviour during the episode
        network_random_gen (np.random.Generator): Random generator controlling all randomicity involved in the network model.

        MAX_OFFERED_LOAD (int): the maximum offered load
        TRANSMITTER_PROB (float): probability for a user to be a source node or not
    """

    def __init__(
        self,
        numbOfNodes: int,
        numbOfJammers: int,
        numbOfUsers: int,
        render_mode: str = None,
        seed: int = None,
        networkConnectedAtStart: bool = False,
        jammersSpawnNextToUsers: bool = False,
        attackerModel: at.AttackerModel = at.Static_GreedyJammers,
        allow_early_ep_finish: bool = False,
    ) -> None:
        """
        Initialize a Less_Static_MANETEnv environment.

        Args:
            numbOfNodes (int): Number of MANET nodes.
            numbOfJammers (int): Number of jammers.
            numbOfUsers (int): Number of users.
            render_mode (str, optional): Use "plot" to get the GUI, else use None. Default is None.
            seed (int, optional): Seed for random number generation. Default is None.
            networkConnectedAtStart (bool, optional): Whether the network is connected at the start. Default is False.
            jammersSpawnNextToUsers (bool, optional): Whether jammers spawn next to users. Default is False.
            attackerModel (at.AttackerModel, optional): Model for the attackers. Default is at.Static_GreedyJammers.
            allow_early_ep_finish (bool, optional): Whether to allow early episode finish. Default is True.
        """
        super().__init__(
            numbOfNodes,
            numbOfJammers,
            numbOfUsers,
            render_mode,
            seed,
            networkConnectedAtStart,
            jammersSpawnNextToUsers,
            attackerModel,
            allow_early_ep_finish,
        )

        # User movement generator
        self.user_movementGen = None

    # * Override to initialize movement generator
    def distributeUsersOverClusters(self, clusterCenters) -> np.ndarray:
        """
        Distribute the users on the playground.

        Args:
            clusterCenters (np.ndarray): Array of cluster center positions.

        Returns:
            np.ndarray: Array of user positions.
        """
        user_positions = []

        cov = np.diag([self.SIZE / 10, self.SIZE / 10])
        for i, center in enumerate(clusterCenters):
            mean = center
            cluster_user_positions = self.user_random_gen.multivariate_normal(
                mean=mean, cov=cov, size=self.clusterSizes[i]
            )
            cluster_user_positions = np.clip(cluster_user_positions, 0, self.SIZE)
            user_positions.extend(cluster_user_positions)

        self.user_movementGen = self.reference_point_group(
            self.clusterSizes,
            user_positions,
            clusterCenters,
            velocity=(0, 0.1),
        )

        return np.array(user_positions, dtype=np.float32)

    # * Move Users
    def moveUsers(self) -> None:
        """
        Move the users according to the reference_point_group mobility model.
        """
        nextPositions = next(self.user_movementGen)
        np.clip(a=nextPositions, a_min=0, a_max=self.SIZE, out=self.user_positions)

    def reference_point_group(
        self,
        clusterSizes,
        user_starting_pos,
        cluster_center_pos,
        velocity=(0.0, 0.1),
        aggregation=0.2,
    ) -> np.ndarray:  # type: ignore
        """
        Modified copy of:

        Reference Point Group Mobility model, discussed in the following paper:

        Xiaoyan Hong, Mario Gerla, Guangyu Pei, and Ching-Chuan Chiang. 1999.
        A group mobility model for ad hoc wireless networks. In Proceedings of the
        2nd ACM international workshop on Modeling, analysis and simulation of
        wireless and mobile systems (MSWiM '99). ACM, New York, NY, USA, 53-60.

        In this implementation, group trajectories follow a random direction model,
        while nodes follow a random walk around the group center.
        The parameter 'aggregation' controls how close the nodes are to the group center.

        Args:
            clusterSizes (list): List of integers, the number of nodes in each group.
            user_starting_pos (np.ndarray): Array of starting positions of the users.
            cluster_center_pos (np.ndarray): Array of cluster center positions.
            velocity (tuple, optional): Tuple of doubles, the minimum and maximum values for group velocity. Default is (0.0, 0.1).
            aggregation (float, optional): Parameter (between 0 and 1) used to aggregate the nodes in the group. Default is 0.2.

        Yields:
            np.ndarray: Array of new user positions.
        """
        user_pos = np.array(user_starting_pos, dtype=np.float32)
        cluster_center_pos = np.array(cluster_center_pos, dtype=np.float32)

        # define a Uniform Distribution
        U = lambda MIN, MAX, SAMPLES: self.user_random_gen.uniform(
            MIN, MAX, SAMPLES.shape
        )

        NODES = np.arange(self.numbOfUsers)

        groups = []
        prev = 0
        for n in clusterSizes:
            groups.append(np.arange(prev, n + prev))
            prev += n

        g_ref = np.empty(sum(clusterSizes), dtype=np.int32)
        for i, g in enumerate(groups):
            for n in g:
                g_ref[n] = i

        FL_MAX = max((self.SIZE, self.SIZE))
        MIN_V, MAX_V = velocity
        FL_DISTR = lambda SAMPLES: U(0, FL_MAX, SAMPLES)
        VELOCITY_DISTR = lambda FD: U(MIN_V, MAX_V, FD)

        MAX_X, MAX_Y = (self.SIZE, self.SIZE)
        x = user_pos[:, 0]
        y = user_pos[:, 1]
        velocity = 1.0
        theta = U(0, 2 * np.pi, NODES)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        GROUPS = np.arange(len(groups))
        g_x = cluster_center_pos[:, 0]
        g_y = cluster_center_pos[:, 1]
        g_fl = FL_DISTR(GROUPS)
        g_velocity = VELOCITY_DISTR(g_fl)
        g_theta = U(0, 2 * np.pi, GROUPS)
        g_costheta = np.cos(g_theta)
        g_sintheta = np.sin(g_theta)

        while True:

            x = x + velocity * costheta
            y = y + velocity * sintheta

            g_x = g_x + g_velocity * g_costheta
            g_y = g_y + g_velocity * g_sintheta

            for i, g in enumerate(groups):

                # step to group direction + step to group center
                x_g = x[g]
                y_g = y[g]
                c_theta = np.arctan2(g_y[i] - y_g, g_x[i] - x_g)

                x[g] = (
                    x_g + g_velocity[i] * g_costheta[i] + aggregation * np.cos(c_theta)
                )
                y[g] = (
                    y_g + g_velocity[i] * g_sintheta[i] + aggregation * np.sin(c_theta)
                )

            # node and group bounces on the margins
            b = np.where(x < 0)[0]
            if b.size > 0:
                x[b] = -x[b]
                costheta[b] = -costheta[b]
                g_idx = np.unique(g_ref[b])
                g_costheta[g_idx] = -g_costheta[g_idx]
            b = np.where(x > MAX_X)[0]
            if b.size > 0:
                x[b] = 2 * MAX_X - x[b]
                costheta[b] = -costheta[b]
                g_idx = np.unique(g_ref[b])
                g_costheta[g_idx] = -g_costheta[g_idx]
            b = np.where(y < 0)[0]
            if b.size > 0:
                y[b] = -y[b]
                sintheta[b] = -sintheta[b]
                g_idx = np.unique(g_ref[b])
                g_sintheta[g_idx] = -g_sintheta[g_idx]
            b = np.where(y > MAX_Y)[0]
            if b.size > 0:
                y[b] = 2 * MAX_Y - y[b]
                sintheta[b] = -sintheta[b]
                g_idx = np.unique(g_ref[b])
                g_sintheta[g_idx] = -g_sintheta[g_idx]

            # update info for nodes
            theta = U(0, 2 * np.pi, NODES)
            costheta = np.cos(theta)
            sintheta = np.sin(theta)

            # update info for arrived groups
            g_fl = g_fl - g_velocity
            g_arrived = np.where(np.logical_and(g_velocity > 0.0, g_fl <= 0.0))[0]

            if g_arrived.size > 0:
                g_theta = U(0, 2 * np.pi, g_arrived)
                g_costheta[g_arrived] = np.cos(g_theta)
                g_sintheta[g_arrived] = np.sin(g_theta)
                g_fl[g_arrived] = FL_DISTR(g_arrived)
                g_velocity[g_arrived] = VELOCITY_DISTR(g_fl[g_arrived])

            yield np.dstack((x, y))[0]

    def copy(self) -> "Less_Static_MANETEnv":
        """
        Create a deep copy of the environment instance, excluding generator objects.

        Returns:
            Less_Static_MANETEnv: A deep copy of the environment instance.
        """
        new_instance = Less_Static_MANETEnv(
            self.numbOfNodes,
            self.numbOfJammers,
            self.numbOfUsers,
            self.render_mode,
            seed=self.seed,
            networkConnectedAtStart=self.networkConnectedAtStart,
            jammersSpawnNextToUsers=self.jammersSpawnNextToUsers,
            attackerModel=self.attackerModel,
        )

        # Manually copy attributes
        new_instance.__dict__.update(
            {
                k: (v if not hasattr(v, "__iter__") else list(v))
                for k, v in self.__dict__.items()
                if not callable(v) and not isinstance(v, type(self.copy))
            }
        )

        return new_instance


class Dynamic_MANETEnv(Less_Static_MANETEnv):
    """
    In the Dyanmic_MANETEnv the MANETEnv becomes fully dynamic, with not only dynamic and
    moving user but further modeled traffic. Here, the destinations change randomly according
    to a simplified Markov Chain and the offered load of the users changes accoring
    to and actual Markov Chain.

    Attributes:
        metadata (dict): Metadata for the environment.
        render_mode (str): Mode for rendering the environment.

        action_space (gym.spaces.Box): Action space for the environment.
        observation_space (gym.spaces.Box): Observation space for the environment.

        SIZE (int): Size of the playground.
        numbOfNodes (int): Number of MANET nodesduring the current episode.
        numbOfJammers (int): Number of jammers during the current episode.
        numbOfJammersRange (Union[int, Tuple[int,int]]): Can be either an integer for a constant number of jammers, or an interval, from which for each episode the number of jammers in that episode will be uniformly sampled.
        numbOfUsers (int): Number of users during the current episode.
        numbOfUsersRange (Union[int, Tuple[int,int]]): Can be either an integer for a constant number of users, or am interval, from which for each episode the number of users in that episode will be uniformly sampled.
        nodeSignalPow_mW (int): Signal power of the nodes in mW.
        userSignalPow_mW (int): Signal power of the users in mW.
        jammerSignalPow_mW (int): Signal power of the jammers in mW.

        ep_length (float): Length of an episode.
        ep_step (int): Current step in the episode.
        seed (int): Seed for random number generation.

        networkConnectedAtStart (bool): Whether the network is connected at the start.
        jammersSpawnNextToUsers (bool): Whether jammers spawn next to users.
        steps_till_jammers_active (int): Steps until jammers become active.
        step_jammers_start_moving (int): Steps until the jammers are allowed to start moving.
        allow_early_ep_finish (bool): Whether to allow early episode finish.

        network (Network): The network object.
        routing_func (Callable): Routes the offered loads given the network graph and a set of source, sink and demand tuples.

        numbOfByzantineNodes (int): Number of nodes that behave as Byzantine attackers.
        byzantine_drop_rate (float): Fraction of traffic each Byzantine node drops.
        byzantine_attack_type (str): The type of Byzantine attack (e.g. "greyhole").

        node_positions (np.ndarray): Positions of the nodes.
        nodes (list): List of MANET nodes.
        node_step_size (int): Step size for node movement.
        lastAction (np.ndarray): The previous action executed

        jammer_positions (np.ndarray): Positions of the jammers.
        jammers (list): List of jammers.
        attackerModel (class): Model for the attackers for the current episode.
        attacker (AttackerModel): Attacker model object.

        user_positions (np.ndarray): Positions of the users.
        users (list): List of users.
        userClusters (list): List of user clusters.
        user_to_idx (dict): Mapping from user to index.
        user_to_cluster_idx (dict): Mapping from user to cluster index.
        clusterSizes (np.ndarray): Sizes of the clusters.
        user_movementGen (generator): Generator for user movement.

        max_log_size (int): Maximum size of the log.
        check_interval (int): Interval for checking conditions.
        lastThroughputs (list): List of last throughputs.
        lastDistances (list): List of last distances.

        gui_thread (threading.Thread): Thread for the GUI.
        gui_closed (bool): Whether the GUI is closed.
        gui_queue (queue.Queue): Queue for the GUI.

        attacker_random_gen (np.random.Generator): Random generator controlling the smapling of the attacker models, if a list of attacker models was given.
        numb_of_random_gen (np.random.Generator): Random generator used if given an interval of users or jammers to sample uniformly from it.
        node_random_gen (np.random.Generator): Random generator controlling random distribution of the MANET nodes
        jammer_random_gen (np.random.Generator): Random generator controlling random distribution of the jammers
        user_random_gen (np.random.Generator): Random generator controlling random distribution of the users and the dynamic behaviour during the episode
        network_random_gen (np.random.Generator): Random generator controlling all randomicity involved in the network model.

        MAX_OFFERED_LOAD (int): the maximum offered load
        TRANSMITTER_PROB (float): probability for a user to be a source node or not

        destProbabilities (np.ndarray): Probability matrix for user destinations.
        offLoadProbabilities (np.ndarray): Probability matrix for offered load.
        user_sending_data_type (np.ndarray): Array indicating the type of data each user is sending.
        transmitterProbabilities (np.ndarray): Markov transission matrix
        user_transmitter_flags (np.ndarray): Array indicating whether a user is source node or not
    """

    def __init__(
        self,
        numbOfNodes: int,
        numbOfJammers: int,
        numbOfUsers: int,
        render_mode: str = None,
        seed: int = None,
        networkConnectedAtStart: bool = False,
        jammersSpawnNextToUsers: bool = False,
        attackerModel: at.AttackerModel = at.Static_GreedyJammers,
        allow_early_ep_finish: bool = False,
    ) -> None:
        """
        Initialize a Dynamic_MANETEnv environment.

        Args:
            numbOfNodes (int): Number of MANET nodes.
            numbOfJammers (int): Number of jammers.
            numbOfUsers (int): Number of users.
            render_mode (str, optional): Use "plot" to get the GUI, else use None. Default is None.
            seed (int, optional): Seed for random number generation. Default is None.
            networkConnectedAtStart (bool, optional): Whether the network is connected at the start. Default is False.
            jammersSpawnNextToUsers (bool, optional): Whether jammers spawn next to users. Default is False.
            attackerModel (at.AttackerModel, optional): Model for the attackers. Default is at.Static_GreedyJammers.
            allow_early_ep_finish (bool, optional): Whether to allow early episode finish. Default is True.
        """
        # Initialize user destination choice probability matrix
        self.destProbabilities = None
        self.offLoadProbabilities = self.createOfferedLoadProbMatrix()
        self.user_sending_data_type = None

        self.transmitterProbabilities = self.createTransmitterProbMatrix()
        self.user_transmitter_flags = None

        super().__init__(
            numbOfNodes,
            numbOfJammers,
            numbOfUsers,
            render_mode,
            seed,
            networkConnectedAtStart,
            jammersSpawnNextToUsers,
            attackerModel,
            allow_early_ep_finish,
        )

        self.MAX_OFFERED_LOAD: int = 500e6

    # * Initialize Demand and Destination
    # * Override
    def initializeDemand(self) -> None:
        """
        Initialize probability matrix and sending data.
        """
        self.createOfferedLoadProbMatrix()
        self.initializeSendingData()
        self.simulateOfferedLoadChoice()

    # * Override
    def initializeSendersAndReceivers(self) -> None:
        """
        Initialize probability matrix.
        """
        self.initializeTransmittingFlags()
        self.createDestProbMatrix()
        self.simulateTrafficPartnerChoice()

    # * Destination choice modelling
    def createDestProbMatrix(self) -> None:
        """
        Generates a probability matrix for the destination choice.

        The matrix is initialized with a skewed distribution using the parameters \\(\\alpha\\) and \\(\\beta\\).
        Each row corresponds to a user and holds the probabilities of sending a message to another user in the columns.
        The probability is set to nearly zero for sending to users in the same cluster and zero for sending to oneself.
        Each row of the matrix is normalized so that the sum of each row is 1.

        The steps involved are:

        1. Initialize a matrix with a skewed distribution using the Beta distribution with parameters \\(\\alpha\\) and \\(\\beta\\).
        2. Set the probability to nearly zero for sending to users in the same cluster.
        3. Set the probability to zero for sending to oneself.
        4. Normalize each row so that the sum of each row is 1.

        Returns:
            np.ndarray: The probability matrix with shape (numbOfUsers, numbOfUsers).
                Each element represents the probability of sending a message from one user to another.
        """
        alpha = 1
        beta = 4

        # Step 1: Initialize a matrix with a skewed distribution
        destProbabilities = self.user_random_gen.beta(
            alpha, beta, (self.numbOfUsers, self.numbOfUsers)
        )

        # Step 2: Set probability to nearly zero for sending to users in the same cluster
        for cluster_users in self.userClusters:
            cluster_indices = [self.user_to_idx[user] for user in cluster_users]
            destProbabilities[np.ix_(cluster_indices, cluster_indices)] = 1e-9

        # Step 3: Set probability to zero for sending to oneself
        np.fill_diagonal(destProbabilities, 0)

        # Step 4: Normalize each row so that the sum of each row is 1
        row_sums = destProbabilities.sum(axis=1, keepdims=True)
        self.destProbabilities = destProbabilities / row_sums

    def simulateTrafficPartnerChoice(self) -> None:
        """
        Simulating Communication Partner Choice

        This function makes users more likely to continue conversations with the same partner. It updates the probability matrix to favor resending and responding, then normalizes it to keep probabilities valid.

        How it Works:

        1. Each user picks a communication partner based on the current probability matrix.
        2. The matrix is updated to:
            - Increase the chance of sending to the same user again.
            - Encourage responses.
        3. The matrix is normalized to ensure each row sums to 1.

        Why?
        - Users tend to continue conversations rather than pick random new partners.
        - Keeps communication behavior dynamic while ensuring probabilities remain valid.
        """
        continue_sending_value = 3
        answer_value = 1

        # Simulate transmitting choices
        self.simulateTransmittingChoices()

        # Ensure at least one user is transmitting
        if not self.user_transmitter_flags.any():  # Check if all flags are 0
            random_user = self.user_random_gen.integers(0, self.numbOfUsers)
            self.user_transmitter_flags[random_user] = 1

        # Model destination choice
        chosen_destinations = []

        for user, is_transmitting in zip(self.users, self.user_transmitter_flags):
            user: User
            if is_transmitting:
                user_index = self.user_to_idx[user]
                dest_index = self.user_random_gen.choice(
                    self.numbOfUsers, p=self.destProbabilities[user_index]
                )
                dest = self.users[dest_index]
                user.dest = dest
                chosen_destinations.append((user_index, dest_index))
            else:
                user.dest = None

        for user_index, dest_index in chosen_destinations:
            # Update destination probabilities
            self.destProbabilities[dest_index][user_index] = answer_value
            self.destProbabilities[user_index][
                dest_index
            ] = continue_sending_value  # Increase probability for chosen destination

        # Normalize the probabilities
        row_sums = self.destProbabilities.sum(axis=1, keepdims=True)
        self.destProbabilities = self.destProbabilities / row_sums

    def simulateTransmittingChoices(self) -> None:
        """
        Uses MarkovDecision matrix for choosing whether a user should transmit/send its offered load to a dest
        or not depending on whether it is currently or not.
        """
        for i in range(self.numbOfUsers):
            is_transmitting = self.user_transmitter_flags[i]

            keep_transmitting = self.user_random_gen.choice(
                [0, 1], p=self.transmitterProbabilities[is_transmitting]
            )
            self.user_transmitter_flags[i] = keep_transmitting

    # Transmitter modelling
    def createTransmitterProbMatrix(self) -> np.ndarray:
        """Creates the MarkovDecision matrix for choosing whether a user should transmit/send its offered load to a dest
        or not depending on whether it is currently or not.

        Returns:
            np.ndarray: 2x2 probability matrix
        """
        transmitterProbMatrix = np.array([[0.99, 0.01], [0.01, 0.99]])

        return transmitterProbMatrix

    def initializeTransmittingFlags(self) -> None:
        """Creates an array with 1 whether a user does transmitt its offered load or 0 if not."""
        # Use Bernoulli distribution to determine transmitting users
        transmitting_flags = self.user_random_gen.binomial(
            1, self.TRANSMITTER_PROB, self.numbOfUsers
        )

        # Ensure at least one user is transmitting
        if not transmitting_flags.any():  # Check if all flags are 0
            random_user = self.user_random_gen.integers(0, self.numbOfUsers)
            transmitting_flags[random_user] = 1

        self.user_transmitter_flags = transmitting_flags

    # * Offered Load modelling
    def createOfferedLoadProbMatrix(self) -> np.ndarray:
        """
        Create a probability matrix for offered load.

        Returns:
            np.ndarray: The probability matrix for offered load.
        """
        offLoadProbabilities = np.array([[0.7, 0.3], [0.3, 0.7]])

        return offLoadProbabilities

    # * Override
    def initializeSendingData(self) -> None:
        """
        Set constant user and offered load for each episode.
        """
        prob_small_data = 0.7  # probability of sending small data -> 1 is 70%
        self.user_sending_data_type = self.user_random_gen.binomial(
            1, prob_small_data, self.numbOfUsers
        )

    def simulateOfferedLoadChoice(self) -> None:
        """
        Simulates the data type selection and load generation for each user in the system.

        This function performs two main operations for each user:

        1. Determines whether the user will send small or large data based on transition probabilities
        2. Generates the offered load amount using a truncated normal distribution

        Data types are represented as:

        - 0: Large data type
        - 1: Small data type

        The load is generated from a normal distribution with parameters depending on the data type:

        - Small data: \\( \\mathcal{N}(\\mu=16\\text{MB}, \\sigma^2=8\\text{MB}) \\)
        - Large data: \\( \\mathcal{N}(\\mu=300\\text{MB}, \\sigma^2=150\\text{MB}) \\)

        The generated load is clipped between 0 and MAX_OFFERED_LOAD (500MB) to ensure
        realistic values within system constraints.

        For each user, the data type transition follows a Markov process where:

        - Transition probabilities from current data type to new data type are defined in self.offLoadProbabilities[current_data_type]
        - These probabilities determine whether a user switches between small and large data types

        Returns:
            None: Updates self.user_sending_data_type and generates offered loads in place
        """
        small_data_mean = 16e6  # 2MB
        small_data_std = 8e6  # 1MB
        large_data_mean = 300e6  # 300Mb
        large_data_std = 150e6  # 150Mb

        low_bound = 0
        up_bound = self.MAX_OFFERED_LOAD  # 500Mb

        for i in range(self.numbOfUsers):
            data_type = self.user_sending_data_type[i]

            new_data_type = self.user_random_gen.choice(
                [0, 1], p=self.offLoadProbabilities[data_type]
            )
            self.user_sending_data_type[i] = new_data_type

            mean = small_data_mean if new_data_type == 1 else large_data_mean
            std = small_data_std if new_data_type == 1 else large_data_std
            offered_load = self.user_random_gen.normal(loc=mean, scale=std)
            offered_load = np.clip(offered_load, a_min=low_bound, a_max=up_bound)

    def copy(self) -> "Dynamic_MANETEnv":
        """
        Create a deep copy of the environment instance, excluding generator objects.

        Returns:
            Dynamic_MANETEnv: A deep copy of the environment instance.
        """
        new_instance = Dynamic_MANETEnv(
            self.numbOfNodes,
            self.numbOfJammers,
            self.numbOfUsers,
            self.render_mode,
            seed=self.seed,
            networkConnectedAtStart=self.networkConnectedAtStart,
            jammersSpawnNextToUsers=self.jammersSpawnNextToUsers,
            attackerModel=self.attackerModel,
        )

        # Manually copy attributes
        new_instance.__dict__.update(
            {
                k: (v if not hasattr(v, "__iter__") else list(v))
                for k, v in self.__dict__.items()
                if not callable(v) and not isinstance(v, type(self.copy))
            }
        )

        return new_instance


def resolve_numb_of(
    int_or_range: Union[int, Tuple[int, int]], rng: np.random.Generator
) -> int:
    """If an integer was given returns the integer, if a range was given returns a random integer within in that range.

    Args:
        int_or_range (Union[int, Tuple[int, int]]): An integer or a range e.g. (2,5)
        rng (np.random.Generator): Random generator to generate the random integer within the specified range

    Returns:
        int: If an integer was given returns the integer, if a range was given returns a random integer within in that range.
    """
    if isinstance(int_or_range, int):
        return int_or_range
    return rng.integers(*int_or_range)
