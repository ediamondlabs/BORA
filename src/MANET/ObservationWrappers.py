from typing import Callable
from copy import deepcopy

# Default dictionary
from collections import deque

# Standard stuff
import numpy as np

# Networkx to find graph nodes to have a path to specific node
import networkx as nx

# ObservationWrapper to customize the observation space
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium import spaces

from .Environments import (
    Basic_MANETEnv,
)
import MANET.Components as net_comps

"""
# ObservationWrappers.py

Observation Wrappers provide a neat and flexible way to define different observation spaces without needing to modify the environment class itself. This design enables efficient customization of observation spaces tailored to specific requirements.

The created observation wrappers account for the fact that while the full observation space—considering node positions, node signal received, interference, user positions, and other metrics—can be helpful, it is often computationally costly. Therefore, the wrappers allow selective inclusion of observations to streamline training. However, the observation space in StableBaselines3 must have a constant size. This requirement motivates the distinction between fixed-sized and variable-sized observation spaces, both of which comply with this constraint:

- **Fixed-sized observation spaces**: These always maintain the size of the full observation space. This allows flexibility in including or excluding variables without altering the observation space size. It is particularly useful for retraining with previously excluded variables.
  
- **Variable-sized observation spaces**: These reduce the observation space to the minimum required size, including only the selected observations. While more efficient, they do not allow for easy expansion without redefining the space.

Both fixed-sized and variable-sized spaces maintain constant dimensions in practice, adhering to the requirements of StableBaselines3.

### Invariable Versions

To handle symmetries in the observation space, invariable versions of these wrappers are introduced. Symmetry elimination ensures that the agent does not need to learn redundant states. For example, in a scenario where users $u_1$ and $u_2$ swap positions but exhibit identical signal power and behavior, the agent only needs to learn one of these states. The invariable versions sort observations based on positions, removing redundant symmetry.

## Observation Wrappers

Currently, the following observation wrappers are available:

- [FS_COW](#MANET.ObservationWrappers.FS_COW): Fixed-sized configurable observation wrapper.
- [VS_COW](#MANET.ObservationWrappers.VS_COW): Variable-sized configurable observation wrapper.
- [FS_iCOW](#MANET.ObservationWrappers.FS_iCOW): Invariable fixed-sized configurable observation wrapper.
- [VS_iCOW](#MANET.ObservationWrappers.VS_COW): Invariable variable-sized configurable observation wrapper.

The full observation space is detailed below, offering comprehensive options, including positions for nodes and users, and customizable selections for signals, interference, and actions. While this space is robust, it can be computationally expensive:

```python
{
    "nodes": {
        "position": True,
        "receivedSignal": False,
        "interference": True,
        "signalPower": True,
        "trafficHosted": False,
        "inCapacities": True,
        "outCapacities": True,
        "sensedSenderDemands": True,
        "sensedReceiverDemands": True,
        "inUserCount": True,
        "outUserCount": True,
        "inSenderCount": True,
        "outSenderCount": False,
        "inReceiverCount": False,
        "outReceiverCount": True,
        "closestUserDir": True,
        "closestUserPos": False,
        "closestSenderDir": True,
        "closestSenderPos": False,
        "closestReceiverDir": True,
        "closestReceiverPos": False,
        "closestUserInterference": True,
    },
    "users": {
        "position": False,
        "receivedSignal": False,
        "interference": False,
        "signalPower": False,
        "offeredLoad": False,
        "destination": False,
    },
    "capacity_matrix": False,
}
```
"""

DEFAULT_CONFIG = {
    "nodes": {
        "position": True,
        "receivedSignal": False,
        "interference": True,
        "signalPower": True,
        "trafficHosted": False,
        "inCapacities": True,
        "outCapacities": True,
        "sensedSenderDemands": True,
        "sensedReceiverDemands": True,
        "inUserCount": True,
        "outUserCount": True,
        "inSenderCount": True,
        "outSenderCount": False,
        "inReceiverCount": False,
        "outReceiverCount": True,
        "closestUserDir": True,
        "closestUserPos": False,
        "closestSenderDir": True,
        "closestSenderPos": False,
        "closestReceiverDir": True,
        "closestReceiverPos": False,
        "closestUserInterference": True,
        # Byzantine robustness signals (disabled by default to preserve existing model compatibility)
        "signal_consistency": False,
        "pos_consistency": False,
        "unexplained_interference": False,
        "trust": False,
    },
    "users": {
        "position": False,
        "receivedSignal": False,
        "interference": False,
        "signalPower": False,
        "offeredLoad": False,
        "destination": False,
    },
    "capacity_matrix": False,
}

NODE_KEYS = [
    "position",
    "receivedSignal",
    "interference",
    "signalPower",
    "trafficHosted",
    "inCapacities",
    "outCapacities",
    "sensedSenderDemands",
    "sensedReceiverDemands",
    "inUserCount",
    "outUserCount",
    "inSenderCount",
    "outSenderCount",
    "inReceiverCount",
    "outReceiverCount",
    "closestUserDir",
    "closestUserPos",
    "closestSenderDir",
    "closestSenderPos",
    "closestReceiverDir",
    "closestReceiverPos",
    "closestUserInterference",
    # Byzantine robustness signals
    "signal_consistency",
    "pos_consistency",
    "unexplained_interference",
    "trust",
]

USER_KEYS = [
    "position",
    "receivedSignal",
    "interference",
    "signalPower",
    "offeredLoad",
    "destination",
]

GENERAL_KEYS = ["capacity_matrix"]


class Base_ObservationWrapper:
    """The base of all observation wrappers. This one handles all around collecting and calculating observations, with parameters to set that allow to pad non-used observations to keep a fixed size or not for a variablze sized observation vector. However, this is only a blueprint and doesn't yet follow Gymnasium standards.

    Attributes:
        env (gym.Env): The environment to wrap.
        config (dict): Configuration dictionary for the wrapper, which indicates which observations to include.
        unwrapped_env (Basic_MANETEnv): The unwrapped environment.
        is_padded (bool): Whether to pad not included observations. Defaults to False.
        is_invariable (bool): Whether to order the observations such that they are invariable about which User is which. Defaults to False.

    """

    def __init__(
        self,
        env: gym.Env,
        config: dict = None,
    ) -> None:
        """Initialize the Base Observation Wrapper

        Args:
            env (gym.Env): Environment to wrap
            config (dict, optional): Configuration dictates which observations to include in the observation space. Defaults to None.
        """
        # Set padded and ivariable
        self.is_padded = False
        self.is_invariable = False

        # Func dicts
        self.NODE_OBSERVATION_FUNCTIONS = {}
        self.USER_OBSERVATION_FUNCTIONS = {}
        self.GENERAL_OBSERVATION_FUNCTIONS = {}

        # Set env attributes
        self.env: gym.Env = env
        self.unwrapped_env: Basic_MANETEnv = self.env.unwrapped

        self.nodes = None
        self.users = None
        self.node_positions = None
        self.node_to_idx = None
        self.user_positions = None
        self.user_to_idx = None
        self.numbOfNodes = self.unwrapped_env.numbOfNodes
        self._replay_buffers: dict = {}
        self.initializeAllComponents()

        # Ensure valid config
        if config is None:
            config = DEFAULT_CONFIG
        else:
            try:
                config = filter_config(config)
                config = extend_with_missing_keys(config)
            except Exception:
                print("Unable to resolve observation config. Must be very outdated")
                config = DEFAULT_CONFIG
        self.config = config

        # Set min max tuples for observation normalization
        self.initializeMinMaxTuples()
        self.initializeFuncDicts()

    def initializeFuncDicts(self) -> None:
        """Initialize the dicts that are iterated over in the observation gathering methods to
        use the correct functions to gather individual observations.
        """

        self.NODE_OBSERVATION_FUNCTIONS = {
            "position": self.get_node_position,
            "receivedSignal": self.get_node_signal_received,
            "interference": self.get_node_interference,
            "signalPower": self.get_node_signal_power,
            "trafficHosted": self.get_node_traffic_hosted,
            "inCapacities": self.get_node_in_capacity,
            "outCapacities": self.get_node_out_capacity,
            "sensedSenderDemands": self.get_node_sensed_sender_demands,
            "sensedReceiverDemands": self.get_node_sensed_receiver_demands,
            "inUserCount": self.get_node_user_in_count,
            "outUserCount": self.get_node_user_out_count,
            "inSenderCount": self.get_node_sender_in_count,
            "outSenderCount": self.get_node_sender_out_count,
            "inReceiverCount": self.get_node_receiver_in_count,
            "outReceiverCount": self.get_node_receiver_out_count,
            "closestUserDir": self.get_node_closest_user_dir,
            "closestUserPos": self.get_node_closest_user_position,
            "closestSenderDir": self.get_node_closest_sender_dir,
            "closestSenderPos": self.get_node_closest_sender_position,
            "closestReceiverDir": self.get_node_closest_receiver_dir,
            "closestReceiverPos": self.get_node_closest_receiver_position,
            "closestUserInterference": self.get_node_closest_user_interference,
            # Byzantine robustness signals
            "signal_consistency": self.get_node_signal_consistency,
            "pos_consistency": self.get_node_pos_consistency,
            "unexplained_interference": self.get_node_unexplained_interference,
            "trust": self.get_node_trust,
        }

        self.USER_OBSERVATION_FUNCTIONS = {
            "position": self.get_user_position,
            "receivedSignal": self.get_user_signal_received,
            "interference": self.get_user_interference,
            "destination": self.get_user_dest_indices,
            "signalPower": self.get_user_signal_power,
            "offeredLoad": self.get_user_offered_load,
        }

        self.GENERAL_OBSERVATION_FUNCTIONS = {
            "capacity_matrix": self.get_capacity_matrix,
        }

    def initializeMinMaxTuples(self) -> None:
        """
        Initialize the (min_value, max_value) tuples used to normalize the observations to fit into [-1,1]
        """
        # Position is normed by the size of the playground
        playground_size = self.unwrapped_env.SIZE
        self.posMinMaxTuple = (0, playground_size)

        # Signal and interference are normed by cutting max signal or interference received in practice arond -30 dBm
        self.signalMinMaxTuple = (-100, -20)  # dBm
        self.interferenceMinMaxTuple = self.signalMinMaxTuple

        # Signal power is normed by dividing it by 1W, assumed to be maximum value
        self.signalPowMinMaxTuple = (0, 1)  # W

        # Offered Load norm factor
        self.offeredLoadMinMaxTuple = (0, self.unwrapped_env.MAX_OFFERED_LOAD)
        self.offeredLoadSumMinMaxTuple = (
            0,
            self.unwrapped_env.numbOfUsers / 2 * self.unwrapped_env.MAX_OFFERED_LOAD,
        )

        # Capacity is normed by dividing it by the maximum possible capacity:
        max_signal_received = self.unwrapped_env.nodeSignalPow_mW
        min_interference_received = 0
        env_noise = net_comps.ENVIRONMENT_NOISE

        nodes: np.ndarray[net_comps.MANETNode] = self.unwrapped_env.nodes
        max_bandwidth = max([node.BANDWIDTH for node in nodes])

        max_capacity = net_comps.shannonTheorem(
            signalReceived_mW=max_signal_received,
            interferenceReceived_mW=min_interference_received,
            noiseReceived_mW=env_noise,
            BANDWIDTH=max_bandwidth,
        )
        self.capacityMinMaxTuple = (0, max_capacity)
        self.capacitySumMinMaxTuple = (0, 2 * max_capacity)

        # Traffic hosted normalization
        self.trafficHostedMinMaxTuple = (0, max_capacity)

        # Maximum imaginable numb of users
        self.numbOfUsersMinMaxTuple = (0, 100)

    def get_observation(self) -> np.ndarray:
        """
        Returns the observations, which here is a combination of the nodes' observations,
        the users' observations and the general observations, such as the capacity matrix

        Returns:
            (np.ndarray): The observations.
        """

        self.updateComponents()

        node_obs: np.ndarray = self.get_selected_node_observations()
        user_obs: np.ndarray = self.get_selected_user_observations()
        general_obs: list[np.ndarray] = self.get_selected_general_observations()

        # Flatten
        node_obs = node_obs.flatten()
        user_obs = user_obs.flatten()
        general_obs: np.ndarray = (
            np.concatenate([arr.flatten() for arr in general_obs])
            if general_obs != []
            else np.array([], dtype=np.float32)
        )

        # Concatenate
        observations = np.concatenate(
            (node_obs.flatten(), user_obs.flatten(), general_obs.flatten())
        )
        return observations

    def initializeAllComponents(self) -> None:
        """Ensure that self.nodes and self.users reference the correct objects."""
        self.nodes = self.unwrapped_env.nodes
        self.users = self.unwrapped_env.users
        self.node_positions = self.unwrapped_env.node_positions
        self.node_to_idx = self.unwrapped_env.node_to_idx
        self.user_positions = self.unwrapped_env.user_positions
        self.user_to_idx = self.unwrapped_env.user_to_idx

    def updateComponents(self) -> None:
        """Update components if necessary."""
        if self.hasEnvBeenReset():
            self.initializeAllComponents()

    def hasEnvBeenReset(self) -> bool:
        """Whether the underyling environmnet has been reset or not.

        Returns:
            bool: True if the underyling env has been reset, meaning a new episode has startet, else False.
        """

        return self.unwrapped_env.ep_step == 0

    def setPadded(self, is_padded: bool) -> None:
        """
        Sets the padded status of the object.

        Args:
            is_padded (bool): A boolean value indicating whether the object
                              should be marked as padded (True) or not (False).

        Returns:
            None
        """
        self.is_padded = is_padded

    def setInvariable(self, is_invariable: bool) -> None:
        """
        Sets the invariable status of the object.

        Args:
            is_invariable (bool): A flag indicating whether the object should be marked as invariable.
        """
        self.is_invariable = is_invariable

    # * Node obs matrix
    def get_selected_node_observations(self) -> np.ndarray:
        """Collects all node related observations, given the config, which specifies which observations to include.
        If the observatidons should be padded then instead of not including them a fixed value array of same shape is included
        in its place.

        Returns:
            np.ndarray: Stacked observations of shape(numbOfNodes, x) where x is the number of observations, however e.g. the positions since 2D add +2 to x.
        """

        observations = []
        node_config: dict = self.config["nodes"]
        for key in NODE_KEYS:
            use_obs = node_config.get(key, False)
            if use_obs or self.is_padded:
                obs = self.NODE_OBSERVATION_FUNCTIONS[key](
                    use_replacement_value=(not use_obs)
                )
                observations.append(obs)

        node_obs_matrix = (
            np.hstack(observations)
            if observations != []
            else np.array([], dtype=np.float32)
        )

        # Apply full-obs offset for obs_full Byzantine nodes (set by ByzantineAttackerEnv).
        if node_obs_matrix.ndim == 2:
            for i, node in enumerate(self.nodes):
                offset = getattr(node, "obs_full_offset", None)
                if offset is not None:
                    node_obs_matrix[i] += offset

        # Inject zero-mean Gaussian noise for obs_noise Byzantine nodes.
        # Each Byzantine node's row in the observation matrix is corrupted so
        # neighbouring nodes receive a falsified broadcast state.
        if node_obs_matrix.ndim == 2:
            for i, node in enumerate(self.nodes):
                std = getattr(node, "observed_obs_noise_std", 0.0)
                if std > 0.0:
                    node_obs_matrix[i] += np.random.normal(
                        0.0, std, node_obs_matrix[i].shape
                    )

        # Replay stale observations for obs_replay Byzantine nodes.
        # Buffer lives on the wrapper (not on node copies) so it persists across steps.
        if node_obs_matrix.ndim == 2:
            for i, node in enumerate(self.nodes):
                delay = getattr(node, "obs_replay_delay", 0)
                if delay > 0 and "obs_replay" in getattr(node, "attack_types", set()):
                    node_id = node.id
                    if node_id not in self._replay_buffers:
                        self._replay_buffers[node_id] = deque(maxlen=delay + 1)
                    buf = self._replay_buffers[node_id]
                    buf.append(node_obs_matrix[i].copy())
                    if len(buf) > delay:
                        node_obs_matrix[i] = buf[0]

        # Track per-node detection errors for obs_user_dir Byzantine nodes.
        # Stored on unwrapped_env so the evaluator can read it without a wrapper reference.
        # detection_error = ||offset||_2: what an oracle observer would measure as the lie.
        if node_obs_matrix.ndim == 2:
            detection_errors: dict = {}
            for node in self.nodes:
                offset = getattr(node, "observed_user_dir_offset", None)
                if offset is not None:
                    detection_errors[node.id] = float(np.linalg.norm(offset))
            self.unwrapped_env._detection_errors = detection_errors

        return node_obs_matrix

    # * User obs matrix
    def get_selected_user_observations(self) -> np.ndarray:
        """Collects all user related observations, given the config, which specifies which observations to include.
        If the observations should be padded then instead of not including them a fixed value array of same shape is included
        in its place. If the user' observations should be invariable, their observations are sorted by their positions. This
        helps avoid symmetries.

        Returns:
            np.ndarray: Stacked observations of shape(numbOfUsers, x) where x is the number of observations, however e.g. the positions since 2D add +2 to x.
        """

        observations = []
        user_config: dict = self.config["users"]
        for key in USER_KEYS:
            use_obs = user_config.get(key, False)
            if use_obs or self.is_padded:
                # We choose the destinations as positions instead of indices if the user observations are chosen as invariable.
                if key == "destination" and self.is_invariable:
                    obs = self.get_user_dest_position(
                        use_replacement_value=(not use_obs)
                    )
                else:
                    obs = self.USER_OBSERVATION_FUNCTIONS[key](
                        use_replacement_value=(not use_obs)
                    )

                observations.append(obs)
        observations = (
            np.hstack(observations)
            if observations != []
            else np.array([], dtype=np.float32)
        )

        # Sort the observations if to be invariable by the users' positions.
        if self.is_invariable and observations.size > 0:
            positions = self.USER_OBSERVATION_FUNCTIONS["position"](
                use_replacement_value=(not use_obs)
            )
            sorted_indices = np.lexsort((positions[:, 1], positions[:, 0]))
            observations = observations[sorted_indices, :]

        return observations

    # * Remaining obs matrix
    def get_selected_general_observations(self) -> list[np.ndarray]:
        """Collects all general observations, given the config, which specifies which observations to include, in a list.
        If the observations should be padded then instead of not including them a fixed value array of same shape is included
        in its place.

        Returns:
            list: Collected arrays of observations.
        """

        observations = []
        general_config: dict = self.config
        for key in GENERAL_KEYS:
            use_obs = general_config.get(key, False)
            if use_obs or self.is_padded:
                obs = self.GENERAL_OBSERVATION_FUNCTIONS[key](
                    use_replacement_value=(not use_obs)
                )
                observations.append(obs)

        return observations

    # * Node observations
    def get_node_position(self, use_replacement_value: bool) -> np.ndarray:
        """Get the node's position, which should already be normalized and transformed such
        that they each are inside the observation space and scaled properly over it. So, each observation should be in [-1,1].

        For position_spoofing Byzantine nodes the `observed_pos` property is used instead of
        the real position, so the RL agent receives a corrupted state while physics remain intact.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: observation or padded observation
        """
        if use_replacement_value:
            return get_padded_array((self.numbOfNodes, 2), 0)
        else:
            positions = np.array(
                [getattr(node, "observed_pos", node.pos) for node in self.nodes]
            )
            return normalize_and_transform(positions, self.posMinMaxTuple)

    def get_node_signal_received(self, use_replacement_value: bool) -> np.ndarray:
        """Get the node's signals received, which should already be normalized and transformed such
        that they each are inside the observation space and scaled properly over it. So, each observation should be in [-1,1].

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: observation or padded observation.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            signals = np.array([node.signalReceived_mW for node in self.nodes]).reshape(
                -1, 1
            )

            epsilon = 1e-9
            transform_function = lambda x: 10 * np.log10(x + epsilon)
            return normalize_and_transform(
                signals, self.signalMinMaxTuple, transform_function
            )

    def get_node_interference(self, use_replacement_value: bool) -> np.ndarray:
        """Get the node's interferences, which should already be normalized and transformed such
        that they each are inside the observation space and scaled properly over it. So, each observation should be in [-1,1].

        For obs_interference_lie Byzantine nodes the `observed_interferenceReceived_mW` property
        is used instead of the real value, so neighbours receive a falsified interference reading.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: observation or padded observation.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            interferences = np.array(
                [getattr(node, "observed_interferenceReceived_mW", node.interferenceReceived_mW)
                 for node in self.nodes]
            ).reshape(-1, 1)

            epsilon = 1e-9
            transform_function = lambda x: 10 * np.log10(x + epsilon)

            return normalize_and_transform(
                interferences,
                self.interferenceMinMaxTuple,
                transform_function,
            )

    def get_node_signal_power(self, use_replacement_value: bool) -> np.ndarray:
        """Get the node's signal powers, which should already be normalized and transformed such
        that they each are inside the observation space and scaled properly over it. So, each observation should be in [-1,1].

        For sinkhole Byzantine nodes the `observed_signal_power_mW` property is used, which
        returns an inflated value to attract relay positioning by the RL agent.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: observation or padded observation.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            signalpower = np.array(
                [getattr(node, "observed_signal_power_mW", node.signalPower_mW) for node in self.nodes]
            ).reshape(-1, 1)
            return normalize_and_transform(signalpower, self.signalPowMinMaxTuple)

    def get_node_traffic_hosted(self, use_replacement_value: bool) -> np.ndarray:
        """Get the node's traffic hosted, which should already be normalized and transformed such
        that they each are inside the observation space and scaled properly over it. So, each observation should be in [-1,1].

        For sinkhole Byzantine nodes the `observed_traffic_hosted` property is used, which
        returns an inflated value to attract relay positioning by the RL agent.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: observation or padded observation.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            traffichosted = np.array(
                [getattr(node, "observed_traffic_hosted", node.trafficHosted) for node in self.nodes]
            ).reshape(-1, 1)
            return normalize_and_transform(traffichosted, self.trafficHostedMinMaxTuple)

    def get_node_in_capacity(self, use_replacement_value: bool) -> np.ndarray:
        """For each node returns the sum of the capacities from all incoming edges.

        For obs_capacity Byzantine nodes the `observed_capacity_factor` property is applied,
        scaling the reported capacity to mislead neighbouring nodes about connectivity.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: Sum of the capacities from all incoming edges
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            capacity_matrix = self.unwrapped_env.network.getCapacityMatrix()
            in_capacities = np.sum(
                capacity_matrix[:, range(self.numbOfNodes)], axis=0
            ).reshape(self.numbOfNodes, 1).astype(np.float64)
            for i, node in enumerate(self.nodes):
                factor = getattr(node, "observed_capacity_factor", 1.0)
                if factor != 1.0:
                    in_capacities[i] *= factor
            return normalize_and_transform(in_capacities, self.capacitySumMinMaxTuple)

    def get_node_out_capacity(self, use_replacement_value: bool) -> np.ndarray:
        """For each node returns the sum of the capacities of all outgoing edges.

        For obs_capacity Byzantine nodes the `observed_capacity_factor` property is applied,
        scaling the reported capacity to mislead neighbouring nodes about connectivity.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: Sum of the capacities of all outgoing edges
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            capacity_matrix = self.unwrapped_env.network.getCapacityMatrix()
            out_capacities = np.sum(
                capacity_matrix[range(self.numbOfNodes), :], axis=1
            ).reshape(self.numbOfNodes, 1).astype(np.float64)
            for i, node in enumerate(self.nodes):
                factor = getattr(node, "observed_capacity_factor", 1.0)
                if factor != 1.0:
                    out_capacities[i] *= factor
            return normalize_and_transform(out_capacities, self.capacitySumMinMaxTuple)

    def get_node_sensed_sender_demands(self, use_replacement_value: bool) -> np.ndarray:
        """For each node returns the sum of demands of all users that have an edge to it and actually have a destination.

        For obs_demand Byzantine nodes the `observed_demand_factor` property is applied,
        inflating or deflating the reported demand to attract or repel neighbouring nodes.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: Sum of demands of all users that have an edge to a node for each node
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            # Get capacity matrix and user demands
            capacity_matrix = self.unwrapped_env.network.getCapacityMatrix()

            user_demand_vector = np.zeros((self.unwrapped_env.numbOfUsers, 1))
            for i, user in enumerate(self.users):
                user: net_comps.User
                user_demand_vector[i] += user.offeredLoad if user.hasDest() else 0

            # Initialize the demand array for each node (nodes only)
            node_sensed_demand = np.zeros(
                (self.numbOfNodes, 1)
            )  # Now explicitly (numNodes, 1)

            # Extract the submatrix where rows are users, columns are nodes
            user_rows = np.arange(
                self.numbOfNodes, self.numbOfNodes + self.unwrapped_env.numbOfUsers
            )  # User indices in matrix
            node_cols = np.arange(self.numbOfNodes)  # Node indices in matrix

            # Get the relevant part of the matrix (Users → Nodes)
            user_to_node_matrix = capacity_matrix[user_rows][
                :, node_cols
            ]  # Shape: (numUsers, numNodes)
            user_to_node_matrix = np.sign(user_to_node_matrix)

            # Matrix-vector multiplication to sum up the demands at each node, keeping the (numNodes, 1) shape
            node_sensed_demand = (user_to_node_matrix.T @ user_demand_vector).reshape(
                self.numbOfNodes, 1
            ).astype(np.float64)  # (numNodes, 1)

            for i, node in enumerate(self.nodes):
                factor = getattr(node, "observed_demand_factor", 1.0)
                if factor != 1.0:
                    node_sensed_demand[i] *= factor

            return normalize_and_transform(
                node_sensed_demand, self.offeredLoadSumMinMaxTuple
            )

    def get_node_sensed_receiver_demands(
        self, use_replacement_value: bool
    ) -> np.ndarray:
        """For each node returns the sum of demands that the users it has an edge to are supposed to receive. So, these are the destination users of the demand.

        For obs_demand Byzantine nodes the `observed_demand_factor` property is applied,
        inflating or deflating the reported demand to attract or repel neighbouring nodes.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: Sum of demands that the users it has an edge to are supposed to receive
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            # Get capacity matrix and receiver demands
            capacity_matrix = self.unwrapped_env.network.getCapacityMatrix()

            receiver_demands = np.zeros(
                (self.unwrapped_env.numbOfUsers, 1)
            )  # Preallocate array with the correct shape

            for user in self.users:
                user: net_comps.User
                if user.hasDest():
                    dest_idx = self.user_to_idx[user.dest]
                    receiver_demands[
                        dest_idx
                    ] += user.offeredLoad  # Update directly in the preallocated array

            # Extract the submatrix where rows are nodes, columns are users
            node_rows = np.arange(self.numbOfNodes)  # Node indices in matrix
            user_cols = np.arange(
                self.numbOfNodes, self.numbOfNodes + self.unwrapped_env.numbOfUsers
            )  # User indices in matrix

            # Get the relevant part of the matrix (Users → Nodes)
            node_to_user_matrix = capacity_matrix[node_rows][
                :, user_cols
            ]  # Shape: (numUsers, numNodes)
            node_to_user_matrix = np.sign(node_to_user_matrix)

            # Matrix-vector multiplication to sum up the demands at each node, keeping the (numNodes, 1) shape
            node_sensed_receiver_demand = (node_to_user_matrix @ receiver_demands).astype(np.float64)

            for i, node in enumerate(self.nodes):
                factor = getattr(node, "observed_demand_factor", 1.0)
                if factor != 1.0:
                    node_sensed_receiver_demand[i] *= factor

            return normalize_and_transform(
                node_sensed_receiver_demand, self.offeredLoadSumMinMaxTuple
            )

    def get_node_user_in_count(self, use_replacement_value: bool) -> np.ndarray:
        """For each node returns the number of user that have an edge to it.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: Number of user that have an edge to a node for each node
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            # Get capacity matrix from the environment
            capacity_matrix = self.unwrapped_env.network.getCapacityMatrix()

            # Define row and column indices for users and nodes
            user_rows = np.arange(
                self.numbOfNodes, self.numbOfNodes + self.unwrapped_env.numbOfUsers
            )  # User indices in matrix
            node_cols = np.arange(self.numbOfNodes)  # Node indices in matrix

            # Extract relevant submatrices
            user_to_node_matrix = capacity_matrix[user_rows][
                :, node_cols
            ]  # Shape: (numUsers, numNodes)

            # Use np.sign() + np.sum() for optimized counting
            num_users_to_node = np.sum(np.sign(user_to_node_matrix), axis=0).reshape(
                self.numbOfNodes, 1
            )

            return normalize_and_transform(
                num_users_to_node, self.numbOfUsersMinMaxTuple
            )

    def get_node_user_out_count(self, use_replacement_value: bool) -> np.ndarray:
        """For each node return the number of users it has an edge to.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: For each node the number of users to which it has an edge
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            capacity_matrix = self.unwrapped_env.network.getCapacityMatrix()
            # Define row and column indices for users and nodes
            node_rows = np.arange(self.numbOfNodes)  # Node indices in matrix
            user_cols = np.arange(
                self.numbOfNodes, self.numbOfNodes + self.unwrapped_env.numbOfUsers
            )  # User indices in matrix

            node_to_user_matrix = capacity_matrix[node_rows][
                :, user_cols
            ]  # Shape: (numNodes, numUsers)
            num_users_from_node = np.sum(np.sign(node_to_user_matrix), axis=1).reshape(
                self.numbOfNodes, 1
            )

            return normalize_and_transform(
                num_users_from_node, self.numbOfUsersMinMaxTuple
            )

    def get_node_sender_in_count(self, use_replacement_value: bool) -> np.ndarray:
        """For each node returns the count of incoming edges from users that send, in other words, have a destination.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: For each node the count of incoming edges from users that send, in other words, have a destination.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            # Get all sending user indices
            sending_users_indices = np.array(
                [self.user_to_idx[user] for user in self.users if user.hasDest()]
            )
            if sending_users_indices.size == 0:
                return np.zeros(shape=(self.numbOfNodes, 1))

            # Get capacity matrix from the environment
            capacity_matrix = self.unwrapped_env.network.getCapacityMatrix()

            # Define row and column indices for users and nodes
            user_rows = sending_users_indices + self.numbOfNodes
            node_cols = np.arange(self.numbOfNodes)  # Node indices in matrix

            # Extract relevant submatrices
            user_to_node_matrix = capacity_matrix[user_rows][
                :, node_cols
            ]  # Shape: (numUsers, numNodes)

            # Use np.sign() + np.sum() for optimized counting
            num_users_to_node = np.sum(np.sign(user_to_node_matrix), axis=0).reshape(
                self.numbOfNodes, 1
            )

            return normalize_and_transform(
                num_users_to_node, self.numbOfUsersMinMaxTuple
            )

    def get_node_sender_out_count(self, use_replacement_value: bool) -> np.ndarray:
        """For each node returns the count of outgoing edges to users that send, in other words, have a destination.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: For each node the count of outgoing edges to users that send, in other words, have a destination.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            capacity_matrix = self.unwrapped_env.network.getCapacityMatrix()

            # Get all sending user indices
            sending_users_indices = np.array(
                [self.user_to_idx[user] for user in self.users if user.hasDest()]
            )
            if sending_users_indices.size == 0:
                return np.zeros(shape=(self.numbOfNodes, 1))

            # Define row and column indices for users and nodes
            node_rows = np.arange(self.numbOfNodes)  # Node indices in matrix
            user_cols = sending_users_indices + self.numbOfNodes

            node_to_user_matrix = capacity_matrix[node_rows][
                :, user_cols
            ]  # Shape: (numNodes, numUsers)
            num_users_from_node = np.sum(np.sign(node_to_user_matrix), axis=1).reshape(
                self.numbOfNodes, 1
            )

            return normalize_and_transform(
                num_users_from_node, self.numbOfUsersMinMaxTuple
            )

    def get_node_receiver_in_count(self, use_replacement_value: bool) -> np.ndarray:
        """For each node returns the count of incoming edges from users that receive, in other words, users that are the destination of another user's offered load.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: For each node the count of incoming edges from users that receive, in other words, users that are the destination of another user's offered load.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            # Get all receiving user indices
            receiving_users_indices = np.array(
                [self.user_to_idx[user.dest] for user in self.users if user.hasDest()]
            )
            if receiving_users_indices.size == 0:
                return np.zeros(shape=(self.numbOfNodes, 1))
            # Get capacity matrix from the environment
            capacity_matrix = self.unwrapped_env.network.getCapacityMatrix()

            # Define row and column indices for users and nodes
            user_rows = receiving_users_indices + self.numbOfNodes
            node_cols = np.arange(self.numbOfNodes)  # Node indices in matrix

            # Extract relevant submatrices
            user_to_node_matrix = capacity_matrix[user_rows][
                :, node_cols
            ]  # Shape: (numUsers, numNodes)

            # Use np.sign() + np.sum() for optimized counting
            num_users_to_node = np.sum(np.sign(user_to_node_matrix), axis=0).reshape(
                self.numbOfNodes, 1
            )

            return normalize_and_transform(
                num_users_to_node, self.numbOfUsersMinMaxTuple
            )

    def get_node_receiver_out_count(self, use_replacement_value: bool) -> np.ndarray:
        """For each node returns the count of outgoin edges to users that receive, in other words, users that are the destination of another user's offered load.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: For each node the count of outgoin edges to users that receive, in other words, users that are the destination of another user's offered load.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            capacity_matrix = self.unwrapped_env.network.getCapacityMatrix()

            # Get all receiving user indices
            receiving_users_indices = np.array(
                [self.user_to_idx[user.dest] for user in self.users if user.hasDest()]
            )
            if receiving_users_indices.size == 0:
                return np.zeros(shape=(self.numbOfNodes, 1))

            # Define row and column indices for users and nodes
            node_rows = np.arange(self.numbOfNodes)  # Node indices in matrix
            user_cols = receiving_users_indices + self.numbOfNodes

            node_to_user_matrix = capacity_matrix[node_rows][
                :, user_cols
            ]  # Shape: (numNodes, numUsers)
            num_users_from_node = np.sum(np.sign(node_to_user_matrix), axis=1).reshape(
                self.numbOfNodes, 1
            )

            return normalize_and_transform(
                num_users_from_node, self.numbOfUsersMinMaxTuple
            )

    def get_node_closest_user_dir(self, use_replacement_value: bool) -> np.ndarray:
        """For each node returns the normed direction to the closest user.

        For obs_user_dir Byzantine nodes the `observed_user_dir_offset` property is applied,
        adding a fixed offset to the reported direction to mislead neighbours about user locations.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: For each node the normed direction to the closest user.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 2), pad_value=-1)
        else:
            # Get node and user positions directly as NumPy arrays
            node_positions = self.node_positions
            user_positions = self.user_positions

            # Compute pairwise differences using broadcasting
            # Shape: (N, U, 2)
            differences = (
                user_positions[np.newaxis, :, :] - node_positions[:, np.newaxis, :]
            )

            # Compute distances (norms) for all pairs
            distances = np.linalg.norm(differences, axis=2)  # Shape: (N, U)

            # Find the index of the closest user for each node
            closest_user_indices = np.argmin(distances, axis=1)  # Shape: (N,)

            # Extract directions to the closest users
            directions = differences[
                np.arange(self.numbOfNodes), closest_user_indices
            ]  # Shape: (N, 2)

            def transform_func(x: np.ndarray) -> np.ndarray:
                norms = np.linalg.norm(x, axis=1, keepdims=True)  # Shape: (N, 1)
                # Replace norms < 1 with 1
                adjusted_norms = (
                    np.maximum(norms, 1.0) + 1e-9
                )  # Small epsilon for numerical stability
                return (
                    x / adjusted_norms
                )  # Broadcasting divides each row by its adjusted norm

            # Apply transformation with default min_max_tuple=(-1, 1)
            normalized = normalize_and_transform(
                directions, transform_function=transform_func
            )
            # Offset is applied after normalisation so that [0.5, 0.0] is a
            # meaningful shift in the [-1, 1] direction space rather than a
            # sub-pixel nudge on a raw pixel-distance vector.
            for i, node in enumerate(self.nodes):
                offset = getattr(node, "observed_user_dir_offset", None)
                if offset is not None:
                    normalized[i] += offset
            return normalized

    def get_node_closest_sender_dir(self, use_replacement_value: bool) -> np.ndarray:
        """For each node returns the normed direction to the closest sending user, one that has a destination.

        For obs_user_dir Byzantine nodes the `observed_user_dir_offset` property is applied,
        adding a fixed offset to the reported direction to mislead neighbours about sender locations.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: Normed direction to the closest sending user, one that has a destination.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 2), pad_value=0)
        else:
            # Get node positions directly as NumPy array
            node_positions = self.node_positions

            # Identify sender users (those with a destination)
            sending_users_indices = np.array(
                [self.user_to_idx[user] for user in self.users if user.hasDest()]
            )
            if sending_users_indices.size == 0:
                return get_padded_array(shape=(self.numbOfNodes, 2), pad_value=0)

            # Get positions of sender users
            user_positions = self.user_positions[sending_users_indices]

            # Compute pairwise differences using broadcasting
            # Shape: (N, S, 2) where S is number of sender users
            differences = (
                user_positions[np.newaxis, :, :] - node_positions[:, np.newaxis, :]
            )

            # Compute distances (norms) for all pairs
            distances = np.linalg.norm(differences, axis=2)  # Shape: (N, S)

            # Find the index of the closest sender user for each node
            closest_sender_indices = np.argmin(distances, axis=1)  # Shape: (N,)

            # Extract directions to the closest sender users
            directions = differences[
                np.arange(self.numbOfNodes), closest_sender_indices
            ]  # Shape: (N, 2)

            def transform_func(x: np.ndarray) -> np.ndarray:
                norms = np.linalg.norm(x, axis=1, keepdims=True)  # Shape: (N, 1)
                # Replace norms < 1 with 1
                adjusted_norms = (
                    np.maximum(norms, 1.0) + 1e-9
                )  # Small epsilon for numerical stability
                return (
                    x / adjusted_norms
                )  # Broadcasting divides each row by its adjusted norm

            # Apply transformation with default min_max_tuple=(-1, 1)
            normalized = normalize_and_transform(
                directions, transform_function=transform_func
            )
            # Offset applied after normalisation — see get_node_closest_user_dir.
            for i, node in enumerate(self.nodes):
                offset = getattr(node, "observed_user_dir_offset", None)
                if offset is not None:
                    normalized[i] += offset
            return normalized

    def get_node_closest_receiver_dir(self, use_replacement_value: bool) -> np.ndarray:
        """For each node returns the normed direction to the closest receiving user, one that is the destination of another users offered load.

        For obs_user_dir Byzantine nodes the `observed_user_dir_offset` property is applied,
        adding a fixed offset to the reported direction to mislead neighbours about receiver locations.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: Normed direction to the closest receiving user, one that is the destination of another users offered load.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 2), pad_value=0)
        else:
            # Store node positions for easier reference
            node_positions = self.node_positions

            # Find unique indices of all receiver users (users that are destinations for other users)
            receiver_users_indices = list(
                {self.user_to_idx[user.dest] for user in self.users if user.hasDest()}
            )
            if len(receiver_users_indices) == 0:
                return get_padded_array(shape=(self.numbOfNodes, 2), pad_value=0)

            # Extract positions of only the receiver users (without duplicates)
            receiver_positions = self.user_positions[receiver_users_indices]

            # Compute pairwise differences using broadcasting
            # Shape: (N, S, 2) where S is number of sender users
            differences = (
                receiver_positions[np.newaxis, :, :] - node_positions[:, np.newaxis, :]
            )

            # Compute distances (norms) for all pairs
            distances = np.linalg.norm(differences, axis=2)  # Shape: (N, S)

            # Find the index of the closest sender user for each node
            closest_sender_indices = np.argmin(distances, axis=1)  # Shape: (N,)

            # Extract directions to the closest sender users
            directions = differences[
                np.arange(self.numbOfNodes), closest_sender_indices
            ]  # Shape: (N, 2)

            def transform_func(x: np.ndarray) -> np.ndarray:
                norms = np.linalg.norm(x, axis=1, keepdims=True)  # Shape: (N, 1)
                # Replace norms < 1 with 1
                adjusted_norms = (
                    np.maximum(norms, 1.0) + 1e-9
                )  # Small epsilon for numerical stability
                return (
                    x / adjusted_norms
                )  # Broadcasting divides each row by its adjusted norm

            # Apply transformation with default min_max_tuple=(-1, 1)
            normalized = normalize_and_transform(
                directions, transform_function=transform_func
            )
            # Offset applied after normalisation — see get_node_closest_user_dir.
            for i, node in enumerate(self.nodes):
                offset = getattr(node, "observed_user_dir_offset", None)
                if offset is not None:
                    normalized[i] += offset
            return normalized

    def get_node_closest_user_interference(
        self, use_replacement_value: bool
    ) -> np.ndarray:
        """For each node, returns the interference received from the closest user, normalized to [-1, 1].

        Args:
            use_replacement_value (bool): Whether to return a padded array instead of actual observations.

        Returns:
            np.ndarray: Normalized interference from the closest user for each node, shape (numbOfNodes, 1).
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            # Get node and user positions directly as NumPy arrays
            node_positions = self.node_positions
            user_positions = self.user_positions

            # Compute pairwise distances using broadcasting
            # Shape: (N, 1, 2) - (1, U, 2) -> (N, U, 2)
            differences = (
                node_positions[:, np.newaxis, :] - user_positions[np.newaxis, :, :]
            )
            distances = np.linalg.norm(differences, axis=2)  # Shape: (N, U)

            # Find the index of the closest user for each node
            closest_user_indices = np.argmin(distances, axis=1)  # Shape: (N,)

            # Get interference values from all users
            user_interferences = self.get_user_interference(False)

            # Extract interference from the closest user for each node
            closest_interferences = user_interferences[closest_user_indices, :]

            return closest_interferences

    def get_node_closest_user_position(self, use_replacement_value: bool) -> np.ndarray:
        """For each node, returns the position received from the closest user, normalized to [-1, 1]^2.

        Args:
            use_replacement_value (bool): Whether to return a padded array instead of actual observations.

        Returns:
            np.ndarray: Normalized position from the closest user for each node, shape (numbOfNodes, 2).
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 2), pad_value=0)
        else:
            # Get node and user positions directly as NumPy arrays
            node_positions = self.node_positions
            user_positions = self.user_positions

            # Compute pairwise distances using broadcasting
            # Shape: (N, 1, 2) - (1, U, 2) -> (N, U, 2)
            differences = (
                node_positions[:, np.newaxis, :] - user_positions[np.newaxis, :, :]
            )
            distances = np.linalg.norm(differences, axis=2)  # Shape: (N, U)

            # Find the index of the closest user for each node
            closest_user_indices = np.argmin(distances, axis=1)  # Shape: (N,)

            # Get interference values from all users
            user_positions = self.get_user_position(False)

            # Extract interference from the closest user for each node
            closest_positions = user_positions[closest_user_indices, :]

            return closest_positions

    def get_node_closest_sender_position(
        self, use_replacement_value: bool
    ) -> np.ndarray:
        """For each node, returns the position of the closest sending user, a user that has a destination.

        Args:
            use_replacement_value (bool): Whether to return a padded array instead of actual observations.

        Returns:
            np.ndarray: Position of the closest sending user, a user that has a destination.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 2), pad_value=-1)
        else:
            # Get node positions directly as NumPy array
            node_positions = self.node_positions

            # Identify sender users (those with a destination)
            sending_users_indices = np.array(
                [self.user_to_idx[user] for user in self.users if user.hasDest()]
            )
            if sending_users_indices.size == 0:
                return np.zeros(shape=(self.numbOfNodes, 2))

            # Get positions of sender users
            user_positions = self.user_positions[sending_users_indices]

            # Compute pairwise distances using broadcasting
            # Shape: (N, 1, 2) - (1, U, 2) -> (N, U, 2)
            differences = (
                node_positions[:, np.newaxis, :] - user_positions[np.newaxis, :, :]
            )
            distances = np.linalg.norm(differences, axis=2)  # Shape: (N, U)

            # Find the index of the closest user for each node
            closest_user_indices = np.argmin(distances, axis=1)  # Shape: (N,)

            # Get interference values from all users
            user_positions = self.get_user_position(False)

            # Extract interference from the closest user for each node
            closest_interferences = user_positions[closest_user_indices, :]

            return closest_interferences

    def get_node_closest_receiver_position(
        self, use_replacement_value: bool
    ) -> np.ndarray:
        """For each node, returns the position of the closest receiving user, a user that is another's destination.

        Args:
            use_replacement_value (bool): Whether to return a padded array instead of actual observations.

        Returns:
            np.ndarray: Position of the closest receiving user, a user that is another's destination.
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 2), pad_value=-1)
        else:
            # Store node positions for easier reference
            node_positions = self.node_positions

            # Find unique indices of all receiver users (users that are destinations for other users)
            receiver_users_indices = list(
                {self.user_to_idx[user.dest] for user in self.users if user.hasDest()}
            )
            if len(receiver_users_indices) == 0:
                return np.zeros(shape=(self.numbOfNodes, 2))

            # Extract positions of only the receiver users (without duplicates)
            receiver_positions = self.user_positions[receiver_users_indices]

            # Compute pairwise distances using broadcasting
            # Shape: (N, 1, 2) - (1, U, 2) -> (N, U, 2)
            differences = (
                node_positions[:, np.newaxis, :] - receiver_positions[np.newaxis, :, :]
            )
            distances = np.linalg.norm(differences, axis=2)  # Shape: (N, U)

            # Find the index of the closest user for each node
            closest_user_indices = np.argmin(distances, axis=1)  # Shape: (N,)

            # Get interference values from all users
            user_positions = self.get_user_position(False)

            # Extract interference from the closest user for each node
            closest_interferences = user_positions[closest_user_indices, :]

            return closest_interferences

    # * Byzantine robustness signal observations

    def get_node_signal_consistency(self, use_replacement_value: bool) -> np.ndarray:
        """For each node, returns the signal_consistency score normalised to [-1, 1].

        signal_consistency = true_signal_power / observed_signal_power.
        Value is 1.0 for honest nodes and < 1.0 when a sinkhole node inflates its
        advertised signal power.  Computed each step by Network.computeRobustnessSignals().

        Args:
            use_replacement_value (bool): Whether to return a padded array instead.

        Returns:
            np.ndarray: Shape (numbOfNodes, 1), values in [-1, 1].
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=1)
        else:
            values = np.array(
                [node.signal_consistency for node in self.nodes], dtype=np.float32
            ).reshape(-1, 1)
            return normalize_and_transform(values, (0.0, 1.0))

    def get_node_pos_consistency(self, use_replacement_value: bool) -> np.ndarray:
        """For each node, returns the pos_consistency score normalised to [-1, 1].

        pos_consistency measures how well a node's claimed position agrees with
        positions inferred from received signal strengths at its neighbours.
        Value is 1.0 for honest nodes and < 1.0 when position spoofing is detected.
        Computed each step by Network.computeRobustnessSignals().

        Args:
            use_replacement_value (bool): Whether to return a padded array instead.

        Returns:
            np.ndarray: Shape (numbOfNodes, 1), values in [-1, 1].
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=1)
        else:
            values = np.array(
                [node.pos_consistency for node in self.nodes], dtype=np.float32
            ).reshape(-1, 1)
            return normalize_and_transform(values, (0.0, 1.0))

    def get_node_unexplained_interference(self, use_replacement_value: bool) -> np.ndarray:
        """For each node, returns the unexplained interference normalised to [-1, 1].

        unexplained_interference is the interference received by a node beyond what
        all external jammers can account for.  A selective-jamming Byzantine node raises
        this above zero for every component it can reach.
        Normalised with the same dBm log-scale transform used for interference.

        Args:
            use_replacement_value (bool): Whether to return a padded array instead.

        Returns:
            np.ndarray: Shape (numbOfNodes, 1), values in [-1, 1].
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=-1)
        else:
            values = np.array(
                [node.unexplained_interference for node in self.nodes], dtype=np.float32
            ).reshape(-1, 1)
            epsilon = 1e-9
            transform_function = lambda x: 10 * np.log10(x + epsilon)
            return normalize_and_transform(values, self.interferenceMinMaxTuple, transform_function)

    def get_node_trust(self, use_replacement_value: bool) -> np.ndarray:
        """For each node, returns the EMA trust score normalised to [-1, 1].

        trust is an exponential moving average that accumulates evidence from
        signal_consistency, pos_consistency, and unexplained_interference over time.
        Value 1.0 = no anomalous behaviour observed; 0.0 = consistently anomalous.
        Computed each step by Network.updateTrust().

        Args:
            use_replacement_value (bool): Whether to return a padded array instead.

        Returns:
            np.ndarray: Shape (numbOfNodes, 1), values in [-1, 1].
        """
        if use_replacement_value:
            return get_padded_array(shape=(self.numbOfNodes, 1), pad_value=1)
        else:
            values = np.array(
                [node.trust for node in self.nodes], dtype=np.float32
            ).reshape(-1, 1)
            return normalize_and_transform(values, (0.0, 1.0))

    # * User observations
    def get_user_position(self, use_replacement_value: bool) -> np.ndarray:
        """Get the users' positions, which should already be normalized and transformed such
        that they each are inside the observation space and scaled properly over it. So, each observation should be in [-1,1].

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: observation or padded observation.
        """
        if use_replacement_value:
            return get_padded_array((self.unwrapped_env.numbOfUsers, 2), 0)
        else:
            user_positions = self.user_positions
            return normalize_and_transform(user_positions, self.posMinMaxTuple)

    def get_user_signal_received(self, use_replacement_value: bool) -> np.ndarray:
        """Get the users' signals received, which should already be normalized and transformed such
        that they each are inside the observation space and scaled properly over it. So, each observation should be in [-1,1].

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: observation or padded observation.
        """
        if use_replacement_value:
            return get_padded_array(
                shape=(self.unwrapped_env.numbOfUsers, 1), pad_value=-1
            )
        else:
            signals = np.array([user.signalReceived_mW for user in self.users]).reshape(
                -1, 1
            )

            epsilon = 1e-9
            transform_function = lambda x: 10 * np.log10(x + epsilon)

            return normalize_and_transform(
                signals, self.signalMinMaxTuple, transform_function
            )

    def get_user_interference(self, use_replacement_value: bool) -> np.ndarray:
        """Get the users' interferences, which should already be normalized and transformed such
        that they each are inside the observation space and scaled properly over it. So, each observation should be in [-1,1].

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: observation or padded observation.
        """
        if use_replacement_value:
            return get_padded_array(
                shape=(self.unwrapped_env.numbOfUsers, 1), pad_value=-1
            )
        else:
            interferences = np.array(
                [user.interferenceReceived_mW for user in self.users]
            ).reshape(-1, 1)

            epsilon = 1e-9
            transform_function = lambda x: 10 * np.log10(x + epsilon)

            return normalize_and_transform(
                interferences,
                self.interferenceMinMaxTuple,
                transform_function,
            )

    def get_user_dest_indices(self, use_replacement_value: bool) -> np.ndarray:
        """Get the users' destination indices, which should already be normalized and transformed such
        that they each are inside the observation space and scaled properly over it. So, each observation should be in [-1,1].

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: observation or padded observation.
        """
        if use_replacement_value:
            return get_padded_array(
                shape=(self.unwrapped_env.numbOfUsers, 1), pad_value=-1
            )
        else:
            dest_indices = np.array(
                [
                    self.user_to_idx.get(user.dest, self.unwrapped_env.numbOfUsers)
                    for user in self.users
                ]
            ).reshape(-1, 1)
            return normalize_and_transform(
                dest_indices, (0, self.unwrapped_env.numbOfUsers)
            )

    def get_user_dest_position(self, use_replacement_value: bool) -> np.ndarray:
        """Get the users' destination positions, which should already be normalized and transformed such
        that they each are inside the observation space and scaled properly over it. So, each observation should be in [-1,1].

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: observation or padded observation.
        """
        if use_replacement_value:
            return get_padded_array((self.unwrapped_env.numbOfUsers, 2), -1)
        else:
            dest_pos = np.array(
                [
                    (user.dest.pos if user.dest is not None else np.zeros((2,)))
                    for user in self.users
                ]
            )
            return normalize_and_transform(dest_pos, self.posMinMaxTuple)

    def get_user_signal_power(self, use_replacement_value: bool) -> np.ndarray:
        """Get the users' signal powers, which should already be normalized and transformed such
        that they each are inside the observation space and scaled properly over it. So, each observation should be in [-1,1].

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: observation or padded observation.
        """
        if use_replacement_value:
            return get_padded_array(
                shape=(self.unwrapped_env.numbOfUsers, 1), pad_value=-1
            )
        else:
            signalpower = np.array(
                [user.signalPower_mW for user in self.users]
            ).reshape(-1, 1)
            return normalize_and_transform(signalpower, self.signalPowMinMaxTuple)

    def get_user_offered_load(self, use_replacement_value: bool) -> np.ndarray:
        """Get the users' offered loads, which should already be normalized and transformed such
        that they each are inside the observation space and scaled properly over it. So, each observation should be in [-1,1].

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: observation or padded observation.
        """
        if use_replacement_value:
            return get_padded_array(
                shape=(self.unwrapped_env.numbOfUsers, 1), pad_value=-1
            )
        else:
            offeredload = np.array([user.offeredLoad for user in self.users]).reshape(
                -1, 1
            )
            return normalize_and_transform(offeredload, self.offeredLoadMinMaxTuple)

    # * General observations
    def get_capacity_matrix(self, use_replacement_value: bool) -> np.ndarray:
        """
        Returns the normalized capacity matrix.

        Args:
            use_replacement_value (bool): Whether instead of the actual observations a padded array should be returned.

        Returns:
            np.ndarray: The normalized capacity matrix.
        """
        if use_replacement_value:
            return get_padded_array(
                (
                    self.numbOfNodes + self.unwrapped_env.numbOfUsers,
                    self.numbOfNodes + self.unwrapped_env.numbOfUsers,
                ),
                -1,
            )
        else:
            capacity_matrix = self.unwrapped_env.network.getCapacityMatrix()

        return normalize_and_transform(capacity_matrix, self.capacityMinMaxTuple)

    # Configure observation space size
    def get_observation_size(self) -> int:
        """Calculates the size of the observation space by querying observation functions.

        Returns:
            int: The total size of the observation space.
        """
        total_size = 0

        # Node observations
        for key in NODE_KEYS:
            if self.config["nodes"].get(key, False) or self.is_padded:
                obs_array = self.NODE_OBSERVATION_FUNCTIONS[key](
                    use_replacement_value=True
                )
                total_size += obs_array.size

        # User observations
        for key in USER_KEYS:
            if self.config["users"].get(key, False) or self.is_padded:
                if key == "destination" and self.is_invariable:
                    obs_array = self.get_user_dest_position(use_replacement_value=True)
                else:
                    obs_array = self.USER_OBSERVATION_FUNCTIONS[key](
                        use_replacement_value=True
                    )
                total_size += obs_array.size

        # General observations
        for key in GENERAL_KEYS:
            if self.config.get(key, False) or self.is_padded:
                obs_array = self.GENERAL_OBSERVATION_FUNCTIONS[key](
                    use_replacement_value=True
                )
                total_size += obs_array.size

        return total_size


class Blocking_Base_ObservationWrapper(Base_ObservationWrapper):
    """To simulate state loss and realistic observation updates in deployment of our rl-agent, the `Blocking_Base_ObservationWrapper` serves as a blueprint to allow one to customely define how observations are blocked/updated or not."""

    # * Override
    def updateComponents(self) -> None:
        """Update components if necessary."""
        if self.hasEnvBeenReset():
            self.initializeAllComponents()
        else:
            self.customBlockedUpdates()

    def customBlockedUpdates(self) -> None:
        """Custom update process that can be implemented e.g. by the blocking wrappers"""

        pass

    # * Overwrite
    def initializeAllComponents(self):
        """
        Updates all components by deep copying nodes and users to ensure independence,
        and shallow copying positions since they are numpy arrays of primitive types.
        """
        self._replay_buffers = {}
        # Shallow copy positions since they are numpy arrays of primitive types
        self.node_positions = np.zeros(shape=(self.unwrapped_env.numbOfNodes, 2))
        self.user_positions = np.zeros(shape=(self.unwrapped_env.numbOfUsers, 2))

        # Deep copy nodes and users to ensure independence
        self.nodes: np.ndarray[net_comps.MANETNode] = np.array(
            [
                net_comps.MANETNode(
                    id=f"Node {i}",
                    pos=pos,
                    signalPower_mW=self.unwrapped_env.nodeSignalPow_mW,
                )
                for i, pos in enumerate(self.node_positions)
            ]
        )
        self.users: np.ndarray[net_comps.User] = np.array(
            [
                net_comps.User(
                    id=f"User {i}",
                    pos=pos,
                    signalPower_mW=self.unwrapped_env.userSignalPow_mW,
                )
                for i, pos in enumerate(self.user_positions)
            ]
        )

        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.user_to_idx = {user: idx for idx, user in enumerate(self.users)}


class Real_ObservationWrapper(Blocking_Base_ObservationWrapper):
    """The `Real_ObservationWrapper` simulates realistic inter-node communication for exchanging the observation information. Further, it allows to probabilistically model data loss, through the `state_loss_rate` param. Each update fails with probability equal to the `state_loss_rate`. Assumes rl agent is deployed decentralised."""

    def __init__(
        self,
        node_idx: int,
        state_loss_rate: float,
        env: Basic_MANETEnv,
        config: dict = None,
        gossip_hops: int = None,
    ):
        """Initialize.

        Args:
            node_idx (int): Id of the node it is deployed on. (in {0,..., numbOfNodes-1})
            state_loss_rate (float): Probability for component properties update to fail
            env (Basic_MANETEnv): The gymnasium environment which runs the scenario
            config (dict, optional): Config describing which observations to include. Defaults to None.
            gossip_hops (int, optional): If set, limits observation propagation to this many
                hops via BFS instead of the default full multi-hop flooding (nx.ancestors).
                gossip_hops=1 means only direct graph predecessors are visible.
                Default None preserves the original flooding behaviour.
        """
        super().__init__(env, config)

        self.node_idx = node_idx

        state_loss_rate = state_loss_rate if state_loss_rate is not None else 0
        assert (
            0 <= state_loss_rate <= 1
        ), "state_loss_rate must be between 0 and 1 (inclusive)."
        self.state_loss_rate = state_loss_rate
        self.gossip_hops = gossip_hops

        self.np_random = np.random.default_rng(seed=103 + self.node_idx)

    # * Overwrite
    def customBlockedUpdates(self):
        """
        Updates self.nodes and self.users to include only the components (MANET nodes and users)
        reachable by the current node within gossip_hops hops (BFS), or all ancestors via
        multi-hop flooding when gossip_hops is None. State loss is applied after reachability
        filtering. Each update fails with probability equal to the state_loss_rate.
        """
        # Get the current MANET node and the graph
        curr_node = self.unwrapped_env.nodes[self.node_idx]
        graph = self.unwrapped_env.network.getGraph()

        if self.gossip_hops is None:
            # Default: multi-hop flooding via nx.ancestors
            connected_comps: list = sorted(
                nx.ancestors(graph, curr_node), key=lambda x: x.id
            )
        else:
            # BFS limited to gossip_hops hops along graph predecessors
            connected_comps = []
            frontier = {curr_node}
            visited = {curr_node}
            for _ in range(self.gossip_hops):
                next_frontier = set()
                for node in frontier:
                    for pred in graph.predecessors(node):
                        if pred not in visited:
                            visited.add(pred)
                            next_frontier.add(pred)
                            connected_comps.append(pred)
                frontier = next_frontier
                if not frontier:
                    break
            connected_comps = sorted(connected_comps, key=lambda x: x.id)

        # Apply binomial distribution to filter connected components if state_loss_rate > 0
        if self.state_loss_rate > 0:
            mask = self.np_random.binomial(
                1, 1 - self.state_loss_rate, len(connected_comps)
            )
            connected_comps = [
                comp for comp, keep in zip(connected_comps, mask) if keep
            ]

        connected_comps.append(curr_node)  # Include the current node itself

        # Get the indices of the connected MANET nodes and users
        node_indices = [
            self.unwrapped_env.node_to_idx[comp]
            for comp in connected_comps
            if comp in self.unwrapped_env.node_to_idx
        ]
        user_indices = [
            self.unwrapped_env.user_to_idx[comp]
            for comp in connected_comps
            if comp in self.unwrapped_env.user_to_idx
        ]

        # Perform a deep copy of the nodes and users
        self.nodes[node_indices] = np.array(
            [deepcopy(self.unwrapped_env.nodes[idx]) for idx in node_indices]
        )
        self.users[user_indices] = np.array(
            [deepcopy(self.unwrapped_env.users[idx]) for idx in user_indices]
        )
        self.node_positions[node_indices] = np.copy(
            self.unwrapped_env.node_positions[node_indices]
        )
        self.user_positions[user_indices] = np.copy(
            self.unwrapped_env.user_positions[user_indices]
        )

        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.user_to_idx = {user: idx for idx, user in enumerate(self.users)}


class Random_ObservationWrapper(Blocking_Base_ObservationWrapper):
    """Assumes rl-agent is deployed centralized. For each node or user's properties randomly determines, whether they are updated given the `state_loss_rate` param. Each update fails with probability equal to the `state_loss_rate`."""

    def __init__(
        self,
        state_loss_rate: float,
        env: Basic_MANETEnv,
        config: dict = None,
    ):
        """Initialize.

        Args:
            state_loss_rate (float): Probability for component properties update to fail
            env (Basic_MANETEnv): The gymnasium environment which runs the scenario
            config (dict, optional): Config describing which observations to include. Defaults to None.
        """
        super().__init__(env, config)

        state_loss_rate = state_loss_rate if state_loss_rate is not None else 0
        assert (
            0 <= state_loss_rate <= 1
        ), "state_loss_rate must be between 0 and 1 (inclusive)."
        self.state_loss_rate = state_loss_rate

        self.np_random = np.random.default_rng(seed=42)

    # * Overwrite
    def customBlockedUpdates(self):
        """
        Updates self.nodes and self.users with states of nodes and users, using binomial distribution
        to simulate state loss rate, where indices are selected with p=1-state_loss_rate.
        """

        # Get all node and user indices
        node_indices = np.arange(self.unwrapped_env.numbOfNodes)
        user_indices = np.arange(self.unwrapped_env.numbOfUsers)

        # Apply binomial distribution to select indices to update
        node_update_mask = self.np_random.binomial(
            1, 1 - self.state_loss_rate, self.unwrapped_env.numbOfNodes
        )
        user_update_mask = self.np_random.binomial(
            1, 1 - self.state_loss_rate, self.unwrapped_env.numbOfUsers
        )

        # Filter indices to update
        node_indices_to_update = node_indices[node_update_mask == 1]
        user_indices_to_update = user_indices[user_update_mask == 1]

        # Perform a deep copy of the nodes and users
        self.nodes[node_indices_to_update] = np.array(
            [deepcopy(self.unwrapped_env.nodes[idx]) for idx in node_indices_to_update]
        )
        self.users[user_indices_to_update] = np.array(
            [deepcopy(self.unwrapped_env.users[idx]) for idx in user_indices_to_update]
        )

        # Shallow copy positions since they are numpy arrays of primitive types
        self.node_positions[node_indices_to_update] = np.copy(
            self.unwrapped_env.node_positions[node_indices_to_update]
        )
        self.user_positions[user_indices_to_update] = np.copy(
            self.unwrapped_env.user_positions[user_indices_to_update]
        )

        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.user_to_idx = {user: idx for idx, user in enumerate(self.users)}


class ReliableBroadcast_ObservationWrapper(Blocking_Base_ObservationWrapper):
    """Resiliency protocol that resists observation inflation attacks via min-merge.

    Uses the same multi-hop flooding as Real_ObservationWrapper to receive
    neighbor state, then applies an element-wise minimum against the cached
    value from the previous step for inflation-susceptible node attributes
    (observed_capacity_factor, observed_demand_factor).  This prevents a
    Byzantine node from progressively inflating its reported capacity or demand
    across steps — the honest node's cached minimum clamps any upward change.

    Limitation: provides NO protection against obs_user_dir (direction lies)
    because there is no meaningful minimum for a normalised 2D direction vector.
    """

    def __init__(
        self,
        node_idx: int,
        state_loss_rate: float,
        env: Basic_MANETEnv,
        config: dict = None,
    ):
        super().__init__(env, config)
        self.node_idx = node_idx
        state_loss_rate = state_loss_rate if state_loss_rate is not None else 0
        assert 0 <= state_loss_rate <= 1, "state_loss_rate must be between 0 and 1."
        self.state_loss_rate = state_loss_rate
        self.np_random = np.random.default_rng(seed=103 + node_idx)
        self._cache: dict = {}  # node row index → deepcopy from previous step

    def customBlockedUpdates(self):
        curr_node = self.unwrapped_env.nodes[self.node_idx]
        graph = self.unwrapped_env.network.getGraph()

        connected_comps: list = sorted(
            nx.ancestors(graph, curr_node), key=lambda x: x.id
        )

        if self.state_loss_rate > 0:
            mask = self.np_random.binomial(
                1, 1 - self.state_loss_rate, len(connected_comps)
            )
            connected_comps = [c for c, k in zip(connected_comps, mask) if k]

        connected_comps.append(curr_node)

        node_indices = [
            self.unwrapped_env.node_to_idx[comp]
            for comp in connected_comps
            if comp in self.unwrapped_env.node_to_idx
        ]
        user_indices = [
            self.unwrapped_env.user_to_idx[comp]
            for comp in connected_comps
            if comp in self.unwrapped_env.user_to_idx
        ]

        self.nodes[node_indices] = np.array(
            [deepcopy(self.unwrapped_env.nodes[idx]) for idx in node_indices]
        )
        self.users[user_indices] = np.array(
            [deepcopy(self.unwrapped_env.users[idx]) for idx in user_indices]
        )

        # Min-merge: clamp inflation-susceptible attributes against cached values.
        _INFLATION_ATTRS = ("observed_capacity_factor", "observed_demand_factor")
        for idx in node_indices:
            node = self.nodes[idx]
            if idx in self._cache:
                for attr in _INFLATION_ATTRS:
                    fresh = getattr(node, attr, 1.0)
                    cached = getattr(self._cache[idx], attr, 1.0)
                    if fresh != 1.0 or cached != 1.0:
                        setattr(node, attr, min(fresh, cached))
            self._cache[idx] = deepcopy(node)

        self.node_positions[node_indices] = np.copy(
            self.unwrapped_env.node_positions[node_indices]
        )
        self.user_positions[user_indices] = np.copy(
            self.unwrapped_env.user_positions[user_indices]
        )

        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.user_to_idx = {user: idx for idx, user in enumerate(self.users)}


# * Configurable Obswrappers
class Configurable_ObservationWrapper(ObservationWrapper):
    """Base class for configurable observation wrappers."""

    def __init__(
        self,
        wrapped_env: Base_ObservationWrapper,
        is_padded: bool = False,
        is_invariable: bool = False,
    ):
        """Transforms the any observation wrapper that inherits from the `Base_ObservationWrapper` into a Gymnasium compliant `ObservationWrapper` through wrapping around it.

        Args:
            wrapped_env (Base_ObservationWrapper): _description_
            is_padded (bool, optional): _description_. Defaults to False.
            is_invariable (bool, optional): _description_. Defaults to False.
        """
        assert isinstance(wrapped_env, Base_ObservationWrapper)

        super().__init__(wrapped_env.env)
        self.unwrapped_env: Basic_MANETEnv = wrapped_env.unwrapped_env
        self.env: gym.Env = wrapped_env.env
        self.wrapped_env: Base_ObservationWrapper = wrapped_env
        self.wrapped_env.setPadded(is_padded)
        self.wrapped_env.setInvariable(is_invariable)

        # Set the observation space
        numbOfObservations = self.wrapped_env.get_observation_size()
        observation_space = spaces.Box(
            low=-1, high=1, shape=(numbOfObservations,), dtype=np.float32
        )
        self.unwrapped_env.setObservationSpace(observation_space)

    def observation(self, obs=None) -> np.ndarray:
        """Return the observations.

        Args:
            obs (np.ndarray, optional): Given observation, that may be processed to. Defaults to None.

        Returns:
            np.ndarray: _description_
        """
        return self.wrapped_env.get_observation()

    def initializeComps(self) -> None:
        self.wrapped_env.initializeAllComponents()

    def updateComps(self) -> None:
        self.wrapped_env.updateComponents()


# * FixedSize_ConfigurableObservationWrapper
class FS_COW(Configurable_ObservationWrapper):
    """
    FixedSize_ConfigurableObservationWrapper customizes the observation space for a fixed size.
    This ensures that the observation space always has the same fixed size, regardless of which
    observations are included or excluded. Missing observations are padded with default values.

    Attributes:
        wrapped_env (Base_ObservationWrapper): The wrapped environment to customize.
    """

    def __init__(self, wrapped_env: Base_ObservationWrapper) -> None:
        """
        Initializes the FixedSize_ConfigurableObservationWrapper.

        Args:
            wrapped_env (Base_ObservationWrapper): The environment to wrap.
        """
        super().__init__(wrapped_env, True, False)


# * VariableSize_ConfigurableObservationWrapper
class VS_COW(Configurable_ObservationWrapper):
    """
    VariableSize_ConfigurableObservationWrapper customizes the observation space for a variable size.
    The observation space dynamically adjusts to include only the selected observations, resulting
    in a more compact representation.

    Attributes:
        wrapped_env (Base_ObservationWrapper): The wrapped environment to customize.
    """

    def __init__(self, wrapped_env: Base_ObservationWrapper) -> None:
        """
        Initializes the VariableSize_ConfigurableObservationWrapper.

        Args:
            wrapped_env (Base_ObservationWrapper): The environment to wrap.
        """
        super().__init__(wrapped_env, False, False)


# * invariable_FixedSize_ConfigurableObservationWrapper
class FS_iCOW(Configurable_ObservationWrapper):
    """
    invariable_FixedSize_ConfigurableObservationWrapper customizes the observation space for a fixed size
    while ensuring invariability. This means that the observation size is fixed, and user observations
    are cleared of symmetries by sorting them based on their positions.

    Attributes:
        wrapped_env (Base_ObservationWrapper): The wrapped environment to customize.
    """

    def __init__(self, wrapped_env: Base_ObservationWrapper) -> None:
        """
        Initializes the invariable_FixedSize_ConfigurableObservationWrapper.

        Args:
            wrapped_env (Base_ObservationWrapper): The environment to wrap.
        """
        super().__init__(wrapped_env, True, True)


# * invariable_VariableSize_ConfigurableObservationWrapper
class VS_iCOW(Configurable_ObservationWrapper):
    """
    invariable_VariableSize_ConfigurableObservationWrapper customizes the observation space for a variable size
    while ensuring invariability. The observation size dynamically adjusts to include only the selected observations,
    and user observations are cleared of symmetries by sorting them based on their positions.

    Attributes:
        wrapped_env (Base_ObservationWrapper): The wrapped environment to customize.
    """

    def __init__(self, wrapped_env: Base_ObservationWrapper) -> None:
        """
        Initializes the invariable_VariableSize_ConfigurableObservationWrapper.

        Args:
            wrapped_env (Base_ObservationWrapper): The environment to wrap.
        """
        super().__init__(wrapped_env, False, True)


def is_valid_config(config: dict) -> bool:
    """
    Checks if the given configuration is valid.

    Args:
        config (dict): The configuration dictionary to validate.

    Returns:
        is_valid (bool): True if the configuration is valid, False otherwise.
    """

    def keys_in_other_config(config: dict, other_config: dict) -> bool:
        for key in config:
            if key not in other_config:
                return False
            if type(config[key]) != type(other_config[key]):
                return False
            if isinstance(config[key], dict):
                if not keys_in_other_config(config[key], other_config[key]):
                    return False
        return True

    return keys_in_other_config(config, DEFAULT_CONFIG)


def filter_config(config: dict) -> dict:
    """
    Filters the given config to remove keys that do not belong to the default config.

    Args:
        config (dict): The custom configuration dictionary to filter.

    Returns:
        dict: The filtered configuration dictionary.
    """

    def recursive_filter(config: dict, default: dict) -> dict:
        filtered_config = {}
        for key, value in config.items():
            if key in default:
                if isinstance(value, dict) and isinstance(default[key], dict):
                    filtered_config[key] = recursive_filter(value, default[key])
                elif isinstance(value, bool):
                    filtered_config[key] = value
        return filtered_config

    return recursive_filter(config, DEFAULT_CONFIG)


def extend_with_missing_keys(config: dict) -> dict:
    """
    Ensures all keys from the default config are present in the given config.
    Missing keys (including nested ones) are added with the value False.

    Args:
        config (dict): The config to extend.

    Returns:
        dict: The extended config with missing keys filled with False.
    """
    default_config = DEFAULT_CONFIG

    def recursive_extend(config: dict, default: dict) -> None:
        for key, default_value in default.items():
            if key not in config:
                config[key] = False  # Add missing key with False
            elif isinstance(default_value, dict):
                # Ensure that the sub-config matches the structure of the default
                if not isinstance(config[key], dict):
                    config[key] = False  # Fallback to False if it's not a dict
                else:
                    # Recursively call on the sub-dictionaries to ensure they match
                    recursive_extend(config[key], default_value)

    # Create a copy of the original config to avoid modifying it directly
    extended_config = config.copy()
    recursive_extend(extended_config, default_config)

    return extended_config


def keys_match(dict1: dict, dict2: dict) -> bool:
    """
    Checks if the keys of two dictionaries match.

    Args:
        dict1 (dict): The first dictionary.
        dict2 (dict): The second dictionary.

    Returns:
        keys_match (bool): True if the keys match, False otherwise.
    """
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1.keys():
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            if not keys_match(dict1[key], dict2[key]):
                return False
    return True


def get_padded_array(shape: tuple[int, int], pad_value: float) -> np.ndarray:
    return np.full(shape, fill_value=pad_value, dtype=np.float32)


def normalize_and_transform(
    values: np.ndarray,
    min_max_tuple: tuple[float, float] = (-1, 1),
    transform_function: Callable[[np.ndarray], np.ndarray] = None,
) -> np.ndarray:
    """
    Normalizes and optionally transforms the input values.

    Args:
        values (np.ndarray): Values to normalize and transform.
        min_max_tuple (tuple[float, float], optional): Tuple with a minimum and maximum value for a certain observation, defining its bounds.
        transform_function (Callable[[np.ndarray], np.ndarray], optional): Function to transform the observations, as some - e.g. the interference - scale non-linearly. Defaults to None.

    Returns:
        np.ndarray: The normalized and transformed values.
    """

    if transform_function:
        transformed_values = transform_function(values)
    else:
        transformed_values = values

    min_value, max_value = min_max_tuple
    normed_values = normalize_values(transformed_values, min_value, max_value)

    np.clip(a=normed_values, a_min=-1, a_max=1, out=normed_values)

    return normed_values.astype(np.float32)


def normalize_values(
    values: np.ndarray, min_value: float, max_value: float
) -> np.ndarray:
    """Normalizes the values, such that they are spread over [-1, 1]

    Args:
        values (np.ndarray): Values to normalize
        min_value (float): Minimum possible value, they can take.
        max_value (float): Maximum possible value, they can take.

    Returns:
        np.ndarray: Normalized values
    """

    normed_values = 2 * (values - min_value) / (max_value - min_value) - 1

    return normed_values
