# Standard Python Libs
import numpy as np
from abc import ABC, abstractmethod

# Own classes
from .Environments import Basic_MANETEnv
import MANET.Components as net_comps

"""
# ConditionStrategy.py

In [Agents.py](#MANET.Agents) multiple agent models have been created and similar in [Attackers.py](#MANET.Attackers), though these then are attackers, not agents. Overall, they all have one thing in common: They have to sample for some data, they can perceive, similar to the RL-Agents, which sample through their observation space. These `Agents` and `Attackers` are focus on what they do with what they sample and how they sample, but not what they sample. Hence, the `Condition Strategies` were implemented such that for `Agents` and `Attackers` they are modular and one can use the same `Agent`/`Attacker` can be more effective by simply using a different `ConditionStrategy`. 

So, the `ConditionStrategy` decides, what value to sample and further each strategy includes a condition for the agent to opt to stay at a position. For example, if the throughput is at a feasible level, then the agents could stop moving and by that stop wasting energy. To create a custom condition strategy, one should either inherit from [LocalConditionStrategy](#MANET.ConditionStrategy.LocalConditionStrategy) or [GlobalConditionStrategy](#MANET.ConditionStrategy.GlobalConditionStrategy), depending on whether the sampled value is local (measured at the node's level) or global (measured at the environment's level).


__ConditionStrategies__

- Local:
    - [SINR](#MANET.ConditionStrategy.SINRConditionStrategy): Maximizing the signal-to-interference-plus-noise ratio.
    - [Signal](#MANET.ConditionStrategy.SignalConditionStrategy): Maximizing the signal.
    - [Interference](#MANET.ConditionStrategy.InterferenceConditionStrategy): Minimizing the interference.
    - [TrafficHosted](#MANET.ConditionStrategy.TrafficHostedConditionStrategy): Maximizing the traffic the node hosts.

- Global:
    - [Throughput](#MANET.ConditionStrategy.ThroughputConditionStrategy): Maximizing the global throughput.
"""


class ConditionStrategy(ABC):
    """
    Used to define a strategy of sampling with a fitting condition
    of starting and stopping to sample. E.g. the SINR condition strategy
    samples for the highest SINR and can be used for different agents.
    So, then the GreedyMaximizer for instance, would maximize the SINR measured.
    The condition is defined through the threshold. If the sampling in a starting or
    ending state of the Agents finds values below that threshold the agent will try to
    improve the current position. Else it will stay.

    Attributes:
        threshold (float): The threshold value that dictates whether to stay or start a relocation.
    """

    def __init__(self, threshold: float) -> None:
        """
        Initialize the ConditionStrategy with a threshold value.

        Args:
            threshold (float): The threshold value for the condition strategy.
        """
        self.threshold = threshold

    def is_unviable(self, node: net_comps.MANETNode) -> bool:
        """
        Check if the node's condition is unviable.

        Args:
            node (net_comps.MANETNode): The MANET node to check.

        Returns:
            bool: True if the node's condition is unviable, False otherwise.
        """
        value = self.sample(node)
        is_unviable = self.value_is_unviable(value)
        return is_unviable

    def value_is_unviable(self, value: float) -> bool:
        """
        Check if the given value is unviable.

        Args:
            value (float): The value to check.

        Returns:
            bool: True if the value is unviable, False otherwise.
        """
        is_unviable = self.threshold > value
        return is_unviable

    @abstractmethod
    def sample(self, node: net_comps.MANETNode) -> float:
        """
        Sample the condition value for the given node.

        Args:
            node (net_comps.MANETNode): The MANET node to sample.

        Returns:
            float: The sampled condition value.
        """
        pass

    @abstractmethod
    def is_global(self) -> bool:
        """
        Check if the condition strategy is global.

        Returns:
            bool: True if the condition strategy is global, False otherwise.
        """
        pass

    @abstractmethod
    def is_local(self) -> bool:
        """
        Check if the condition strategy is local.

        Returns:
            bool: True if the condition strategy is local, False otherwise.
        """
        pass


class LocalConditionStrategy(ConditionStrategy):
    """
    Local condition strategies are `ConditionStrategies` that use a local strategy,
    such as for instance the signal a node receives. So, the strategy is specific to a value one node can sample or has local access to.

    Attributes:
        threshold (float): The threshold value for the condition strategy.
    """

    def is_local(self) -> bool:
        """
        Check if the condition strategy is local.

        Returns:
            bool: True if the condition strategy is local, False otherwise.
        """
        return True

    def is_global(self) -> bool:
        """
        Check if the condition strategy is global.

        Returns:
            bool: True if the condition strategy is global, False otherwise.
        """
        return not self.is_local()


class GlobalConditionStrategy(ConditionStrategy):
    """
    Global condition strategies are `ConditionStrategies` that use a global strategy,
    such as for instance the network throughput. Hence, they can take the environment as an attribute.

    Attributes:
        threshold (float): The threshold value for the condition strategy.
        env (Basic_MANETEnv): The environment to use for the condition strategy.
    """

    def __init__(self, threshold: float, env: Basic_MANETEnv = None) -> None:
        """
        Initialize the GlobalConditionStrategy with a threshold value and an optional environment.

        Args:
            threshold (float): The threshold value for the condition strategy.
            env (Basic_MANETEnv, optional): The environment to use for the condition strategy.
        """
        super().__init__(threshold)
        self._env: Basic_MANETEnv = env

    def is_global(self) -> bool:
        """
        Check if the condition strategy is global.

        Returns:
            bool: True if the condition strategy is global, False otherwise.
        """
        return True

    def is_local(self) -> bool:
        """
        Check if the condition strategy is local.

        Returns:
            bool: True if the condition strategy is local, False otherwise.
        """
        return not self.is_global()

    def setEnv(self, env: Basic_MANETEnv) -> None:
        """
        Set the environment for the condition strategy.

        Args:
            env (Basic_MANETEnv): The environment to set.
        """
        self._env = env

    def envExists(self) -> bool:
        """
        Check if the environment exists.

        Returns:
            bool: True if the environment exists, False otherwise.
        """
        return self._env is not None


class SINRConditionStrategy(LocalConditionStrategy):
    """
    Strategy for maximizing the local SINR.

    Attributes:
        threshold_sinr (float): The threshold SINR value in dB.
    """

    def __init__(self, threshold_sinr: float = 10) -> None:
        """
        Initialize the SINRConditionStrategy with a threshold SINR value.

        Args:
            threshold_sinr (float, optional): The threshold SINR value in dB. Defaults to 10.
        """
        super().__init__(threshold=threshold_sinr)

    def sample(self, node: net_comps.MANETNode) -> float:
        """
        Sample the SINR value for the given node.

        Args:
            node (net_comps.MANETNode): The MANET node to sample.

        Returns:
            float: The sampled SINR value in dB.
        """
        epsilon = 1e-10  # Small value to prevent log10(0)
        sinr = 10 * np.log10(
            (node.signalReceived_mW + epsilon)
            / (node.interferenceReceived_mW + net_comps.ENVIRONMENT_NOISE)
        )
        return sinr


class SignalConditionStrategy(LocalConditionStrategy):
    """
    Strategy for maximizing the local sum of signals received.

    Attributes:
        threshold_signal (float): The threshold signal value in mW.
    """

    def __init__(self, threshold_signal: float = 1) -> None:
        """
        Initialize the SignalConditionStrategy with a threshold signal value.

        Args:
            threshold_signal (float, optional): The threshold signal value in mW. Defaults to 1.
        """
        super().__init__(threshold=threshold_signal)

    def sample(self, node: net_comps.MANETNode) -> float:
        """
        Sample the signal value for the given node.

        Args:
            node (net_comps.MANETNode): The MANET node to sample.

        Returns:
            float: The sampled signal value in mW.
        """
        signal = node.signalReceived_mW
        return signal


class InterferenceConditionStrategy(LocalConditionStrategy):
    """
    Strategy to minimize the local sum of interference received.

    Attributes:
        threshold_interference (float): The threshold interference value in mW.
    """

    def __init__(self, threshold_interference: float = 1e-7) -> None:
        """
        Initialize the InterferenceConditionStrategy with a threshold interference value.

        Args:
            threshold_interference (float, optional): The threshold interference value in mW. Defaults to 1e-3.
        """
        super().__init__(threshold=threshold_interference)

    def sample(self, node: net_comps.MANETNode) -> float:
        """
        Sample the interference value for the given node.

        Args:
            node (net_comps.MANETNode): The MANET node to sample.

        Returns:
            float: The sampled interference value in mW.
        """
        interference = -node.interferenceReceived_mW
        return interference


class TrafficHostedConditionStrategy(LocalConditionStrategy):
    """
    Strategy to maximize the local traffic hosted. Traffic hosted means data routed over the node. So, the sum of all ingoing / outgoing flows, since ingoing = outgoing.

    Attributes:
        traffic_threshold (int): The traffic threshold value in bits.
    """

    def __init__(self, traffic_threshold: int = 10e6) -> None:
        """
        Initialize the TrafficHostedConditionStrategy with a traffic threshold value.

        Args:
            traffic_threshold (int, optional): The traffic threshold value in bits. Defaults to 10e6.
        """
        super().__init__(threshold=traffic_threshold)

    def sample(self, node: net_comps.MANETNode) -> float:
        """
        Sample the traffic hosted value for the given node.

        Args:
            node (net_comps.MANETNode): The MANET node to sample.

        Returns:
            float: The sampled traffic hosted value in bits.
        """
        traffic_hosted: float = float(node.trafficHosted)
        return traffic_hosted


class ThroughputConditionStrategy(GlobalConditionStrategy):
    """
    Strategy to maximize the global network throughput.

    Attributes:
        threshold_throughput (float): The throughput threshold value.
        env (Basic_MANETEnv): The environment to use for the condition strategy.
    """

    def __init__(self, threshold_throughput: float, env: Basic_MANETEnv = None) -> None:
        """
        Initialize the ThroughputConditionStrategy with a throughput threshold value and an environment.

        Args:
            threshold_throughput (float): The throughput threshold value.
            env (Basic_MANETEnv): The environment to use for the condition strategy.
        """
        super().__init__(threshold=threshold_throughput, env=env)

    def sample(self, node: net_comps.MANETNode) -> float:
        """
        Sample the throughput value for the given node.

        Args:
            node (net_comps.MANETNode): The MANET node to sample.

        Returns:
            float: The sampled throughput value.
        """
        self._env: Basic_MANETEnv
        throughput = self._env.network.getThroughput()
        return throughput
