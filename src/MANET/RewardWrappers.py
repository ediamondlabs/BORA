# standard stuff
import numpy as np

# Type hinting
from abc import ABC, abstractmethod

# ObservationWrapper to customize the observation space
from gymnasium import RewardWrapper
import gymnasium as gym

# MANET Environment
from .Environments import Basic_MANETEnv

"""
# RewardWrappers.py

Reward Wrappers provide a neat and flexible way to define different reward functions without needing to modify the environment class itself. This design enables efficient customization of reward functions tailored to specific requirements.

__Reward Wrappers__

There are, so far, four reward wrappers:

- [MovementCost](#MANET.RewardWrappers.MovementCost): Penalizes the agent for moving, encouraging it to quickly move to the optimum position and then stop moving.
- [LinearReward](#MANET.RewardWrappers.LinearReward): Is simply the network throughput without modifications.
- [SquaredReward](#MANET.RewardWrappers.SquaredReward): Adjusts the reward based on the square of the network throughput, increasing the slope of the reward function.
- [EvaluationReward](#MANET.RewardWrappers.EvaluationReward): Provides a simple evaluation reward based on network throughput. This is practically the same as the `LinearReward` but is intended for evaluation purposes.
"""


class BaseRewardWrapper(RewardWrapper, ABC):
    """
    Base class for reward wrappers to handle common logic when the episode is over.

    Attributes:
        env (gym.Env): The environment to wrap.
        done_factor (float): Factor to multiply the reward by when the episode is done.
        truncated_penalty (float): Penalty to subtract from the reward when the episode is truncated.
    """

    def __init__(
        self, env: gym.Env, done_factor: float = 5.0, truncated_penalty: float = 0
    ) -> None:
        """
        Initialize the BaseRewardWrapper.

        Args:
            env (gym.Env): The environment to wrap.
            done_factor (float): Factor to multiply the reward by when the episode is done.
            truncated_penalty (float): Penalty to subtract from the reward when the episode is truncated.
        """
        super().__init__(env)

        self._env: Basic_MANETEnv = self.env.unwrapped
        self.done_factor = done_factor
        self.truncated_penalty = truncated_penalty

    def reward(self, reward: float) -> float:
        """
        Adjust the reward based on the environment's state.

        Args:
            reward (float): The original reward.

        Returns:
            float: The adjusted reward.
        """
        if self._env.done():
            reward = self.calculate_done_reward()
        elif self._env.truncated():
            reward = self.calculate_truncated_reward() - self.calculate_penalty()
        else:
            reward = self.calculate_reward() - self.calculate_penalty()

        return reward

    @abstractmethod
    def calculate_reward(self) -> float:
        """
        Calculate the reward based on the environment's state.

        Returns:
            float: The calculated reward.
        """
        pass

    def calculate_done_reward(self) -> float:
        """
        Calculate the reward when the episode is done.

        Returns:
            float: The calculated done reward.
        """
        return self.done_factor * self.calculate_reward()

    def calculate_truncated_reward(self) -> float:
        """
        Calculate the reward when the episode is truncated.

        Returns:
            float: The calculated truncated reward.
        """
        return self.calculate_reward() - self.truncated_penalty

    def calculate_penalty(self) -> float:
        """
        Calculate the penalty to subtract from the reward.

        Returns:
            float: The calculated penalty.
        """
        return 0


class MovementCost(BaseRewardWrapper):
    """
    Wraps the environment to adjust the reward based on movement cost and network throughput.
    Penalizes the agent for moving, encouraging it to quickly move to the optimum position and then stop moving.

    Attributes:
        env (Basic_MANETEnv): The environment to wrap.
        alpha (float): The factor to multiply the movement cost by.
    """

    def __init__(
        self, env: gym.Env, done_factor: float = 2, truncated_penalty: float = 0
    ) -> None:
        """
        Initialize the MovementCost wrapper.

        Args:
            env (Basic_MANETEnv): The environment to wrap.
            done_factor (float): Factor to multiply the reward by when the episode is done.
            truncated_penalty (float): Penalty to subtract from the reward when the episode is truncated.
        """
        super().__init__(env, done_factor, truncated_penalty)
        self.alpha: float = 0.2

    def calculate_reward(self) -> float:
        """
        Calculate the reward based on network throughput.

        Returns:
            float: The calculated reward.
        """
        reward = self._env.network.getThroughput() / 1e6
        return reward

    def calculate_penalty(self) -> float:
        """
        Calculate the penalty based on movement cost.

        Returns:
            float: The calculated penalty.
        """
        movement_cost = np.sum(
            np.linalg.norm(self._env.node_step_size * self._env.lastAction, axis=1)
        )
        penalty = self.alpha * movement_cost
        return penalty


class LinearReward(BaseRewardWrapper):
    """
    Wraps the environment to adjust the reward linearly based on network throughput.
    This is a normal reward function.

    Attributes:
        env (Basic_MANETEnv): The environment to wrap.
    """

    def calculate_reward(self) -> float:
        """
        Calculate the reward based on network throughput.

        Returns:
            float: The calculated reward.
        """
        reward = self._env.network.getThroughput() / 1e6
        return reward


class SquaredReward(BaseRewardWrapper):
    """
    Wraps the environment to adjust the reward based on the square of the network throughput.
    Increases the slope of the reward function.

    Attributes:
        env (Basic_MANETEnv): The environment to wrap.
    """

    def calculate_reward(self) -> float:
        """
        Calculate the reward based on the square of the network throughput.

        Returns:
            float: The calculated reward.
        """
        reward = (self._env.network.getThroughput() / 1e6) ** 2
        return reward


class EvaluationReward(BaseRewardWrapper):
    """
    Wraps the environment to provide a simple evaluation reward based on network throughput.
    This is practically the same as the LinearReward but is intended for evaluation purposes.

    Attributes:
        env (Basic_MANETEnv): The environment to wrap.
    """

    def calculate_reward(self) -> float:
        """
        Calculate the reward based on network throughput.

        Returns:
            float: The calculated reward.
        """
        reward = self._env.network.getThroughput() / 1e6
        return reward

    def calculate_done_reward(self) -> float:
        """
        Calculate the reward when the episode is done.

        Returns:
            float: The calculated done reward.
        """
        reward = self.calculate_reward()
        return reward

    def calculate_truncated_reward(self) -> float:
        """
        Calculate the reward when the episode is truncated.

        Returns:
            float: The calculated truncated reward.
        """
        reward = self.calculate_reward()
        return reward
