import numpy as np

# For type declaration
from typing import Optional, Union, Generator
from abc import ABC, abstractmethod

# RL library
import torch as th

# Gymnasium
from gymnasium import spaces

# Stable Baselines 3
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv

# Own classes
from .Components import MANETNode
from .Environments import Basic_MANETEnv
import MANET.ConditionStrategy as cs

"""
# Agents.py

To compare the trained Agents to some baseline models, [Agents.py](#MANET.Agents) was created. Here, multiple agents have been created, and one can simply do so too by inheriting from [AgentBase](#MANET.Agents.AgentBase).

Since all Agents use a sampling process (what else would one expect), different sampling strategies have been created too. The agents themselves are not focused on what values to sample, rather how and how to react to the sampled values. So, the values to be sampled are interchangeable and that is made possible by using a [ConditionStrategy](ConditionStrategy.md)

__Agents__

There are, so far, five agents:

- [RandomAgent](#MANET.Agents.RandomAgent): Randomly chooses an action.
- [GreedyMaximizer](#MANET.Agents.GreedyMaximizer): Samples the values in all directions and moves for some steps in the direction with the best value.
- [OutAndThroughAgent](#MANET.Agents.OutAndThroughAgent): First moves away from noise or towards where any condition improves (out). If there is no large improvement, it moves in the other direction and checks there (through).
- [SpiralAgent](#MANET.Agents.SpiralSearch): Samples for the best position by moving in a spiral.
- [RandomSampler](#MANET.Agents.RandomSampler): Randomly chooses positions to sample and moves to the best one, including the ones taken while moving to those positions.
"""


class AgentBase(OnPolicyAlgorithm, ABC):
    """
    A blueprint to use to implement simple agents without using RL.

    Attributes:
        env (Union[GymEnv, str]): The environment to learn from (if registered in Gym, can be str).
        condition_strategy (cs.ConditionStrategy): The condition strategy to use.
        verbose (int): The verbosity level: 0 none, 1 training information, 2 debug.
        device (Union[th.device, str]): Device on which the code should be run. Can be a string or a torch.device.
        init_setup_model (bool): Whether to setup the model during initialization.
        nodes (list): List of MANET nodes.
        operators (list): List of movement operators.
        poss_moves (dict): Dictionary of possible moves.
        max_timeout (int): The maximum number of steps to wait before starting the sampling loop for each node. If not specified, it defaults to the number of nodes. The initial delay for each node is uniformly sampled from U({0, ..., max_timeout-1}).
    """

    def __init__(
        self,
        env: Union[GymEnv, str],
        condition_strategy: Union[
            cs.GlobalConditionStrategy, cs.LocalConditionStrategy
        ] = cs.SINRConditionStrategy(),
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        init_setup_model: bool = False,
        max_timeout: int = None,
    ) -> None:
        """
        Initialize the BaseAgent.

        Args:
            env (Union[GymEnv, str]): The environment to learn from (if registered in Gym, can be str).
            condition_strategy (cs.ConditionStrategy, optional): The condition strategy to use. Defaults to cs.SINRConditionStrategy().
            verbose (int, optional): The verbosity level: 0 none, 1 training information, 2 debug. Defaults to 0.
            device (Union[th.device, str], optional): Device on which the code should be run. Can be a string or a torch.device. Defaults to "auto".
            init_setup_model (bool, optional): Whether to setup the model during initialization. Defaults to False.
            max_timeout (int): The maximum number of steps to wait before starting the sampling loop for each node. If not specified, it defaults to the number of nodes. The initial delay for each node is uniformly sampled from U({0, ..., max_timeout-1}).
        """
        super().__init__(
            policy=None,
            env=env,
            learning_rate=0,
            n_steps=0,
            gamma=0,
            gae_lambda=0,
            ent_coef=0,
            vf_coef=0,
            max_grad_norm=0,
            use_sde=False,
            sde_sample_freq=0,
            rollout_buffer_class=None,
            rollout_buffer_kwargs=None,
            stats_window_size=0,
            tensorboard_log=None,
            policy_kwargs=None,
            verbose=verbose,
            device=device,
            seed=None,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        self.np_random_gen = np.random.default_rng(seed=101)  # Independent generator
        self.env: Basic_MANETEnv = env.unwrapped
        self.nodes: np.ndarray[MANETNode] = self.env.nodes
        self.max_timeout: int = self.nodes.size if max_timeout is None else max_timeout

        # Ensure condition_strategy is well setup
        if condition_strategy.is_global():
            self.condition_strategy: cs.GlobalConditionStrategy = condition_strategy

            if not self.condition_strategy.envExists():
                self.condition_strategy.setEnv(self.env)

        else:
            self.condition_strategy: cs.LocalConditionStrategy = condition_strategy

        self.operators = []

        self.reset_movement_operators()

        # Define relative positions to test from node's position (N, NE, E, SE, S, ...)
        self.poss_moves = {
            "up": np.array([0.0, 1.0], dtype=np.float32),
            "up-left": np.array([-0.5, 0.5], dtype=np.float32),
            "left": np.array([-1.0, 0.0], dtype=np.float32),
            "down-left": np.array([-0.5, -0.5], dtype=np.float32),
            "down": np.array([0.0, -1.0], dtype=np.float32),
            "down-right": np.array([0.5, -0.5], dtype=np.float32),
            "right": np.array([1.0, 0.0], dtype=np.float32),
            "up-right": np.array([0.5, 0.5], dtype=np.float32),
        }

    def train(self):
        """
        Train the agent. This method is not used for the RandomAgent as it does not learn.
        """
        pass

    def reset_movement_operators(self) -> None:
        """
        Resets the movement operators to ensure they start from the beginning.
        """
        self.nodes = self.env.nodes
        self.operators = [self.movementOperator(node) for node in self.nodes]

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple:
        """
        Collects the predicted action of each node's movement operator and returns it as a tuple of the actions and None.

        Args:
            observation (np.ndarray): The current observation of the environment.
            state (Optional[np.ndarray], optional): The last states (can be None, used in recurrent policies). Defaults to None.
            episode_start (Optional[np.ndarray], optional): The last episode start signal. Defaults to None.
            deterministic (bool, optional): Whether or not to return deterministic actions. Defaults to False.

        Returns:
            tuple: The action to take and the next state (here None).
        """
        if self.env.ep_step == 0:
            self.reset_movement_operators()
        action = np.array(
            [next(operator) for operator in self.operators], dtype=np.float32
        )
        return action, None

    def movementOperator(self, node: MANETNode) -> "Generator[np.ndarray, None, None]":
        """
        A generator function that determines the movement sequence for a given MANET node.

        This function first applies a timeout period during which the node remains stationary.
        After the timeout, it delegates the movement logic to the `customMovementOperator` method,
        which must be implemented by subclasses to define specific movement strategies.

        Args:
            node (MANETNode): The MANET node for which the movement sequence is being generated.

        Yields:
            np.ndarray: A 2D vector representing the movement action for the node at each step.
            The vector values are constrained to the range [-1, 1].
        """
        if self.max_timeout > 0:
            timeout = self.np_random_gen.integers(0, self.max_timeout)
        else:
            timeout = 0  # or handle as appropriate

        for _ in range(timeout):
            yield np.array([0, 0], dtype=np.float32)

        yield from self.customMovementOperator(node=node)

    @abstractmethod
    def customMovementOperator(self, node: MANETNode) -> np.ndarray:
        """
        Abstract method to define the custom movement logic for a MANET node.

        Subclasses must implement this method to specify how a node decides its movement
        actions based on the environment and the condition strategy. The method should
        yield a sequence of 2D movement vectors, where each vector represents a single
        movement step.

        Args:
            node (MANETNode): The MANET node for which the movement logic is being defined.

        Yields:
            np.ndarray: A 2D vector representing the movement action for the node at each step.
            The vector values are constrained to the range [-1, 1].
        """
        pass

    def setConditionStrat(self, condition_strategy: cs.ConditionStrategy) -> None:
        """
        Set the condition strategy for the agent.

        Args:
            condition_strategy (cs.ConditionStrategy): The condition strategy to be set for the agent.

        Returns:
            None
        """
        self.condition_strategy = condition_strategy

    def getName(self) -> str:
        """
        Get the name of the agent, including the name of the condition strategy.

        Returns:
            str: The name of the agent with its condition strategy.
        """
        strategy_name = self.condition_strategy.__class__.__name__
        return f"{self.__class__.__name__}_{strategy_name}"


class RandomAgent(AgentBase):
    """
    A SB3 compatible agent that chooses actions randomly by sampling the action space. So, each action is chosen completly at random.
    """

    # * Override
    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple:
        """
        Predict the next action given the current observation.

        Args:
            observation (np.ndarray): The current observation of the environment.
            state (Optional[np.ndarray], optional): The last states (can be None, used in recurrent policies).
            episode_start (Optional[np.ndarray], optional): The last episode start signal.
            deterministic (bool, optional): Whether or not to return deterministic actions.

        Returns:
            tuple: The action to take and the next state (here None).
        """
        action = np.array([self.action_space.sample()])[0]
        return action, None


class OutAndThroughAgent(AgentBase):
    """
    The agent first tries to move away from the jammer's interference signal and,
    if it doesn't find a position, it attempts to move through the jammer's interference out the other side to find more viable position.

    In more detail:

        1. It will sample in eight directions (N, NE, E, SE, S, ...)
        2. It will move away in the most promising direction (initially, away from the interference, but other conditions strategies can be applied too).
        3. However, if this moving away didn't help much it will try to move out the other way by moving into the opposite direction for twice as far. So, it turns around and tries to go through the jammer.
        4. After going through, it will choose the best position from all so far sampled positions.
    """

    def __init__(
        self,
        env: Union[GymEnv, str],
        condition_strategy: cs.ConditionStrategy = cs.SINRConditionStrategy(),
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        numb_of_sampling_steps: int = None,
        max_timeout: int = None,
    ) -> None:
        """
        Initialize the OutAndThroughAgent.

        Args:
            env (Union[GymEnv, str]): The environment to learn from (if registered in Gym, can be str).
            condition_strategy (cs.ConditionStrategy, optional): The condition strategy to be used by the agent.
            verbose (int, optional): The verbosity level: 0 none, 1 training information, 2 debug.
            device (Union[th.device, str], optional): Device on which the code should be run. Can be a string or a torch.device.
            numb_of_sampling_steps (int): The number of steps to move away from the jammers.
            max_timeout (int): The maximum number of steps to wait before starting the sampling loop for each node. If not specified, it defaults to the number of nodes. The initial delay for each node is uniformly sampled from U({0, ..., max_timeout-1}).

        Returns:
            None
        """
        super().__init__(
            env=env,
            condition_strategy=condition_strategy,
            verbose=verbose,
            device=device,
            init_setup_model=False,
            max_timeout=max_timeout,
        )

        self.numb_of_sampling_steps = (
            int(self.env.SIZE / 8)
            if numb_of_sampling_steps is None
            else numb_of_sampling_steps
        )

    # * Override
    def customMovementOperator(self, node: MANETNode) -> np.ndarray:  # type: ignore
        """
        Operates one node, meaning it decides what actions to take for the node given.
        In this case the decisions are based on the out and through policy, which is that if the node's condition
        becomes rather unfeasible it will move away from the jamming interference first, sample for better positions and either, if no position improves the condition move
        the opposite way and through the node and sample again, or move into the best position according to the condition.

        Args:
            node (MANETNode): MANET node

        Yields:
            np.ndarray: Action for the node to take, which is a 2D-vector in [-1,1]^2
        """
        while True:
            if self.condition_strategy.is_unviable(node):
                # * Probe for directions
                samples = {}
                init_pos = node.pos.copy()
                last_pos = np.zeros(
                    2, dtype=np.float32
                )  # position relative to node's initial position

                for desired_pos in self.poss_moves.values():
                    next_move = desired_pos - last_pos
                    yield next_move

                    last_pos = desired_pos
                    samples[tuple(desired_pos)] = self.condition_strategy.sample(node)

                # Move back to starting pos
                back_to_init_pos = init_pos - node.pos
                yield back_to_init_pos

                # Find the best direction based on the condition samples to improve the state
                bestDir = np.array(max(samples, key=samples.get))

                # * Sample based on condition for best position by moving into best direction
                samples = {}
                sample = self.condition_strategy.sample(node)
                samples[tuple(node.pos)] = sample

                for _ in range(self.numb_of_sampling_steps):
                    next_pos = node.pos + bestDir
                    # Check if new position would be out of bounds
                    out_of_bounds = np.any((next_pos < 0) | (next_pos > self.env.SIZE))
                    if out_of_bounds:
                        break
                    yield bestDir

                    sample = self.condition_strategy.sample(node)
                    samples[tuple(node.pos)] = sample

                bestPos = max(samples, key=samples.get)

                not_has_viable_pos = self.condition_strategy.value_is_unviable(
                    samples[bestPos]
                )

                if not_has_viable_pos:
                    for _ in range(2 * self.numb_of_sampling_steps):
                        # Check if each position is out of bounds
                        position = node.pos - bestDir
                        out_of_bounds = np.any(
                            (position < 0) | (position > self.env.SIZE)
                        )
                        if out_of_bounds:
                            break
                        yield -bestDir

                        sample = self.condition_strategy.sample(node)
                        samples[tuple(node.pos)] = sample

                bestPos = max(samples, key=samples.get)

                # * move to best position
                movesToPos = getStepsTo(current_pos=node.pos, dest=bestPos)
                for move in movesToPos:
                    yield move
            else:
                # Don't move its position if it is still okay
                yield np.array([0.0, 0.0], dtype=np.float32)


class RandomSampler(AgentBase):
    """
    This agent randomly normally chooses a number of positions around it and moves to these to sample
    their viability provided by the ConditionStrategy. It will too sample all the positions on its way.
    In the end it chooses the most viable one of all those sampled positions and repeats. The random positions are distributed randomly normally with mu = node's position and
    cov = playground size divided by ten.
    """

    def __init__(
        self,
        env: Union[GymEnv, str],
        condition_strategy: cs.ConditionStrategy = cs.SINRConditionStrategy(),
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        numb_of_samples: int = 3,
        max_timeout: int = None,
    ) -> None:
        """
        Initialize the RandomSampler.

        Args:
            env (Union[GymEnv, str]): The environment to learn from (if registered in Gym, can be str).
            condition_strategy (cs.ConditionStrategy, optional): The condition strategy to be used by the agent.
            verbose (int, optional): The verbosity level: 0 none, 1 training information, 2 debug.
            device (Union[th.device, str], optional): Device on which the code should be run. Can be a string or a torch.device.
            numb_of_samples (int): The number of random samples to take for each sampling round. Defaults to 3.
            max_timeout (int): The maximum number of steps to wait before starting the sampling loop for each node. If not specified, it defaults to the number of nodes. The initial delay for each node is uniformly sampled from U({0, ..., max_timeout-1}).

        Returns:
            None
        """
        super().__init__(
            env=env,
            condition_strategy=condition_strategy,
            verbose=verbose,
            device=device,
            init_setup_model=False,
            max_timeout=max_timeout,
        )
        self.numb_of_samples: int = numb_of_samples

    # * Override
    def customMovementOperator(self, node) -> np.ndarray:  # type: ignore
        """
        Operates one node, meaning it decides what actions to take for the node given.
        Here, if the condition becomes unviable, the node chooses random multivariate gaussian distributed points to go to and sample. After sampling these 10 positions
        plus all the positions on the way to those positions for their viability it moves the most viable position.

        Args:
            node (MANETNode): MANET node

        Yields:
            np.array: Action for the node to take, which is a 2D-vector in [-1,1]^2
        """
        while True:
            if self.condition_strategy.is_unviable(node):
                # * Randomly determine points to sample
                mean = node.pos
                cov = np.diag([self.env.SIZE / 10, self.env.SIZE / 10])

                positions_to_sample = self.np_random_gen.multivariate_normal(
                    mean=mean, cov=cov, size=self.numb_of_samples
                )
                positions_to_sample = np.clip(positions_to_sample, 0, self.env.SIZE)

                # * Sample points and all positions on their way
                samples = {}

                for dest in positions_to_sample:
                    movesToPos = getStepsTo(
                        current_pos=node.pos, dest=dest
                    )  # returns the moves that are in the action space in order to get to the destination
                    for move in movesToPos:
                        yield move

                        sample_value = self.condition_strategy.sample(node)
                        samples[tuple(node.pos)] = sample_value

                # * Go to the best of all of those positions
                bestPos = max(samples, key=samples.get)
                movesToPos = getStepsTo(current_pos=node.pos, dest=bestPos)
                for move in movesToPos:
                    yield move
            else:
                yield np.array([0.0, 0.0], dtype=np.float32)


class GreedyMaximizer(AgentBase):
    """
    Greedily maximizes the sampled values according to the condition strategy by:

        1. Sampling values in all eight directions (N, NE, E, SE, E, ..)
        2. Moving twice into the best direction (Why twice? Because, after the sampling of the eight directions it will first go back to the initial position and then move in the best direction)
        3. Repeat
    """

    # * Override
    def customMovementOperator(self, node) -> np.ndarray:  # type: ignore
        """
        Operates one node, meaning it decides what actions to take for the node given.
        Here, when the condition becomes infeasible, it will start moving in most viable direction,
        which is determined by sampling in eight directions (N, NE, E, SE, S, ...). And then it repeats.

        Args:
            node (MANETNode): MANET node

        Yields:
            np.array: Action for the node to take, which is a 2D-vector in [-1,1]^2
        """
        while True:
            if self.condition_strategy.is_unviable(node):
                # * Probe for directions
                samples = {}

                init_pos = node.pos.copy()
                last_pos = np.zeros(
                    2, dtype=np.float32
                )  # relative to nodes actual position

                for desired_pos in self.poss_moves.values():
                    next_move = desired_pos - last_pos
                    yield next_move

                    last_pos = desired_pos
                    samples[tuple(desired_pos)] = self.condition_strategy.sample(node)

                # *  Move back to initial start of the sampling
                back_to_init_pos = init_pos - node.pos
                yield back_to_init_pos

                samples[tuple((0, 0))] = self.condition_strategy.sample(node)

                bestDir = np.array(max(samples, key=samples.get), dtype=np.float32)

                noMove = np.array([0, 0], dtype=np.float32)

                if np.array_equal(bestDir, noMove):
                    yield np.array([0, 0], dtype=np.float32)
                else:
                    # * move  once into the best direction
                    yield bestDir

            else:
                # Don't move sampled position is fine
                yield np.array([0, 0], dtype=np.float32)


class SpiralSearch(AgentBase):
    """
    The agent follows a spiral path outward from its initial position to systematically sample for a viable position.
    With radius of Playground size divided by 10.
    """

    def __init__(
        self,
        env: Union[GymEnv, str],
        condition_strategy: cs.ConditionStrategy = cs.SINRConditionStrategy(),
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        max_radius: int = None,
        max_timeout: int = None,
    ) -> None:
        """
        Initialize the SpiralSearch Agent.

        Args:
            env (Union[GymEnv, str]): The environment to learn from (if registered in Gym, can be str).
            condition_strategy (cs.ConditionStrategy, optional): The condition strategy to be used by the agent.
            verbose (int, optional): The verbosity level: 0 none, 1 training information, 2 debug.
            device (Union[th.device, str], optional): Device on which the code should be run. Can be a string or a torch.device.
            max_radius (int): The maximum radius for the spiral
            max_timeout (int): The maximum number of steps to wait before starting the sampling loop for each node. If not specified, it defaults to the number of nodes. The initial delay for each node is uniformly sampled from U({0, ..., max_timeout-1}).

        Returns:
            None
        """
        super().__init__(
            env=env,
            condition_strategy=condition_strategy,
            verbose=verbose,
            device=device,
            init_setup_model=False,
            max_timeout=max_timeout,
        )

        self.max_radius = self.env.SIZE / 10 if max_radius is None else max_radius

    # * Override
    def customMovementOperator(self, node) -> np.ndarray:  # type: ignore
        """
        Operates one node, meaning it decides what actions to take for the node given.
        Here, when the position becomes unviable, it performs a spiral search to find the most viable position and moves the agent to that position.

        Args:
            node (MANETNode): MANET node

        Yields:
            np.array: Action for the node to take, which is a 2D-vector in [-1,1]^2
        """
        angle_step = (
            np.pi / 8
        )  # Define how much to turn each step of the spiral (smaller value = tighter spiral)
        distance_step = 1  # Distance to move in each step of the spiral

        samples = {}  # To store SNR values for different positions

        while True:
            if self.condition_strategy.is_unviable(node):
                current_angle = 0
                current_radius = 0
                radius_increment = 0
                init_pos = node.pos.copy()

                # * Perform spiral search
                while current_radius <= self.max_radius:
                    # Calculate the movement direction in the spiral
                    dx = distance_step * np.cos(current_angle)
                    dy = distance_step * np.sin(current_angle)

                    # Apply the move in steps of [dx, dy]
                    num_moves = int(radius_increment / distance_step)
                    for _ in range(num_moves):
                        # Ensure the move step is within bounds
                        move = np.clip(np.array([dx, dy]), -1, 1)

                        # Yield the movement
                        yield move

                        # Record SNR value at this position
                        sampled_value = self.condition_strategy.sample(node)
                        samples[tuple(node.pos)] = sampled_value

                    current_angle += angle_step
                    if current_angle >= 2 * np.pi:
                        current_angle = 0
                        radius_increment += distance_step

                    current_radius = np.linalg.norm(node.pos - init_pos)

                # * After spiral search, move to the position with the maximum sampled_value
                best_pos = max(samples, key=samples.get)
                moves_to_best_pos = getStepsTo(
                    node.pos, np.array(best_pos, dtype=np.float32)
                )
                for move in moves_to_best_pos:
                    # Ensure the move is within action space
                    move = np.clip(move, -1, 1)
                    yield move

                # Clear the samples and reset the spiral
                samples.clear()

            else:
                # If the SNR is viable, yield [0, 0] to stay in place
                yield np.array([0.0, 0.0], dtype=np.float32)


# * Helper methods
def getStepsTo(current_pos, dest) -> np.ndarray:
    """
    Returns an array of directions with norm not larger than 1, that lead from the current_pos of an object to
    the destination.

    Args:
        current_pos (np.array): Current position of an object.
        dest (np.array): Destination of where the object wants to end up.

    Returns:
        np.array: List of moves to do to get to the destination.
    """
    # Calculate the direction vector and its norm (distance)
    direct_to_best_pos = dest - current_pos
    dist = np.linalg.norm(direct_to_best_pos)

    # Normalize the direction vector (handle zero distance)
    normalized_vector = direct_to_best_pos / dist if dist != 0 else direct_to_best_pos

    # Generate an array with ones and the fractional part (if non-zero)
    stepsSizes = np.ones(int(dist))
    if dist % 1 != 0:
        stepsSizes = np.concatenate((stepsSizes, [dist % 1]))

    stepsToGoal = np.array(
        [normalized_vector * stepSize for stepSize in stepsSizes],
        dtype=np.float32,
    )

    return stepsToGoal
