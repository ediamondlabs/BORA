import numpy as np

# Network Components
from .Components import MANETNode, User, Jammer

# Utility functions
from Utils import MANET_utils as ut
from abc import abstractmethod


"""
# Attackers.py
To model "realistic" attackers, different attacker models have been created in [Attackers.py](#MANET.Attackers). The attacker models control the jammers and bring them into a good position to start jamming. A jammer only starts jamming, once it was activated.

__Attackers__

There are three levels of attackers which differ in there knowledge about the network formed by the [Users](#MANET.Components.User) and [MANETNodes](#MANET.Components.MANETNode). So, it is differentiated between no knowledge, only knowledge about the topology and knowledge about the whole network and for each their is an AttackerModel with a strategy corresponding to its knowledge. 

Further, for all three `AttackerModels` they can either be `static` or `moving`, meaning that as soon as they start jamming, if they are `static`, they stop moving or if there are `moving` jammers the keep moving. However, the latter is obviously rather unrealistic in practice.

- [GreedyJammers](#MANET.Attackers.GreedyJammers): The `GreedyJammers` move towards the direction with the highest signal strength received. The jammers sample signals from multiple directions (N, NE, E, SE, ...) and choose the direction with the highest signal. They continue moving in that direction while periodically sampling adjacent directions to stay on the right path. `GreedyJammers` do not have knowledge of the network topology and find the best spot by sampling signals.
- [ClusterJammers](#MANET.Attackers.ClusterJammers): The `ClusterJammers` know of the network topology and want to jam the biggest user cluster. Hence, they move to the largest user clusters to jam them. The first jammer moves to the center of the largest user cluster, the second to the second largest, etc.
- [TrafficJammers](#MANET.Attackers.TrafficJammers): The `TrafficJammers` are supposed to know all about the networks topology. Hence, the try to jam the nodes that route the most data
"""


class AttackerModel:
    """
    Base class for attacker models.

    Attributes:
        nodes (np.ndarray): Array of MANET nodes.
        jammers (np.ndarray): Array of jammers in the network.
        users (np.ndarray): Array of users in the network.
        clusterSizes (np.ndarray): Array of cluster sizes.
        jammersAreActive (bool): Indicates whether the jammers are active.
        numbOfJammers(int): The number of jammers
    """

    def __init__(
        self,
        nodes: np.ndarray[MANETNode],
        jammers: np.ndarray[Jammer],
        users: np.ndarray[User],
        clusterSizes: np.ndarray,
        isStatic: bool = True,
    ) -> None:
        """
        Initialize the AttackerModel.

        Args:
            nodes (np.ndarray): Array of MANET nodes.
            jammers (np.ndarray): Array of jammers in the network.
            users (np.ndarray): Array of users in the network.
            clusterSizes (np.ndarray): Array of cluster sizes.
            isStatic (bool): Whether jammer is static or can move while jamming.
        """
        self.nodes = nodes
        self.jammers = jammers
        self.users = users
        self.clusterSizes = clusterSizes
        self.jammersAreActive = False
        self.isStatic = isStatic
        self.numbOfJammers = len(self.jammers)
        self.random_attacker_gen = np.random.default_rng(seed=102)

    def getMoves(self) -> np.ndarray:
        """Get the moves of the attackers. Depends on the kind of attackers, however this handles
        how the jammers are allowed to move if they are static and how if they are not.

        Returns:
            np.ndarray: the moves (2D-Arrays)
        """
        if self.jammersAreActive and self.isStatic:
            moves = np.zeros(shape=(self.numbOfJammers, 2), dtype=np.float32)
        else:
            moves = self._calculateMoves()

        return moves

    @abstractmethod
    def _calculateMoves(self) -> np.ndarray:
        """Get the next moves disregarding whether the jammers are static or not.

        Returns:
            np.ndarray: the moves (2D-Arrays)
        """
        pass

    def activateJammers(self) -> None:
        """
        Activate all jammers in the network.

        Returns:
            None
        """
        for jammer in self.jammers:
            jammer: Jammer
            jammer.isActive = True
        self.jammersAreActive = True

    def moveTowardsPositions(self, desiredPositions: np.ndarray) -> np.ndarray:
        """General method that given a set of desired position the jammers want to move to
        returns the necessary moves within the limits of the action space A = [-1,1]^2.

        Args:
            desiredPositions (np.ndarray): Positions the jammers want to move to.

        Returns:
            np.ndarray: Includes the move for each jammer to get to the desired position.
        """
        moves = []

        for i, jammer in enumerate(self.jammers):
            jammer: Jammer

            new_pos = desiredPositions[i % len(desiredPositions)]

            # Calculate the direction vector from jammer to the position
            direction = new_pos - jammer.pos

            # Normalize the direction vector
            dir_norm = np.linalg.norm(direction)
            move = (
                direction / dir_norm if dir_norm > 0 else np.zeros(2, dtype=np.float32)
            )
            moves.append(move)

        return np.array(moves)

    def set_seed(self, seed: int = None):
        self.random_attacker_gen = np.random.default_rng(seed=seed)


# * Greedy Jammers
class GreedyJammers(AttackerModel):
    """
    The GreedyJammers class inherits from the AttackerModel and implements a movement strategy where jammers
    move towards the direction with the highest signal strength received. The jammers sample signals from
    multiple directions (N, NE, E, SE, ...) and choose the direction with the highest signal. Once a direction is chosen, the
    jammers continue moving in that direction while periodically sampling adjacent directions to ensure they
    are on the right path.

    So, the GreedyJammers don't know anything about the network and hence, try to find the best spot by sampling the signals.

    Attributes:
        nodes (np.ndarray): Array of MANET nodes.
        users (np.ndarray): Array of users in the network.
        jammers (np.ndarray): Array of jammers in the network.
        clusterSizes (np.ndarray): Array of cluster sizes.
        isStatic (bool): Indicates whether the jammers are static or can move while jamming.
        numbOfJammers (int): Number of jammers in the network.
        operators (list): List of movement operators for each jammer.
        poss_moves (dict): Dictionary of possible moves for the jammers.
    """

    def __init__(
        self,
        nodes: np.ndarray,
        jammers: np.ndarray,
        users: np.ndarray,
        clusterSizes: np.ndarray,
        isStatic: bool = True,
    ) -> None:
        """
        Initialize the GreedJammers.

        Args:
            nodes (np.ndarray): Array of MANET nodes.
            jammers (np.ndarray): Array of jammers in the network.
            users (np.ndarray): Array of users in the network.
            clusterSizes (np.ndarray): Array of cluster sizes.
            isStatic (bool): Whether jammer is static or can move while jamming.
        """
        super().__init__(nodes, jammers, users, clusterSizes, isStatic)

        self.numbOfJammers = len(self.jammers)

        self.operators = []

        self.reset_operators()

        # Define relative positions to test from node's position
        self.poss_moves = {
            0: np.array([1.0, 0.0], dtype=np.float32),
            1: np.array([0.5, 0.5], dtype=np.float32),
            2: np.array([0.0, 1.0], dtype=np.float32),
            3: np.array([-0.5, 0.5], dtype=np.float32),
            4: np.array([-1.0, 0.0], dtype=np.float32),
            5: np.array([-0.5, -0.5], dtype=np.float32),
            6: np.array([0.0, -1.0], dtype=np.float32),
            7: np.array([0.5, -0.5], dtype=np.float32),
            8: np.array([0.0, 0.0], dtype=np.float32),
        }

    def _calculateMoves(self) -> np.ndarray:
        """Get the next moves disregarding whether the jammers are static or not.

        Returns:
            np.ndarray: the moves (2D-Arrays)
        """
        moves = np.array([next(operator) for operator in self.operators])
        return moves

    def reset_operators(self) -> None:
        """
        Resets the movement operators to ensure they start from the beginning.

        Returns:
            None
        """
        self.operators = [self.movement_operator(jammer) for jammer in self.jammers]

    def movement_operator(self, jammer: Jammer) -> np.ndarray:  # type: ignore
        """
        Controls the movement of the jammer. For the greedy jammer, first the jammer samples signals from all possible directions
        (only 8 are possible: E, NE, N, NW, W, SW, S, SE) and moves into the direction with the highest signal received (implying there is traffic being sent).
        After that the jammer continues going in that direction while sampling the adjacent directions to ensure the jammer is on the right path.

        Args:
            jammer (Jammer): Jammer being controlled by the movement operator.

        Yields:
            np.ndarray: Move to make.
        """
        # First sample all directions
        bestDir_idx = yield from self.sample_directions(jammer)

        # At the start we want to move, so no move is not allowed,
        # hence randomly sample move until it is not direction 8
        while bestDir_idx == 8:
            bestDir_idx = self.random_attacker_gen.choice(list(self.poss_moves.keys()))

        while True:
            while True:
                # Sample only adjacent to move quicker but also keep on the right path
                bestDir_idx = yield from self.sample_adjacent_directions(
                    jammer, bestDir_idx
                )
                # If direction returned is to stay static, then restart
                if bestDir_idx == 8:
                    break
                # Move 3 times in the best direction received from the adjacent sampling
                # Why 3 times? Balance of speed and agility!
                yield from self.move_in_direction(bestDir_idx)

            # First sample all directions
            bestDir_idx = yield from self.sample_directions(jammer)

            # Move in most promising directions
            yield from self.move_in_direction(bestDir_idx)

    def sample_directions(self, jammer: Jammer) -> np.ndarray:  # type: ignore
        """
        Samples signals from all possible directions (only 8 are possible: E, NE, N, NW, W, SW, S, SE) the jammer could move in.

        Args:
            jammer (Jammer): Malicious attacker.

        Returns:
            int: Index of best direction to move.

        Yields:
            np.ndarray: Move to make in 2D.
        """
        # Iniate values
        init_pos = jammer.pos.copy()
        samples = {8: jammer.signalReceived_mW}

        # last_pos relative position helps to move directly from one sample pos to the next
        last_pos = self.poss_moves[8]

        # sample all directions
        for i in self.poss_moves:
            if i == 8:
                continue
            yield self.move(last_pos, i)

            # store signal sample and store last_pos
            samples[i] = jammer.signalReceived_mW
            last_pos = self.poss_moves[i]

        # get best index for best direction/move in poss_moves
        bestDir_idx = max(samples, key=samples.get)

        # move to that direction seen as relative position from jammer origin/init_pos
        back_to_best_dir = self.poss_moves[bestDir_idx] + init_pos - jammer.pos
        yield back_to_best_dir

        return bestDir_idx

    def sample_adjacent_directions(self, jammer: Jammer, last_dir_idx: int) -> np.ndarray:  # type: ignore
        """
        Similar to sample_directions, but samples only left and right to current directions (e.g: jammer is heading N, then it samples NW, N and NE).
        To increase jammers speed.

        Args:
            jammer (Jammer): Malicious attacker.
            last_dir_idx (int): Index of the last direction.

        Returns:
            int: Index of best direction to move.

        Yields:
            np.ndarray: Move to make in 2D.
        """
        ## Iniate values
        init_pos = jammer.pos.copy()
        samples = {8: jammer.signalReceived_mW}

        # last_pos helps to move directly from one sample pos to the next
        last_pos = self.poss_moves[8]

        # indices of directions to sample (slightly-left, current, slightly-right)
        sample_dirs_idx = [
            (last_dir_idx - 1) % 8,
            last_dir_idx,
            (last_dir_idx + 1) % 8,
        ]

        # now actually sample them
        for i in sample_dirs_idx:
            yield self.move(last_pos, i)
            samples[i] = jammer.signalReceived_mW
            last_pos = self.poss_moves[i]

        # get the index of the best direction to move to
        best_dir_index = max(samples, key=samples.get)

        # move to that direction seen as relative position from jammer origin/init_pos
        back_to_dir = self.poss_moves[best_dir_index] + init_pos - jammer.pos
        yield back_to_dir

        return best_dir_index

    def move(self, last_pos: np.ndarray, i: int) -> np.ndarray:  # type: ignore
        """
        Move the jammer to the desired position.

        Args:
            last_pos (np.ndarray): Last position of the jammer.
            i (int): Index of the desired position.

        Returns:
            np.ndarray: Next move to make.
        """
        desired_position = self.poss_moves[i]
        next_move = desired_position - last_pos

        return next_move

    def move_in_direction(self, bestDir_idx: int) -> np.ndarray:  # type: ignore
        """
        Move in the best direction multiple times.

        Args:
            bestDir_idx (int): Index of the best direction.

        Yields:
            np.ndarray: Move to make.
        """
        bestDir = self.poss_moves[bestDir_idx]
        yield bestDir
        yield bestDir
        yield bestDir


class Static_GreedyJammers(GreedyJammers):
    """
    The Static_GreedyJammers class inherits from the GreedyJammers class and follows its movement strategy where jammers
    move towards the direction with the highest signal strength received. The jammers sample signals from
    multiple directions (N, NE, E, SE, ...) and choose the direction with the highest signal. Once a direction is chosen, the
    jammers continue moving in that direction while periodically sampling adjacent directions to ensure they
    are on the right path. However, once the jammers are activated, they stop moving.

    Attributes:
        nodes (np.ndarray): Array of MANET nodes.
        users (np.ndarray): Array of users in the network.
        jammers (np.ndarray): Array of jammers in the network.
        clusterSizes (np.ndarray): Array of cluster sizes.
    """

    def __init__(
        self,
        nodes: np.ndarray,
        jammers: np.ndarray,
        users: np.ndarray,
        clusterSizes: np.ndarray,
    ) -> None:
        """
        Initialize the Static_GreedyJammers.

        Args:
            nodes (np.ndarray): Array of MANET nodes.
            users (np.ndarray): Array of users in the network.
            jammers (np.ndarray): Array of jammers in the network.
            clusterSizes (np.ndarray): Array of cluster sizes.
        """
        super().__init__(nodes, jammers, users, clusterSizes, True)


class Moving_GreedyJammers(GreedyJammers):
    """
    The Moving_GreedyJammers class inherits from the GreedyJammers class and follows its movement strategy where jammers
    move towards the direction with the highest signal strength received. The jammers sample signals from
    multiple directions (N, NE, E, SE, ...) and choose the direction with the highest signal. Once a direction is chosen, the
    jammers continue moving in that direction while periodically sampling adjacent directions to ensure they
    are on the right path. Unlike Static_GreedyJammers, the jammers continue moving even after they are activated.

    Attributes:
        nodes (np.ndarray): Array of MANET nodes.
        users (np.ndarray): Array of users in the network.
        jammers (np.ndarray): Array of jammers in the network.
        clusterSizes (np.ndarray): Array of cluster sizes.
    """

    def __init__(
        self,
        nodes: np.ndarray,
        jammers: np.ndarray,
        users: np.ndarray,
        clusterSizes: np.ndarray,
    ) -> None:
        """
        Initialize the Moving_GreedyJammers.

        Args:
            nodes (np.ndarray): Array of MANET nodes.
            users (np.ndarray): Array of users in the network.
            jammers (np.ndarray): Array of jammers in the network.
            clusterSizes (np.ndarray): Array of cluster sizes.
        """
        super().__init__(nodes, jammers, users, clusterSizes, False)


# * Cluster Jammers
class ClusterJammers(AttackerModel):
    """The ClusterJammers know of the network topology and want to jam the biggest user cluster. Hence, they move
    to the largest user clusters to jam them.

    Attributes:
        nodes (np.ndarray): Array of MANET nodes.
        users (np.ndarray): Array of users in the network.
        jammers (np.ndarray): Array of jammers in the network.
        clusterSizes (np.ndarray): Array of cluster sizes.
        isStatic (bool): Indicates whether the jammers are static or can move while jamming.
        numbOfJammers (int): Number of jammers in the network.
        userClusters (list): List of lists including the User per cluster. E.g. [[User1,User2], [User3]]
    """

    def __init__(
        self,
        nodes: np.ndarray,
        jammers: np.ndarray,
        users: np.ndarray,
        clusterSizes: np.ndarray,
        isStatic: bool = True,
    ) -> None:
        """
        Initialize the ClusterJammers.

        Args:
            nodes (np.ndarray): Array of MANET nodes.
            jammers (np.ndarray): Array of jammers in the network.
            users (np.ndarray): Array of users in the network.
            clusterSizes (np.ndarray): Array of cluster sizes.
            isStatic (bool): Whether jammer is static or can move while jamming.
        """
        super().__init__(nodes, jammers, users, clusterSizes, isStatic)
        self.numbOfJammers = len(self.jammers)

        self.clusterSizes = clusterSizes
        self.numbOfClusters = len(self.clusterSizes)

        self.userClusters = ut.findUsersPerCluster(users, self.clusterSizes)

    # * Override
    def _calculateMoves(self) -> np.ndarray:
        """Get the moves that move each jammer closer to the cluster it wants to jammer, such that it is still in
        the action space A = [-1,1]^2. The first jammer moves to the center of the largest user cluster, the second
        to the second largest, etc.

        Returns:
            np.ndarray: the moves (2D-Arrays)
        """
        # Compute cluster centers directly
        # The user clusters are already descendingly sorted in size.
        clusterCenters = np.array(
            [
                np.mean([user.pos for user in cluster], axis=0)
                for cluster in self.userClusters
            ]
        )

        # Move towards the cluster centers
        moves = self.moveTowardsPositions(clusterCenters)
        return moves


class Static_ClusterJammers(ClusterJammers):
    """
    The Static_ClusterJammers class inherits from the ClusterJammers class and follows its movement strategy where jammers
    move towards the centers of the largest clusters. The jammers know the network topology and aim to jam the largest user clusters.
    Once the jammers are activated, they stop moving.

    Attributes:
        nodes (np.ndarray): Array of MANET nodes.
        users (np.ndarray): Array of users in the network.
        jammers (np.ndarray): Array of jammers in the network.
        clusterSizes (np.ndarray): Array of cluster sizes.
    """

    def __init__(
        self,
        nodes: np.ndarray,
        jammers: np.ndarray,
        users: np.ndarray,
        clusterSizes: np.ndarray,
    ) -> None:
        """
        Initialize the Static_ClusterJammers.

        Args:
            nodes (np.ndarray): Array of MANET nodes.
            users (np.ndarray): Array of users in the network.
            jammers (np.ndarray): Array of jammers in the network.
            clusterSizes (np.ndarray): Array of cluster sizes.
        """
        super().__init__(nodes, jammers, users, clusterSizes, True)


class Moving_ClusterJammers(ClusterJammers):
    """
    The Moving_ClusterJammers class inherits from the ClusterJammers class and follows its movement strategy where jammers
    move towards the centers of the largest clusters. The jammers know the network topology and aim to jam the largest user clusters.
    Unlike Static_ClusterJammers, the jammers continue moving even after they are activated.

    Attributes:
        nodes (np.ndarray): Array of MANET nodes.
        users (np.ndarray): Array of users in the network.
        jammers (np.ndarray): Array of jammers in the network.
        clusterSizes (np.ndarray): Array of cluster sizes.
    """

    def __init__(
        self,
        nodes: np.ndarray,
        jammers: np.ndarray,
        users: np.ndarray,
        clusterSizes: np.ndarray,
    ) -> None:
        """
        Initialize the Moving_ClusterJammers.

        Args:
            nodes (np.ndarray): Array of MANET nodes.
            users (np.ndarray): Array of users in the network.
            jammers (np.ndarray): Array of jammers in the network.
            clusterSizes (np.ndarray): Array of cluster sizes.
        """
        super().__init__(nodes, jammers, users, clusterSizes, False)


# * Traffic Jammers
class TrafficJammers(AttackerModel):
    """
    The TrafficJammers are supposed to know all about the networks topology. Hence, the try to jam the nodes that
    route the most data / host the most traffic.

    Attributes:
        nodes (np.ndarray): Array of MANET nodes.
        users (np.ndarray): Array of users in the network.
        jammers (np.ndarray): Array of jammers in the network.
        clusterSizes (np.ndarray): Array of cluster sizes.
        isStatic (bool): Whether the jammers are static (stop moving once acitvated) or not (keep moving).
    """

    # * Override
    def _calculateMoves(self) -> np.ndarray:
        """Get the moves that move each jammer closer to the node it wants to jammer, such that it is still in
        the action space A = [-1,1]^2. The first jammer moves to the node that routes the most data, the second
        to the one that routes the second most data, etc.

        Returns:
            np.ndarray: the moves (2D-Arrays)
        """
        # Sort nodes by how much data each hosts in descending order
        sorted_nodes_by_traffic = sorted(
            self.nodes, key=lambda node: node.trafficHosted, reverse=True
        )
        # Get only their positions
        node_positions = [node.pos for node in sorted_nodes_by_traffic]

        # Get the moves towards their position
        moves = self.moveTowardsPositions(node_positions)

        return moves


class Static_TrafficJammers(TrafficJammers):
    """
    The Static_TrafficJammers class inherits from the TrafficJammers class and follows its movement strategy where jammers
    move towards the nodes that route the most data or host the most traffic. The jammers know the network topology and aim to jam
    the nodes with the highest traffic. Once the jammers are activated, they stop moving.

    Attributes:
        nodes (np.ndarray): Array of MANET nodes.
        users (np.ndarray): Array of users in the network.
        jammers (np.ndarray): Array of jammers in the network.
        clusterSizes (np.ndarray): Array of cluster sizes.
    """

    def __init__(
        self,
        nodes: np.ndarray,
        jammers: np.ndarray,
        users: np.ndarray,
        clusterSizes: np.ndarray,
    ) -> None:
        """
        Initialize the Static_TrafficJammers.

        Args:
            nodes (np.ndarray): Array of MANET nodes.
            users (np.ndarray): Array of users in the network.
            jammers (np.ndarray): Array of jammers in the network.
            clusterSizes (np.ndarray): Array of cluster sizes.
        """
        super().__init__(nodes, jammers, users, clusterSizes, True)


class Moving_TrafficJammers(TrafficJammers):
    """
    The Moving_TrafficJammers class inherits from the TrafficJammers class and follows its movement strategy where jammers
    move towards the nodes that route the most data or host the most traffic. The jammers know the network topology and aim to jam
    the nodes with the highest traffic. Unlike Static_TrafficJammers, the jammers continue moving even after they are activated.

    Attributes:
        nodes (np.ndarray): Array of MANET nodes.
        users (np.ndarray): Array of users in the network.
        jammers (np.ndarray): Array of jammers in the network.
        clusterSizes (np.ndarray): Array of cluster sizes.
    """

    def __init__(
        self,
        nodes: np.ndarray,
        jammers: np.ndarray,
        users: np.ndarray,
        clusterSizes: np.ndarray,
    ) -> None:
        """
        Initialize the Moving_TrafficJammers.

        Args:
            nodes (np.ndarray): Array of MANET nodes.
            users (np.ndarray): Array of users in the network.
            jammers (np.ndarray): Array of jammers in the network.
            clusterSizes (np.ndarray): Array of cluster sizes.
        """
        super().__init__(nodes, jammers, users, clusterSizes, False)
