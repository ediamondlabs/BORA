# Type hinting
from typing import Callable

# Matrix math
import numpy as np
from scipy.sparse import csr_array

# Graph network modelling
import networkx as nx

# Own MANET classes
from .Components import (
    MANETNode, Jammer, User, NetworkComponent,
    SPEED_OF_LIGHT, PATH_LOSS_EXPONENT,
)
from .Routing import MaxFlow

# ---------------------------------------------------------------------------
# Trust / robustness constants
# ---------------------------------------------------------------------------

# EMA coefficient for the per-node trust score update each step.
# Higher α → slower to react to fresh anomalies; lower α → faster but noisier.
_TRUST_ALPHA: float = 0.9

# Soft saturation threshold for unexplained interference (mW).
# Anomaly contribution from interference = x / (_INTERFERENCE_THRESHOLD + x).
_INTERFERENCE_THRESHOLD: float = 1e-3

"""
# Network.py
Models the network. Here, `Users` only send data over `MANETNodes` to limit interference. The network is modeled as a directed graph using [NetworkX](https://networkx.org/). Where each edge represents a possible communication route. 

__Routing__
Routing is done by considering all Users send at once. There are different routing algorithms that were implemented. Generally, the MCCMF formulation is the aim to use, to simulate perfect routing. This unfortunately can be slow in execution, thus mostly the sequential max flow algorithm --- and approximation of the MCCMF problem --- is used.

__Byzantine Fault Handling__
After routing, Byzantine attack effects are applied via `updateByzantineEffect()`. Any `ByzantineNode` in the network calculates the amount of traffic it drops (based on its `drop_rate` and `attack_type`), reducing the overall `networkThroughput` accordingly.
"""


class Network:
    """
    Models the network environment for the simulation.

    Attributes:
        nodes (np.ndarray): Array of MANET nodes in the network. May contain `ByzantineNode` instances that silently drop traffic after routing.
        jammers (np.ndarray): Array of jammers in the network.
        users (np.ndarray): Array of users in the network.
        graph (nx.DiGraph): Directed graph representing the network.
        np_random (np.random.Generator): Random number generator.
        networkThroughput (float): Total throughput of the network.
        capacity_matrix (np.ndarray): Matrix representing the capacities between network components.
        routing_func (Callable): Function determining how offered load is routed
    """

    def __init__(
        self,
        nodes: np.ndarray,
        jammers: np.ndarray,
        users: np.ndarray,
        np_random: np.random.Generator = np.random,
    ) -> None:
        """
        Initializes the network with nodes, jammers, users, and edges.

        Args:
            nodes (np.ndarray): Array of MANET nodes in the network.
            jammers (np.ndarray): Array of jammers in the network.
            users (np.ndarray): Array of users in the network.
            np_random (np.random.Generator, optional): Random number generator (default is np.random).

        Returns:
            None
        """
        self.nodes = nodes
        self.jammers = jammers
        self.users = users

        self.graph = nx.DiGraph()
        self.initializeGraph()

        # User random generator from environment if possible
        self.np_random = np_random

        self.networkThroughput = 0

        # Initialize capacity matrix
        self.capacity_matrix = None
        self.updateCapacityMatrix()

        # Routing function
        self.routing_func = MaxFlow.sequential_max_flow

    def initializeGraph(self) -> None:
        """
        Initializes the network graph with nodes and edges.

        Returns:
            None
        """
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_nodes_from(self.users)

    def updateNetwork(self) -> None:
        """
        Updates the network state.

        Returns:
            None
        """
        self.resetThroughput()
        self.resetValues()

        self.updateJammersEffect()
        self.updateByzantineJamming()
        self.updateSignals()
        self.updateEdges()
        self.updateCapacityMatrix()
        self.computeRobustnessSignals()
        self.updateTrust()

    def resetThroughput(self) -> None:
        """
        Resets the throughput of all network components.

        Returns:
            None
        """
        self.networkThroughput = 0

    def resetValues(self) -> None:
        """
        Resets all values, which all network components receive.

        Returns:
            None
        """
        for node in self.nodes:
            node: MANETNode
            node.resetAll()

        for user in self.users:
            user: User
            user.resetAll()

        for jammer in self.jammers:
            jammer: Jammer
            jammer.resetAll()

    def updateSignals(self) -> None:
        """
        Updates the signals between network components.

        Returns:
            None
        """
        for node in self.nodes:
            node: MANETNode
            node.transmitSignal(self.nodes)
            node.transmitSignal(self.users)
            node.transmitSignal(self.jammers)

        for user in self.users:
            user: User
            user.transmitSignal(self.nodes)
            user.transmitSignal(self.jammers)

    def updateJammersEffect(self) -> None:
        """
        Updates the effect of jammers on the network components.

        Returns:
            None
        """
        for jammer in self.jammers:
            jammer: Jammer
            jammer.jamComps(self.nodes)
            jammer.jamComps(self.users)

    def updateByzantineJamming(self) -> None:
        """
        Applies interference from selective_jamming Byzantine nodes.
        Called once per step before signal updates so that the elevated
        interference is reflected in Shannon capacity calculations.

        Returns:
            None
        """
        all_comps = np.concatenate((self.nodes, self.users))
        for node in self.nodes:
            if "selective_jamming" in getattr(node, "attack_types", frozenset()):
                node.jamNeighbors(all_comps)

    def updateByzantineEffect(self, flow_dict: dict) -> None:
        """
        Applies the Byzantine attack effects to the network throughput.
        Each Byzantine node calculates its own dropped traffic based on its attack type.

        Args:
            flow_dict (dict): The flow dictionary from the routing solver.

        Returns:
            None
        """
        for node in self.nodes:
            if getattr(node, 'is_byzantine', False):
                traffic = np.sum(list(flow_dict[node].values()))
                dropped = node.getDropped(traffic)
                self.networkThroughput -= dropped

    def updateEdges(self) -> None:
        """
        Updates the edges between network components.

        Returns:
            None
        """
        self.graph.clear_edges()
        self.addEdges()
        self.updateCapacityMatrix()

    def addEdges(self) -> None:
        """
        Adds edges between network components in the graph.

        Returns:
            None
        """
        self._addNodeEdges()
        self._addUserEdges()

    def _addNodeEdges(self) -> None:
        """
        Adds edges between nodes and other nodes/users.

        Returns:
            None
        """
        users_and_nodes = np.concatenate((self.users, self.nodes))
        for node in self.nodes:
            for comp in users_and_nodes:
                if node != comp:
                    self._addEdgeIfReaches(node, comp)

    def _addUserEdges(self) -> None:
        """
        Adds edges between users and nodes.

        Returns:
            None
        """
        for user in self.users:
            for node in self.nodes:
                self._addEdgeIfReaches(user, node)

    def _addEdgeIfReaches(
        self, comp1: NetworkComponent, comp2: NetworkComponent
    ) -> None:
        """
        Adds an edge between two components if the capacity is greater than zero.

        Args:
            comp1 (NetworkComponent): First network component.
            comp2 (NetworkComponent): Second network component.

        Returns:
            None
        """
        if comp1.reaches(comp2):
            capacity = comp1.calcCapacity(comp2)
            self.graph.add_edge(comp1, comp2, capacity=capacity)

    def updateCapacityMatrix(self) -> None:
        """Updates the capacity matrix to the most recent."""
        capacity_matrix: csr_array = nx.adjacency_matrix(
            G=self.graph,
            nodelist=np.concatenate((self.nodes, self.users)),
            dtype=int,
            weight="capacity",
        )
        self.capacity_matrix = capacity_matrix.toarray()

    def computeRobustnessSignals(self) -> None:
        """
        Computes per-node Byzantine robustness signals after physics have settled for the step.

        Three independent signals are written to each MANET node:

        signal_consistency: ratio of true signal power to the value the node advertises
          (observed_signal_power_mW). Sinkhole nodes inflate this value, so the ratio
          falls below 1.0. Always 1.0 for honest nodes.

        pos_consistency: agreement between a node's claimed (observed) position and the
          positions inferred from actual received signal strengths at its neighbours.
          Position-spoofing nodes shift observed_pos, causing the Friis-predicted signal
          to deviate from the measured one. Always 1.0 for honest nodes.

        unexplained_interference: interference received by a node beyond what all known
          external jammers account for. Selective-jamming Byzantine nodes raise this above
          zero for every component they can reach.

        All three signals are available as node attributes before updateTrust() is called.
        """
        epsilon = 1e-9
        wavelength = SPEED_OF_LIGHT / NetworkComponent.FREQUENCY

        for node in self.nodes:
            # --- signal_consistency ---
            observed_sp = getattr(node, "observed_signal_power_mW", node.signalPower_mW)
            node.signal_consistency = float(
                np.clip(node.signalPower_mW / (observed_sp + epsilon), 0.0, 1.0)
            )

            # --- pos_consistency ---
            # For each other node acting as a receiver, compare the signal actually
            # transmitted (real pos) against Friis predicted from the claimed (observed) pos.
            obs_pos = getattr(node, "observed_pos", node.pos)
            diffs = []
            for receiver in self.nodes:
                if receiver is node:
                    continue
                actual_rx = node.calcSignalTransmitted(receiver)
                if actual_rx < epsilon:
                    continue
                dist_claimed = float(np.linalg.norm(obs_pos - receiver.pos)) + 1e-6
                claimed_rx = node.signalPower_mW * (wavelength / (4 * np.pi)) ** 2 * np.float_power(
                    dist_claimed, -(2 + PATH_LOSS_EXPONENT)
                )
                claimed_rx = float(min(node.signalPower_mW, claimed_rx))
                diffs.append(abs(actual_rx - claimed_rx) / (actual_rx + epsilon))
            avg_diff = float(np.mean(diffs)) if diffs else 0.0
            node.pos_consistency = float(np.clip(1.0 - avg_diff, 0.0, 1.0))

            # --- unexplained_interference ---
            jammer_contrib = sum(j.calcSignalTransmitted(node) for j in self.jammers)
            node.unexplained_interference = float(
                max(0.0, node.interferenceReceived_mW - jammer_contrib)
            )

    def updateTrust(self) -> None:
        """
        Updates the per-node trust score using an exponential moving average (EMA).

        Each step the three robustness signals (signal_consistency, pos_consistency,
        unexplained_interference) are combined into a single anomaly score in [0, 1].
        The trust score is then updated:

            trust ← α · trust + (1 − α) · (1 − anomaly)

        where α = _TRUST_ALPHA (default 0.9).

        Trust of 1.0 means the node has shown no suspicious behaviour; 0.0 means
        consistently anomalous. The EMA smoothing prevents a single noisy measurement
        from permanently blacklisting an honest node.

        Must be called after computeRobustnessSignals() each step.
        """
        for node in self.nodes:
            anomaly_signal = float(np.clip(1.0 - node.signal_consistency, 0.0, 1.0))
            anomaly_pos = float(np.clip(1.0 - node.pos_consistency, 0.0, 1.0))
            # Soft saturation so that very large interference values don't dominate
            ui = node.unexplained_interference
            anomaly_interference = float(
                np.clip(ui / (_INTERFERENCE_THRESHOLD + ui), 0.0, 1.0)
            )
            anomaly = (anomaly_signal + anomaly_pos + anomaly_interference) / 3.0
            node.trust = float(
                _TRUST_ALPHA * node.trust + (1.0 - _TRUST_ALPHA) * (1.0 - anomaly)
            )

    def routeOfferedLoads(self) -> None:
        """
        Routes the offered loads between users.

        Returns:
            None
        """
        # Shuffle to not have preference in case of sequential max flow
        shuffled_users = self.np_random.permutation(self.users)
        # Set sour sink demand tuple
        source_sink_demands = [
            (user, user.dest, user.offeredLoad) for user in shuffled_users if user.dest
        ]

        # Remove users from graph without loss of generality, because flow shoudn't be routed
        # Over a user only sent and received from the users.
        sources_sinks = set(ss[0] for ss in source_sink_demands) | set(
            ss[1] for ss in source_sink_demands
        )
        graph_copy = self.graph.copy()
        graph_copy.remove_nodes_from(
            [user for user in self.users if user not in sources_sinks]
        )

        # Build a trust-scaled graph: outgoing edge capacities are multiplied by the
        # source node's trust score so the solver is naturally steered away from nodes
        # flagged as suspicious (trust < 1.0).  Honest nodes are unaffected (trust = 1.0).
        trust_graph = self.graph.copy()
        for node in self.nodes:
            trust = node.trust
            if trust < 1.0:
                for successor in list(trust_graph.successors(node)):
                    old_cap = trust_graph[node][successor].get("capacity", 0)
                    trust_graph[node][successor]["capacity"] = int(old_cap * trust)

        # Solve to get the maximum flow value with the flow dicts
        max_flow, flow_dict = self.routing_func(trust_graph, source_sink_demands)

        # Update traffic hosted 
        for node in self.nodes:
            node: MANETNode
            traffic = np.sum(list(flow_dict[node].values()))
            node.hostTraffic(traffic)

        # Update data delivered
        for user, _, _ in source_sink_demands:
            user: User
            data_sent = np.sum(list(flow_dict[user].values()))
            user.setDataDelivered(data_sent)

        # Update the throughput
        self.networkThroughput = max_flow

        # Apply Byzantine dropping effect
        self.updateByzantineEffect(flow_dict)

    # * Getters
    def getThroughput(self) -> float:
        """
        Returns the network throughput.

        Returns:
            float: Network throughput.
        """
        return self.networkThroughput

    def getMaxThroughput(self) -> float:
        """
        Returns the maximum throughput of the network.

        Returns:
            float: Maximum throughput of the network.
        """
        return np.sum([user.offeredLoad for user in self.users])

    def getGraph(self) -> nx.DiGraph:
        """
        Returns the graph.

        Returns:
            nx.DiGraph: Graph representing the network.
        """
        return self.graph

    def getCapacityMatrix(self) -> np.ndarray:
        """
        Returns the capacity matrix.

        Returns:
            np.ndarray: Capacity matrix.
        """
        return self.capacity_matrix

    # * Setters
    def set_routing_func(self, routing_func: Callable) -> None:
        """Set the routing function.

        Args:
            routing_func (Callable): Function that routes the offered loads.
        """

        self.routing_func = routing_func
