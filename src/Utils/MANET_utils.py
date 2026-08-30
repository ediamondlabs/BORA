import numpy as np

# Network X for network sim as graph
import networkx as nx

# Own Classes


def are_nodes_connected(nodes) -> bool:
    """
    Checks whether nodes are connected

    Args:
        nodes (list or np.ndarray): the nodes that should be connected

    Returns:
        are_connected (bool): Whether nodes are connected
    """
    # Create an empty undirected graph
    G = nx.Graph()

    # Add nodes to the graph
    for node in nodes:
        G.add_node(node)

    for node in nodes:
        for otherNode in nodes:
            if node.reaches(otherNode):
                G.add_edge(node, otherNode)

    # Check if the graph is connected
    return nx.is_connected(G)


def findUsersPerCluster(users: np.ndarray, clusterSizes: np.ndarray) -> list:
    """
    Distributes users into clusters based on the provided cluster sizes.

    Args:
        users (np.ndarray): An array of users to be distributed into clusters.
        clusterSizes (np.ndarray): An array where each element represents the size of a cluster.

    Returns:
        user_clusters (list): A list where each element is an array of users belonging to a specific cluster.
    """
    # Cumalitively sum up cluster sizes
    cum_sum_clusterSizes = np.cumsum(clusterSizes)

    # Create array for indices belonging to cluster
    user_idx_per_cluster = [
        np.arange(start, end)
        for start, end in zip(
            np.insert(cum_sum_clusterSizes[:-1], 0, 0), cum_sum_clusterSizes
        )
    ]

    # Create array with arrays of the users for each cluster
    user_clusters = [
        users[user_indices] for user_indices in user_idx_per_cluster
    ]

    return user_clusters
