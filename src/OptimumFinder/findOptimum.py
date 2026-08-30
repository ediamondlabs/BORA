from typing import Tuple
from tabulate import tabulate
import numpy as np
from OptimumFinder.MINLP import findBestPositions
from OptimumFinder.ExhaustiveSearch import ExhaustiveSearcher
from evaluate_MANETAgents import BufferedEvaluationLogger, BufferedStateLogger


def findOptimumAndLog(
    log_folder: str, state_log_path: str, use: str = "ExhaustiveSearch"
) -> None:
    """
    Find the optimum node positions and log the results.

    Args:
        log_folder (str): Directory to store the log files.
        state_log_path (str): Path to the state log file.
        use (str, optional): Method to use for finding the optimum positions. Defaults to "ExhaustiveSearch".
    """

    def search(
        nodes: np.ndarray, jammers: np.ndarray, users: np.ndarray, use: str
    ) -> Tuple[float, np.ndarray]:
        """
        Search for the best node positions using the specified method.

        Args:
            nodes (np.ndarray): Array of MANET nodes.
            jammers (np.ndarray): Array of jammers.
            users (np.ndarray): Array of users.
            use (str): Method to use for finding the optimum positions.

        Returns:
            Tuple[float, np.ndarray]: Best throughput and corresponding node positions.
        """
        if use == "ExhaustiveSearch":
            searcher = ExhaustiveSearcher(
                nodes, jammers, users, gridResolution=10, refinementResolution=10
            )
            return searcher.startSearch()
        elif use == "MINLP":
            return findBestPositions(nodes, jammers, users)
        else:
            raise ValueError(f"Unknown search method: {use}")

    data = BufferedStateLogger.load_state(state_log_path)
    logger = BufferedEvaluationLogger(log_folder, buffer_size=1)

    for state in data:
        episode = state["episode"]
        nodes = state["nodes"]
        jammers = state["jammers"]
        users = state["users"]

        best_throughput, best_positions = search(nodes, jammers, users, use)

        logger.log_data(
            "Optimum",
            ep_step=0,
            episode=episode,
            value_name="throughput",
            value=best_throughput,
        )

        data = [
            ["Episode", f"{episode:<18}"],
            ["Best_throughput", f"{best_throughput/1e6:<10} Mbit"],
        ]

        print(tabulate(data, headers=["Parameter", "Value"], tablefmt="grid"))

    logger.close()
