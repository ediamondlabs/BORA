# Finding the log paths
import os
import re
import glob
from pathlib import Path

# Type hinting
from typing import Tuple, Union

# Data manipulation
import numpy as np
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Helper methods
import Utils.training_utils as ut


class LogLoader:
    """
    Class to load the evaluation logs, such that they can be analyzed easily.
    """

    def __init__(self) -> None:
        """Constructor"""
        self.log_paths: list[str] = None
        self.grouped_log_paths: dict = None
        self.group: list = None

    # * Load the log files
    def loadLogs(self, *log_patterns: str) -> None:
        """
        Get all log paths that match the log patterns provided.

        Args:
            log_patterns (str): pattern specifying the log files.
        """
        self.log_paths = findLogFilesFromPattern(*log_patterns)

    def groupLogPaths(
        self,
        byNodes: bool = False,
        byJammers: bool = False,
        byUsers: bool = False,
        byStateLossRate: bool = False,
        byAttackerModel: bool = False,
        byEnv: bool = False,
        distinguish_baselines: bool = False,
        byName: bool = False,
    ):
        """Group the logs by the specified fields.

        Args:
            byNodes (bool, optional): Whether to group it by the number of nodes. Defaults to False.
            byJammers (bool, optional): Whether to group it by the number of jammers. Defaults to False.
            byUsers (bool, optional): Whether to group it by the number of users. Defaults to False.
            distinguish_baselines (bool, optional): Whether to group it by whether it is an RL-Agent or a Baseline. Defaults to False.
            byName (bool, optional): Whether to group it by the name of the agent/baseline. Defaults to False.
        """
        self.grouped_log_paths = {}

        for path in self.log_paths:
            # Get values to group by from the path the evaluations were stored to.
            parts = path.split("/")
            agent_base_name = parts[-1]

            nodes, jammers, users = None, None, None

            path = Path(path)
            config = ut.loadConfig(path.parent, "20_evaluation_config.yaml")

            nodes = config.get("eval_params", {}).get("numbOfNodes", nodes)
            nodes = tuple(nodes) if isinstance(nodes, list) else nodes
            jammers = config.get("eval_params", {}).get("numbOfJammers", jammers)
            jammers = tuple(jammers) if isinstance(jammers, list) else jammers
            users = config.get("eval_params", {}).get("numbOfUsers", users)
            users = tuple(users) if isinstance(users, list) else users

            state_loss_rate = config.get("eval_params", {}).get("state_loss_rate", None)
            eval_type = config.get("eval_params", {}).get("obs_type", None)
            if eval_type == "full":
                state_loss_rate = 0

            attackerModel = config.get("eval_params", {}).get("attackerModel", None)
            env = config.get("env_constr", None)
            # Create dictionary key

            # Create dictionary key
            key = []
            group = []

            if byNodes:
                key.append(nodes)
                group.append("nodes")
            if byJammers:
                key.append(jammers)
                group.append("jammers")
            if byUsers:
                key.append(users)
                group.append("users")

            if byStateLossRate:
                key.append(state_loss_rate)
                group.append("state_loss_rate")
            if byAttackerModel:
                key.append(attackerModel)
                group.append("attackerModel")
            if byEnv:
                key.append(env)
                group.append("env")

            if distinguish_baselines:
                # Check if the name contains "PPO" or numeric patterns (indicating an agent)
                is_agent = "PPO" in agent_base_name or re.search(r"\d", agent_base_name)
                key.append("Agent" if is_agent else "Baseline")
                group.append("agent_or_base")
            if byName:
                key.append(str(agent_base_name))
                group.append("agent_base_name")

            key = tuple(key)

            self.grouped_log_paths[key] = path
            self.group: list = group


class LogDataProcessor:
    """
    Used to process the logged data.

    Attributes:
        all_data_df (pd.DataFrame): DataFrame containing all the loaded data.
        cached_data_df (pd.DataFrame): Cached copy of the loaded data for quick access.
    """

    def __init__(self) -> None:
        """
        Constructor
        """
        self.all_data_df: pd.DataFrame = None
        self.cached_data_df: pd.DataFrame = None

    def loadGroupedDataFrame(self, group: list[str], grouped_log_paths: dict) -> None:
        """Loads the grouped logs into a dataframe based on the way they were grouped.

        Args:
            group (list[str]): Should be the names of the elements in the key of the grouped_log_paths dictionary. E.g. key=(2,0,2), group=("nodes", "jammers", "users")
            grouped_log_paths (dict): The grouped log paths.
        """
        data_frames = []

        # Load each evaluation log
        for key, path in grouped_log_paths.items():
            key: tuple
            df = pd.read_csv(path)

            # Scale throughput values by 10^6 to convert from bits to megabits
            throughput_mask = df["value_name"] == "throughput"
            if throughput_mask.any():
                df.loc[throughput_mask, "value"] *= 1e6

            # Add grouping to the dataframe
            for element, element_name in zip(key, group):
                if isinstance(
                    element, tuple
                ):  # Check if the element is a tuple (range)
                    # Example: Represent the range as a string
                    df[element_name] = f"{element[0]}-{element[1]}"
                    # Alternatively, you could expand the range into multiple rows or handle it differently
                else:
                    df[element_name] = element
            data_frames.append(df)

        self.all_data_df = pd.concat(data_frames, ignore_index=True)
        self.cached_data_df = self.all_data_df.copy()

    def resetCache(self):
        self.cached_data_df = self.all_data_df.copy()

    def use_test_df(self, ep_range: Tuple[int, int] = (0, 100)) -> None:
        self.filter_for_episode_interval(interval=ep_range)

    def filter_for_episode_interval(self, interval: Tuple[int, int] = (0, 100)) -> None:
        low_bound, high_bound = interval
        self.cached_data_df = self.cached_data_df[
            (self.cached_data_df["episode"] >= low_bound)
            & (self.cached_data_df["episode"] <= high_bound)
        ]

    def filter_for_step_interval(self, interval: Tuple[int, int] = (0, 1000)) -> None:
        low_bound, high_bound = interval
        self.cached_data_df = self.cached_data_df[
            (self.cached_data_df["ep_step"] >= low_bound)
            & (self.cached_data_df["ep_step"] <= high_bound)
        ]

    def filter_custom_by_numb_jammers(self, interval: Tuple[int, int] = (1, 4)) -> None:
        low_bound, high_bound = interval
        self.cached_data_df = self.cached_data_df[
            (self.cached_data_df["jammers"] >= low_bound)
            & (self.cached_data_df["jammers"] <= high_bound)
        ]

    # * Calculating stats functions
    # Filter for one value
    def getStatsForValue(self, value_name: str) -> pd.DataFrame:
        """Filter dataframe for a specific value recorded.

        Args:
            value_name (str): Name of the value to filter for.

        Returns:
            pd.DataFrame: DataFrame only including rows with the value given the value_name.
        """
        return self.cached_data_df[self.cached_data_df["value_name"] == value_name]

    def filterByJammerEpSteps(self, jammer_ep_steps: int) -> None:
        """
        Filters the cached DataFrame to include only rows where:
        - jammers == 0
        - OR ep_step >= jammer_ep_steps (if jammers >= 1)

        Args:
            jammer_ep_steps (int): The minimum number of steps for jammers. If 0, no filtering is applied.
        """
        if jammer_ep_steps > 0 and self.cached_data_df is not None:
            self.cached_data_df = self.cached_data_df[
                (self.cached_data_df["jammers"] == 0)
                | (self.cached_data_df["ep_step"] >= jammer_ep_steps)
            ]

    # Episode Means
    def getGroupedEpisodeMeansForValue(
        self, group_by_group: list[str], value_name: str
    ) -> pd.DataFrame:
        """Get the mean values grouped by episodes and additional groups.

        Args:
            group_by_group (list[str]): List of columns to group by.
            value_name (str): Name of the value to calculate means for.

        Returns:
            pd.DataFrame: DataFrame with mean values grouped by specified columns.
        """
        group_by_group = ["episode"] + group_by_group
        return self.getGroupedMeansForValue(group_by_group, value_name)

    def getGroupedEpisodeStabilizedMeansForValue(
        self,
        group_by_group: list[str],
        value_name: str,
        stabilize_value: str = "throughput",
        start_step: int = None,
        end_step: int = None,
    ) -> pd.DataFrame:
        """Get the stabilized mean values grouped by episodes and additional groups.

        Args:
            group_by_group (list[str]): List of columns to group by.
            value_name (str): Name of the value to calculate stabilized means for.
            start_step (int, optional): Start step for analysis. If None, starts from the first step. Defaults to None.
            end_step (int, optional): End step for analysis. If None, continues to the last step. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame with stabilized mean values grouped by specified columns.
        """
        group_by_group = ["episode"] + group_by_group
        return self.getGroupedStabilizedMeansForValue(
            group_by_group,
            value_name,
            stabilize_value,
            start_step=start_step,
            end_step=end_step,
        )

    def getGroupedEpisodeStabilizedStepsForValue(
        self, group_by_group: list[str], value_name: str
    ) -> pd.DataFrame:
        """Get the stabilized steps grouped by episodes and additional groups.

        Args:
            group_by_group (list[str]): List of columns to group by.
            value_name (str): Name of the value to calculate stabilized steps for.

        Returns:
            pd.DataFrame: DataFrame with stabilized steps grouped by specified columns.
        """
        group_by_group = ["episode"] + group_by_group
        return self.getGroupedStabilizedStepsForValue(group_by_group, value_name)

    # Means for any group
    def getGroupedMeansForValue(
        self, group_by_group: list[str], value_name: str
    ) -> pd.DataFrame:
        """Get the mean values grouped by specified columns.

        Args:
            group_by_group (list[str]): List of columns to group by.
            value_name (str): Name of the value to calculate means for.

        Returns:
            pd.DataFrame: DataFrame with mean values grouped by specified columns.
        """
        df_value = self.getStatsForValue(value_name=value_name)
        return df_value.groupby(group_by_group)["value"].mean().reset_index()

    def getGroupedStabilizedMeansForValue(
        self,
        group_by_group: list[str],
        value_name: str,
        stabilize_value: str = "throughput",
        start_step: int = None,
        end_step: int = None,
    ) -> pd.DataFrame:
        """Get the stabilized mean values grouped by specified columns.

        Args:
            group_by_group (list[str]): List of columns to group by.
            value_name (str): Name of the value to calculate stabilized means for.
            stabilize_value (str, optional): Value name to use for stabilization. Defaults to "throughput".
            start_step (int, optional): Start step for analysis. If None, starts from the first step. Defaults to None.
            end_step (int, optional): End step for analysis. If None, continues to the last step. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame with stabilized mean values grouped by specified columns.
        """
        df_value = self.getStatsForValue(value_name=value_name)
        # Which value to consider for stabilization
        stabilize_value_df = self.getStatsForValue(value_name=stabilize_value)

        # Apply step filtering if specified
        if start_step is not None:
            df_value = df_value[df_value["ep_step"] >= start_step]
            stabilize_value_df = stabilize_value_df[
                stabilize_value_df["ep_step"] >= start_step
            ]
        if end_step is not None:
            df_value = df_value[df_value["ep_step"] <= end_step]
            stabilize_value_df = stabilize_value_df[
                stabilize_value_df["ep_step"] <= end_step
            ]

        stabilized_data = []
        for (group_values, group), (stabilize_group_values, stabilize_group) in zip(
            df_value.groupby(group_by_group), stabilize_value_df.groupby(group_by_group)
        ):
            # Set the index to ep_step before creating the series
            stabilize_series = stabilize_group.set_index("ep_step")["value"]
            # Get step
            stabilization_step, has_stablized = find_stabilizing_index(
                stabilize_series, epsilon_factor=0.05
            )
            stabilization_step = (
                stabilization_step if has_stablized else stabilize_series.index[0]
            )

            # Filter data to include only stable steps
            stable_data = group[group["ep_step"] >= stabilization_step]

            stabilized_data.append(stable_data)

        # Concatenate data
        stabilized_df = pd.concat(stabilized_data)
        return stabilized_df.groupby(group_by_group)["value"].mean().reset_index()

    def getGroupedStabilizedStepsForValue(
        self, group_by_group: list[str], value_name: str
    ) -> pd.DataFrame:
        """Get the stabilized steps grouped by specified columns.

        Args:
            group_by_group (list[str]): List of columns to group by.
            value_name (str): Name of the value to calculate stabilized steps for.

        Returns:
            pd.DataFrame: DataFrame with stabilized steps grouped by specified columns.
        """
        df_value = self.getStatsForValue(value_name=value_name)

        results = []
        for group_values, group in df_value.groupby(group_by_group):
            # Get series to check for stabilization
            throughput_series = group.set_index("ep_step")["value"]

            # Get step
            stabilization_step, has_stablized = find_stabilizing_index(
                throughput_series
            )
            stabilization_step = (
                stabilization_step if has_stablized else throughput_series.index[-1]
            )

            results.append(
                {
                    **dict(zip(group_by_group, group_values)),
                    "value_name": "stabilization_step",
                    "value": stabilization_step,
                }
            )
        self.cached_data_df = pd.DataFrame(results)
        df_only_stabilization_step = self.getStatsForValue(
            value_name="stabilization_step"
        )
        return df_only_stabilization_step

    def getStatsForValueByNode(self, value_name: str) -> pd.DataFrame:
        self.cached_data_df = self.getStatsForValue(value_name)
        self.cached_data_df["value"] = (
            self.cached_data_df["value"] / self.cached_data_df["nodes"]
        )
        return self.cached_data_df

    def getGroupedAverageDistanceTravelled(
        self,
        group_by_group: list[str],
    ):
        self.cached_data_df.loc[
            self.cached_data_df["value_name"] == "distance", "value"
        ] /= self.cached_data_df["nodes"]

        # Compute cumulative sum within each episode and group_by_group
        self.cached_data_df["cum_distance"] = (
            self.cached_data_df[self.cached_data_df["value_name"] == "distance"]
            .groupby(["episode"] + group_by_group)["value"]
            .cumsum()
        )

        # Create a copy of the distance rows for cumulative distance
        cum_distance_rows = self.cached_data_df.copy()
        cum_distance_rows["value_name"] = "cum_distance"  # Rename value_name
        cum_distance_rows["value"] = cum_distance_rows[
            "cum_distance"
        ]  # Set cumulative value

        # Append the new rows to the original DataFrame
        self.cached_data_df = pd.concat(
            [self.cached_data_df, cum_distance_rows], ignore_index=True
        )

        # Drop the temporary 'cum_distance' column (since it's now in 'value')
        self.cached_data_df = self.cached_data_df.drop(columns=["cum_distance"])

        # Sort to maintain order
        self.cached_data_df = self.cached_data_df.sort_values(
            by=["episode"] + group_by_group + ["value_name"]
        ).reset_index(drop=True)

        return self.cached_data_df

    def getTotalDistancePerEpisode(self, group_by_group: list[str]):
        # Filter only cumulative distance values
        grouped_df = self.getStatsForValue("cum_distance")

        # Get the last entry per episode and group
        self.cached_data_df = grouped_df.groupby(
            ["episode"] + group_by_group, as_index=False
        ).last()

        return self.cached_data_df

    def clear(self) -> None:
        """Clear all loaded data."""
        self.all_data_df: pd.DataFrame = None
        self.cached_data_df: pd.DataFrame = None

    def clear_cache(self) -> None:
        """Clear the cached data."""
        self.cached_data_df = None


class DataVisualizer:
    """
    Class to visualize the data from the logs.

    Attributes:
        log_loader (LogLoader): Instance of LogLoader to load the logs.
        log_data_processor (LogDataProcessor): Instance of LogDataProcessor to process the loaded logs.
    """

    def __init__(self, *log_patterns: str) -> None:
        """
        Constructor

        Args:
            log_patterns (str): Patterns specifying the log files.
        """
        self.log_loader = LogLoader()
        self.log_loader.loadLogs(*log_patterns)
        self.log_data_processor = LogDataProcessor()

    def _set_theme(self) -> None:

        sns.set_theme(style="whitegrid", context="paper")
        sns.set_palette("Set1")

    # * Create certain plots
    def plotStepsMeanForValue(self, value_name: str) -> None:
        """
        Plots the mean for a value at each step over all episodes with a 95% confidence interval.

        Args:
            value_name (str): Name of the value for which the mean should be calculated (e.g. throughput).
        """
        # Group data
        self.log_loader.groupLogPaths(
            byNodes=True, distinguish_baselines=True, byJammers=True
        )
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )

        # Calculate average distance
        self.log_data_processor.getGroupedAverageDistanceTravelled(
            self.log_loader.group
        )
        df_stepMeanForValue = self.log_data_processor.getStatsForValue(value_name)

        plt.figure()
        self._set_theme()
        # Manually set a high-contrast color palette for "nodes"
        unique_nodes = df_stepMeanForValue[
            "nodes"
        ].nunique()  # Get the unique number of nodes
        custom_palette = sns.color_palette(
            "Set1", unique_nodes
        )  # "Set1" creates distinct hues

        sns.lineplot(
            data=df_stepMeanForValue,
            x="ep_step",
            y="value",
            hue="nodes",
            errorbar=None,
            style="agent_or_base",
            markers=True,
            dashes=False,
            palette=custom_palette,
            markevery=200,
        )

        plt.title(f"Mean {value_name.capitalize()} per episode step")
        plt.xlabel("Step")
        plt.ylabel(f"Mean {value_name.capitalize()}")

        self.log_data_processor.clear_cache()

    # * Create certain plots
    def plotStepsMeanForValueByStateLossRate(
        self, interval: Tuple[int, int] = (0, 2000)
    ) -> None:
        """
        Plots the mean for a value at each step over all episodes with a 95% confidence interval.

        Args:
            value_name (str): Name of the value for which the mean should be calculated (e.g. throughput).
        """
        # Group data
        self.log_loader.groupLogPaths(
            byStateLossRate=True, byJammers=True, distinguish_baselines=True
        )
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )

        self.log_data_processor.filter_for_step_interval(interval=interval)

        df_stepForValue = self.log_data_processor.getStatsForValue("throughput")
        # Create a copy of the baseline entries with state_loss_rate set to "Baseline"
        baseline_df = df_stepForValue[
            df_stepForValue["agent_or_base"] == "Baseline"
        ].copy()
        baseline_df["state_loss_rate"] = "Baseline"
        # Keep only agent entries in the original dataframe
        agent_df = df_stepForValue[df_stepForValue["agent_or_base"] == "Agent"]
        # Combine agent and baseline dataframes
        df_stepForValue = pd.concat([agent_df, baseline_df])

        df_stepForValue = df_stepForValue[df_stepForValue["jammers"] == 1]

        self._init_new_figure(with_legend_space_below=True)
        self._set_theme()
        sns.lineplot(
            data=df_stepForValue,
            x="ep_step",
            y="value",
            errorbar=("ci", 95),
            hue="state_loss_rate",
            markers=True,
            dashes=False,
            palette=sns.color_palette("hls", 8),
            markevery=50,
            linewidth=2,
            legend=True,
        )

        # Update axis labels with LaTeX

        # Customize the legend
        handles, labels = plt.gca().get_legend_handles_labels()

        # Filter out the titles for "state_loss_rate"
        filtered_handles = []
        filtered_labels = []

        def fraction_to_latex(x: float) -> str:
            if x == 0.0:
                return "1"
            if x == 1.0:
                return "0"

            i = 1
            while i < 10:  # arbitrary upper bound
                if x == 1 - 1 / (2**i):

                    return rf"2^{{-{i}}}"
                i += 1
            return str(x)

        # Collect delta values and their corresponding handles
        delta_items = []
        for handle, label in zip(handles, labels):
            if label not in ["state_loss_rate"]:  # Skip the titles
                if label == "DE":
                    filtered_labels.append(rf"{label}")
                    filtered_handles.append(handle)
                elif label == "Baseline":
                    filtered_labels.append(r"GTM")
                    filtered_handles.append(handle)
                else:
                    # Extract numeric delta values for sorting
                    try:
                        delta_value = float(label)
                        delta_items.append(
                            (
                                delta_value,
                                handle,
                                rf"$\delta = {fraction_to_latex(delta_value)}$",
                            )
                        )
                    except ValueError:
                        # Handle non-numeric labels (if any)
                        filtered_labels.append(label)
                        filtered_handles.append(handle)

        # Sort delta items by numeric value
        delta_items.sort(key=lambda x: x[0])

        # Append sorted delta values to the filtered lists
        for _, handle, formatted_label in delta_items:
            filtered_handles.append(handle)
            filtered_labels.append(formatted_label)
        # Create the legend without the subtitle
        plt.legend(
            filtered_handles,
            filtered_labels,
            ncol=4,  # Two columns
            title=None,  # Remove legend title
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            labelspacing=0.5,
            columnspacing=0.5,  # Adjust horizontal spacing between columns
        )

        # Apply scientific notation
        apply_scientific_notation(plt.gca())

        # Update axis labels with LaTeX
        plt.xlabel(r"$t$")  # step
        plt.ylabel(r"$\tau \, \text{[bit/s]}$")

        # Set the title but make it invisible
        plt.title("state_loss_rate_by_step")
        plt.gca().title.set_visible(False)
        self.log_data_processor.clear_cache()

    def plotStateLossRollingWindow(
        self, interval: Tuple[int, int] = (0, 2000), window_size: int = 4
    ) -> None:
        """
        Plots the mean for a value at each step over all episodes with a 95% confidence interval.

        Args:
            value_name (str): Name of the value for which the mean should be calculated (e.g. throughput).
        """
        # Group data
        self.log_loader.groupLogPaths(
            byStateLossRate=True, byJammers=True, distinguish_baselines=True
        )
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )

        self.log_data_processor.filter_for_step_interval(interval=interval)

        df_stepForValue = self.log_data_processor.getStatsForValue("throughput")
        # Create a copy of the baseline entries with state_loss_rate set to "Baseline"
        baseline_df = df_stepForValue[
            df_stepForValue["agent_or_base"] == "Baseline"
        ].copy()
        baseline_df["state_loss_rate"] = "Baseline"
        # Keep only agent entries in the original dataframe
        agent_df = df_stepForValue[df_stepForValue["agent_or_base"] == "Agent"]
        # Combine agent and baseline dataframes
        df_stepForValue = pd.concat([agent_df, baseline_df])

        df_stepForValue = df_stepForValue[df_stepForValue["jammers"] == 1]

        # OPTIMIZED ROLLING WINDOW CALCULATION
        # Sort by grouping columns and ep_step for efficient rolling operations
        group_by_group = ["episode", "agent_or_base", "state_loss_rate"]
        df_stepForValue = df_stepForValue.sort_values(group_by_group + ["ep_step"])

        # Use pandas groupby with rolling - much faster than manual iteration
        df_stepForValue["smoothed_value"] = (
            df_stepForValue.groupby(group_by_group)["value"]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
            .reset_index(level=group_by_group, drop=True)  # Remove multi-index
        )

        # Replace the original "value" column with the smoothed values
        df_stepForValue["value"] = df_stepForValue["smoothed_value"]
        df_stepForValue.drop(columns=["smoothed_value"], inplace=True)

        self._init_new_figure(with_legend_space_below=True)
        self._set_theme()
        sns.lineplot(
            data=df_stepForValue,
            x="ep_step",
            y="value",
            errorbar=("ci", 95),
            hue="state_loss_rate",
            markers=True,
            dashes=False,
            palette=sns.color_palette("hls", 8),
            markevery=50,
            linewidth=2,
            legend=True,
        )

        # Update axis labels with LaTeX

        # Customize the legend
        handles, labels = plt.gca().get_legend_handles_labels()

        # Filter out the titles for "state_loss_rate"
        filtered_handles = []
        filtered_labels = []

        def fraction_to_latex(x: float) -> str:
            if x == 0.0:
                return "1"
            if x == 1.0:
                return "0"

            i = 1
            while i < 10:  # arbitrary upper bound
                if x == 1 - 1 / (2**i):

                    return rf"2^{{-{i}}}"
                i += 1
            return str(x)

        # Collect delta values and their corresponding handles
        delta_items = []
        for handle, label in zip(handles, labels):
            if label not in ["state_loss_rate"]:  # Skip the titles
                if label == "DE":
                    filtered_labels.append(rf"{label}")
                    filtered_handles.append(handle)
                elif label == "Baseline":
                    filtered_labels.append(r"GTM")
                    filtered_handles.append(handle)
                else:
                    # Extract numeric delta values for sorting
                    try:
                        delta_value = float(label)
                        delta_items.append(
                            (
                                delta_value,
                                handle,
                                rf"$\delta = {fraction_to_latex(delta_value)}$",
                            )
                        )
                    except ValueError:
                        # Handle non-numeric labels (if any)
                        filtered_labels.append(label)
                        filtered_handles.append(handle)

        # Sort delta items by numeric value
        delta_items.sort(key=lambda x: x[0])

        # Append sorted delta values to the filtered lists
        for _, handle, formatted_label in delta_items:
            filtered_handles.append(handle)
            filtered_labels.append(formatted_label)
        # Create the legend without the subtitle
        plt.legend(
            filtered_handles,
            filtered_labels,
            ncol=4,  # Two columns
            title=None,  # Remove legend title
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            labelspacing=0.5,
            columnspacing=0.5,  # Adjust horizontal spacing between columns
        )

        # Apply scientific notation
        apply_scientific_notation(plt.gca())

        # Update axis labels with LaTeX
        plt.xlabel(r"$t$")  # step
        plt.ylabel(r"$\tau \, \text{[bit/s]}$")

        # Set the title but make it invisible
        plt.title("state_loss_rate_by_step_smoothed")
        plt.gca().title.set_visible(False)
        self.log_data_processor.clear_cache()

    def plotStepsMeanSmall(
        self,
        jammer_ep_step: int = 0,
    ) -> None:
        """
        Plots the mean for a value at each step over all episodes with a 95% confidence interval.

        Args:
            value_name (str): Name of the value for which the mean should be calculated (e.g. throughput).
        """
        # Group data
        self.log_loader.groupLogPaths(
            byStateLossRate=True, byJammers=True, distinguish_baselines=True
        )
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )

        self.log_data_processor.filter_for_step_interval(
            interval=(jammer_ep_step, jammer_ep_step + 40)
        )

        df_stepForValue = self.log_data_processor.getStatsForValue("throughput")

        # Create a copy of the baseline entries with state_loss_rate set to "Baseline"
        baseline_df = df_stepForValue[
            df_stepForValue["agent_or_base"] == "Baseline"
        ].copy()
        baseline_df["state_loss_rate"] = "Baseline"
        # Keep only agent entries in the original dataframe
        agent_df = df_stepForValue[df_stepForValue["agent_or_base"] == "Agent"]
        # Combine agent and baseline dataframes
        df_stepForValue = pd.concat([agent_df, baseline_df])

        df_stepForValue = df_stepForValue[df_stepForValue["jammers"] == 1]

        self._init_new_figure(with_legend_space_below=True)
        self._set_theme()

        sns.lineplot(
            data=df_stepForValue,
            x="ep_step",
            y="value",
            errorbar=("ci", 95),
            hue="state_loss_rate",
            markers=True,
            dashes=False,
            palette=sns.color_palette("hls", 8)[0:],
            markevery=5,
            linewidth=2,
            legend=True,
        )

        # # Update axis labels with LaTeX

        # Customize the legend
        handles, labels = plt.gca().get_legend_handles_labels()

        # Filter out the titles for "state_loss_rate"
        filtered_handles = []
        filtered_labels = []

        def fraction_to_latex(x: float) -> str:
            if x == 0.0:
                return "1"
            if x == 1.0:
                return "0"

            i = 1
            while i < 10:  # arbitrary upper bound
                if x == 1 - 1 / (2**i):

                    return rf"2^{{-{i}}}"
                i += 1
            return str(x)

        # Collect delta values and their corresponding handles
        delta_items = []
        for handle, label in zip(handles, labels):
            if label not in ["state_loss_rate"]:  # Skip the titles
                if label == "DE":
                    filtered_labels.append(rf"{label}")
                    filtered_handles.append(handle)
                elif label == "Baseline":
                    filtered_labels.append(r"GTM")
                    filtered_handles.append(handle)
                else:
                    # Extract numeric delta values for sorting
                    try:
                        delta_value = float(label)
                        delta_items.append(
                            (
                                delta_value,
                                handle,
                                rf"$\delta = {fraction_to_latex(delta_value)}$",
                            )
                        )
                    except ValueError:
                        # Handle non-numeric labels (if any)
                        filtered_labels.append(label)
                        filtered_handles.append(handle)

        # Sort delta items by numeric value
        delta_items.sort(key=lambda x: x[0])

        # Append sorted delta values to the filtered lists
        for _, handle, formatted_label in delta_items:
            filtered_handles.append(handle)
            filtered_labels.append(formatted_label)
        # Create the legend without the subtitle
        plt.legend(
            filtered_handles,
            filtered_labels,
            ncol=3,
            title=None,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            labelspacing=0.5,
            columnspacing=0.5,
        )

        # Apply scientific notation
        apply_scientific_notation(plt.gca())

        # Update axis labels with LaTeX
        plt.xlabel(r"$t$")  # step
        plt.ylabel(r"$\tau \, \text{[bit/s]}$")

        # Set the title but make it invisible
        plt.title("mini_step")
        plt.gca().title.set_visible(False)

        self.log_data_processor.clear_cache()

    def plotStepsMeanSmallRollingWindow(
        self, jammer_ep_step: int = 0, window_size: int = 4
    ) -> None:
        """
        Plots the mean for a value at each step over all episodes with a 95% confidence interval.

        Args:
            value_name (str): Name of the value for which the mean should be calculated (e.g. throughput).
        """
        # Group data
        self.log_loader.groupLogPaths(
            byStateLossRate=True, byJammers=True, distinguish_baselines=True
        )
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )

        self.log_data_processor.filter_for_step_interval(
            interval=(jammer_ep_step, jammer_ep_step + 40)
        )

        df_stepForValue = self.log_data_processor.getStatsForValue("throughput")

        # Create a copy of the baseline entries with state_loss_rate set to "Baseline"
        baseline_df = df_stepForValue[
            df_stepForValue["agent_or_base"] == "Baseline"
        ].copy()
        baseline_df["state_loss_rate"] = "Baseline"
        # Keep only agent entries in the original dataframe
        agent_df = df_stepForValue[df_stepForValue["agent_or_base"] == "Agent"]
        # Combine agent and baseline dataframes
        df_stepForValue = pd.concat([agent_df, baseline_df])

        df_stepForValue = df_stepForValue[df_stepForValue["jammers"] == 1]

        # OPTIMIZED ROLLING WINDOW CALCULATION
        # Sort by grouping columns and ep_step for efficient rolling operations
        group_by_group = ["episode", "agent_or_base", "state_loss_rate"]
        df_stepForValue = df_stepForValue.sort_values(group_by_group + ["ep_step"])

        # Use pandas groupby with rolling - much faster than manual iteration
        df_stepForValue["smoothed_value"] = (
            df_stepForValue.groupby(group_by_group)["value"]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
            .reset_index(level=group_by_group, drop=True)  # Remove multi-index
        )

        # Replace the original "value" column with the smoothed values
        df_stepForValue["value"] = df_stepForValue["smoothed_value"]
        df_stepForValue.drop(columns=["smoothed_value"], inplace=True)

        self._init_new_figure(with_legend_space_below=True)
        self._set_theme()

        sns.lineplot(
            data=df_stepForValue,
            x="ep_step",
            y="value",
            errorbar=("ci", 95),
            hue="state_loss_rate",
            markers=True,
            dashes=False,
            palette=sns.color_palette("hls", 8)[0:],
            markevery=5,
            linewidth=2,
        )

        # Update axis labels with LaTeX

        # Customize the legend
        handles, labels = plt.gca().get_legend_handles_labels()

        # Filter out the titles for "state_loss_rate"
        filtered_handles = []
        filtered_labels = []

        def fraction_to_latex(x: float) -> str:
            if x == 0.0:
                return "1"
            if x == 1.0:
                return "0"

            i = 1
            while i < 10:  # arbitrary upper bound
                if x == 1 - 1 / (2**i):

                    return rf"2^{{-{i}}}"
                i += 1
            return str(x)

        # Collect delta values and their corresponding handles
        delta_items = []
        for handle, label in zip(handles, labels):
            if label not in ["state_loss_rate"]:  # Skip the titles
                if label == "DE":
                    filtered_labels.append(rf"{label}")
                    filtered_handles.append(handle)
                elif label == "Baseline":
                    filtered_labels.append(r"GTM")
                    filtered_handles.append(handle)
                else:
                    # Extract numeric delta values for sorting
                    try:
                        delta_value = float(label)
                        delta_items.append(
                            (
                                delta_value,
                                handle,
                                rf"$\delta = {fraction_to_latex(delta_value)}$",
                            )
                        )
                    except ValueError:
                        # Handle non-numeric labels (if any)
                        filtered_labels.append(label)
                        filtered_handles.append(handle)

        # Sort delta items by numeric value
        delta_items.sort(key=lambda x: x[0])

        # Append sorted delta values to the filtered lists
        for _, handle, formatted_label in delta_items:
            filtered_handles.append(handle)
            filtered_labels.append(formatted_label)
        # Create the legend without the subtitle
        plt.legend(
            filtered_handles,
            filtered_labels,
            ncol=3,  # Two columns
            title=None,  # Remove legend title
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            labelspacing=0.5,
            columnspacing=0.5,  # Adjust horizontal spacing between columns
        )

        # Apply scientific notation
        apply_scientific_notation(plt.gca())

        # Update axis labels with LaTeX
        plt.xlabel(r"$t$")  # step
        plt.ylabel(r"$\tau \, \text{[bit/s]}$")

        # Set the title but make it invisible
        plt.title("mini_step_smooth")
        plt.gca().title.set_visible(False)

        self.log_data_processor.clear_cache()

    # * Create certain plots
    def plotMeanForValueByStateLossRateAsHistogramm(self) -> None:
        """
        Plots the mean for a value at each step over all episodes with a 95% confidence interval.

        Args:
            value_name (str): Name of the value for which the mean should be calculated (e.g. throughput).
        """
        # Group data
        self.log_loader.groupLogPaths(byStateLossRate=True, byJammers=True)
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )

        # Get grouped stabilized throughput
        group = self.log_loader.group
        df_stabilized_throughput = (
            self.log_data_processor.getGroupedEpisodeStabilizedMeansForValue(
                group, "throughput"
            )
        )

        df_stabilized_throughput = df_stabilized_throughput[
            df_stabilized_throughput["jammers"] == 1
        ]

        self._init_new_figure()
        self._set_theme()

        # sns.histplot(
        #     df_stabilized_throughput,
        #     x="value",
        #     hue="state_loss_rate",
        #     # edgecolor=".3",
        #     # linewidth=0.5,
        #     legend=True,
        # )

        sns.stripplot(
            data=df_stabilized_throughput,
            x="value",
            hue="state_loss_rate",
            dodge=True,
            alpha=0.25,
            zorder=1,
            legend=True,
        )

        # Update axis labels with LaTeX
        plt.xlabel(r"$\tau \, \text{[bit/s]}$")  # Number of nodes

        # # Customize the legend
        # handles, labels = plt.gca().get_legend_handles_labels()
        # # Filter out the titles for "agent_or_base" and "jammers"
        # filtered_handles = []
        # filtered_labels = []
        # for handle, label in zip(handles, labels):
        #     if label not in ["state_loss_rate"]:  # Skip the titles
        #         filtered_labels.append(
        #             rf"$\delta = {label}$"
        #         )  # Replace jammers with |\mathcal{J}|
        #         filtered_handles.append(handle)  # Keep the corresponding handle

        # # Create the legend without the subtitle
        # plt.legend(
        #     filtered_handles,
        #     filtered_labels,
        #     ncol=2,  # Two columns
        #     title=None,  # Remove legend title
        # )

        # Apply scientific notation
        apply_scientific_notation(plt.gca())

        plt.ylabel(r"$count$")

        self.log_data_processor.clear_cache()

    def plotNormalizedGain(self) -> None:
        """
        Plots the normalized gain from 0 jammers to 1 jammer.

        Returns:
            None
        """
        # Group data
        self.log_loader.groupLogPaths(True, True, True, True)
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )

        # Get stabilized througput
        group = self.log_loader.group
        df_stabilized_throughput = (
            self.log_data_processor.getGroupedEpisodeStabilizedMeansForValue(
                group, "throughput"
            )
        )

        # Get normalized gain
        df_normalized_gain = self._getNormalizedGain(df_stabilized_throughput)

        # Plot using Seaborn
        plt.figure()
        self._set_theme()
        sns.boxplot(
            x="nodes",
            y="normalized_gain",
            hue="agent_or_base",
            data=df_normalized_gain,
        )

        plt.title("Normalized Gain from 0 Jammers to 1 Jammer")
        plt.xlabel("Number of Nodes")
        plt.ylabel("Normalized Gain")
        plt.ylim(-1, 2)  # Set y-axis limits
        plt.legend(title="Agent or Base")

    def plotThroughputJammingReaction(self, jammer_ep_step: int) -> None:
        """
        Plots throughput reaction to jamming showing three phases:
        - Stabilized throughput before jamming
        - Throughput at the moment of jamming
        - Stabilized throughput after jamming

        Args:
            jammer_ep_step (int): The step when jamming starts
        """
        self.log_loader.groupLogPaths(
            byNodes=True, byJammers=True, distinguish_baselines=True
        )
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )

        group = self.log_loader.group

        df_throughput = self.log_data_processor.getStatsForValue("throughput")

        # Get throughput at jamming (exactly at jammer_ep_step)
        df_at_jamming = df_throughput.copy()
        df_at_jamming = df_at_jamming[df_at_jamming["ep_step"] == jammer_ep_step]
        df_at_jamming["jamming_phase"] = "At Jamming"

        # Get stabilized throughput before jamming
        df_before_jamming = (
            self.log_data_processor.getGroupedEpisodeStabilizedMeansForValue(
                group, "throughput", end_step=jammer_ep_step - 1
            )
        )
        df_before_jamming["jamming_phase"] = "Before Jamming"

        # Get stabilized throughput after jamming
        df_after_jamming = (
            self.log_data_processor.getGroupedEpisodeStabilizedMeansForValue(
                group, "throughput", start_step=jammer_ep_step
            )
        )
        df_after_jamming["jamming_phase"] = "After Jamming"

        # Combine all three phases
        df_combined = pd.concat([df_before_jamming, df_at_jamming, df_after_jamming])

        self._init_new_figure()
        self._set_theme()

        sns.lineplot(
            data=df_combined,
            x="nodes",
            y="value",
            style="jamming_phase",
            hue="agent_or_base",
            markers=True,
            dashes=True,
            errorbar=("ci", 95),
        )

        plt.xlabel(r"$\vert M \vert$")
        plt.ylabel(r"$\tau \, \text{[bit/s]}$")

        # Apply scientific notation
        apply_scientific_notation(plt.gca())

        # Customize the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        filtered_handles = []
        filtered_labels = []

        for handle, label in zip(handles, labels):
            if label not in ["jamming_phase", "agent_or_base"]:
                if label in AGENT_OR_BASE_DICT.keys():
                    filtered_labels.append(AGENT_OR_BASE_DICT[label])
                else:
                    filtered_labels.append(label)
                filtered_handles.append(handle)

        plt.legend(
            filtered_handles,
            filtered_labels,
            ncol=2,
            title=None,
        )

        plt.title("throughput_jamming_reaction")
        plt.gca().title.set_visible(False)

    def plotThroughputGain(self, jammer_ep_step: int) -> None:
        """
        Plots throughput gain showing the difference between:
        - Stabilized throughput after jamming
        - Throughput at the moment of jamming

        Args:
            jammer_ep_step (int): The step when jamming starts
        """
        self.log_loader.groupLogPaths(
            byNodes=True, byJammers=True, distinguish_baselines=True
        )
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )
        self.log_data_processor.filter_custom_by_numb_jammers((1, 3))
        # Get throughput at jamming (exactly at jammer_ep_step)
        df_throughput = self.log_data_processor.getStatsForValue("throughput").copy()
        df_at_jamming = df_throughput[df_throughput["ep_step"] == jammer_ep_step]
        df_at_jamming = df_at_jamming.drop(
            columns=["ep_step", "value_name"], errors="ignore"
        )

        # Get stabilized throughput
        group = self.log_loader.group

        df_stabilize_throughput_before_jamming = (
            self.log_data_processor.getGroupedEpisodeStabilizedMeansForValue(
                group, "throughput", end_step=jammer_ep_step - 1
            )
        )

        df_stabilize_throughput_after_jamming = (
            self.log_data_processor.getGroupedEpisodeStabilizedMeansForValue(
                group, "throughput", start_step=jammer_ep_step
            )
        )

        # First merge: before jamming and after jamming
        df_temp = pd.merge(
            df_stabilize_throughput_before_jamming,
            df_stabilize_throughput_after_jamming,
            on=["episode", "nodes", "agent_or_base", "jammers"]
            + [
                col
                for col in group
                if col not in ["episode", "nodes", "agent_or_base", "jammers"]
            ],
            suffixes=("_before", "_after"),
        )

        # Second merge: merge the result with at jamming
        # Rename 'value' column in df_at_jamming to 'value_at'
        df_at_jamming = df_at_jamming.rename(columns={"value": "value_at"})

        # Second merge: merge the result with at jamming
        df_recovery = pd.merge(
            df_temp,
            df_at_jamming,
            on=["episode", "nodes", "agent_or_base", "jammers"]
            + [
                col
                for col in group
                if col not in ["episode", "nodes", "agent_or_base", "jammers"]
            ],
        )

        # Calculate recovery ratio
        df_recovery["throughput_gain"] = (
            df_recovery["value_after"] - df_recovery["value_at"]
        )

        # Handle infinities and NaNs
        df_recovery = _replace_infs_nans(df_recovery, "throughput_gain", 1.0, 0.0, 0.0)

        self._init_new_figure()
        self._set_theme()

        sns.lineplot(
            data=df_recovery,
            x="nodes",
            y="throughput_gain",
            hue="agent_or_base",
            style="jammers",
            markers=True,
            dashes=True,
            errorbar=("ci", 95),
        )
        plt.xlabel(r"$\vert M \vert$")
        plt.ylabel(r"$\Delta \tau \, \text{[bit/s]}$")

        # Apply scientific notation
        apply_scientific_notation(plt.gca())

        # Customize the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        filtered_handles = []
        filtered_labels = []

        for handle, label in zip(handles, labels):
            if label not in ["agent_or_base", "jammers"]:
                if label.isdigit():
                    filtered_labels.append(rf"$\vert J \vert = {label}$")
                elif label in AGENT_OR_BASE_DICT.keys():
                    filtered_labels.append(AGENT_OR_BASE_DICT[label])
                else:
                    filtered_labels.append(label)
                filtered_handles.append(handle)

        plt.legend(
            filtered_handles,
            filtered_labels,
            ncol=2,
            title=None,
        )

        plt.title("throughput_gain")
        plt.gca().title.set_visible(False)

    def test_throughput_consistency(self, jammer_ep_step: int) -> None:
        """
        Tests that the throughput before jamming and at jamming is the same for both the baseline and the agent for each episode.

        Args:
            jammer_ep_step (int): The step when jamming starts.
        """
        # Group data
        self.log_loader.groupLogPaths(
            byNodes=True, byJammers=True, distinguish_baselines=True
        )
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )

        # Get throughput at jamming
        df_throughput = self.log_data_processor.getStatsForValue("throughput")
        df_at_jamming = df_throughput[df_throughput["ep_step"] == jammer_ep_step]

        # Pivot the data to compare baseline and agent throughput at jamming
        df_pivot_at_jamming = df_at_jamming.pivot_table(
            index=["episode", "nodes", "jammers"],
            columns="agent_or_base",
            values="value",
        ).reset_index()

        # Check for consistency at jamming
        inconsistent_at_jamming = df_pivot_at_jamming[
            df_pivot_at_jamming["Agent"] != df_pivot_at_jamming["Baseline"]
        ]

        if not inconsistent_at_jamming.empty:
            print("Inconsistencies found in throughput at jamming:")
            print(inconsistent_at_jamming)
        else:
            print("Throughput at jamming is consistent for all episodes.")

        # Get stabilized throughput before jamming
        df_before_jamming = (
            self.log_data_processor.getGroupedEpisodeStabilizedMeansForValue(
                self.log_loader.group, "throughput", end_step=jammer_ep_step - 1
            )
        )

        # Pivot the data to compare baseline and agent throughput before jamming
        df_pivot_before_jamming = df_before_jamming.pivot_table(
            index=["episode", "nodes", "jammers"],
            columns="agent_or_base",
            values="value",
        ).reset_index()

        # Check for consistency before jamming
        inconsistent_before_jamming = df_pivot_before_jamming[
            df_pivot_before_jamming["Agent"] != df_pivot_before_jamming["Baseline"]
        ]

        if not inconsistent_before_jamming.empty:
            print("Inconsistencies found in throughput before jamming:")
            print(inconsistent_before_jamming)
        else:
            print("Throughput before jamming is consistent for all episodes.")

    def plotThroughputRecoveryFraction(self, jammer_ep_step: int) -> None:
        self.log_loader.groupLogPaths(
            byNodes=True, byJammers=True, distinguish_baselines=True
        )
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )

        # Get throughput at jamming (exactly at jammer_ep_step)
        df_throughput = self.log_data_processor.getStatsForValue("throughput").copy()
        df_at_jamming = df_throughput[df_throughput["ep_step"] == jammer_ep_step]

        # Get stabilized throughput
        group = self.log_loader.group

        df_stabilize_throughput_before_jamming = (
            self.log_data_processor.getGroupedEpisodeStabilizedMeansForValue(
                group, "throughput", end_step=jammer_ep_step - 1
            )
        )

        df_stabilize_throughput_after_jamming = (
            self.log_data_processor.getGroupedEpisodeStabilizedMeansForValue(
                group, "throughput", start_step=jammer_ep_step
            )
        )

        # First merge: before jamming and after jamming
        df_temp = pd.merge(
            df_stabilize_throughput_before_jamming,
            df_stabilize_throughput_after_jamming,
            on=["episode", "nodes", "agent_or_base", "jammers"]
            + [
                col
                for col in group
                if col not in ["episode", "nodes", "agent_or_base", "jammers"]
            ],
            suffixes=("_before", "_after"),
        )

        # Second merge: merge the result with at jamming
        df_recovery = pd.merge(
            df_temp,
            df_at_jamming,
            on=["episode", "nodes", "agent_or_base", "jammers"]
            + [
                col
                for col in group
                if col not in ["episode", "nodes", "agent_or_base", "jammers"]
            ],
            suffixes=("", "_at"),
        )

        # Calculate recovery ratio
        df_recovery["throughput_recovery"] = (
            df_recovery["value_after"] / df_recovery["value_before"]
        )

        # Handle infinities and NaNs
        df_recovery = _replace_infs_nans(
            df_recovery, "throughput_recovery", 1.0, 0.0, 0.0
        )

        self._init_new_figure()
        self._set_theme()

        sns.lineplot(
            data=df_recovery,
            x="nodes",
            y="throughput_recovery",
            hue="agent_or_base",
            style="jammers",
            markers=True,
            dashes=True,
            errorbar=("ci", 95),
        )

        plt.xlabel(r"$\vert M \vert$")
        plt.ylabel(r"Throughput Recovery Ratio")

        # Customize the legend with LaTeX formatting as in other methods
        handles, labels = plt.gca().get_legend_handles_labels()
        filtered_handles = []
        filtered_labels = []
        for handle, label in zip(handles, labels):
            if label not in ["agent_or_base", "jammers"]:
                if label.isdigit():
                    filtered_labels.append(rf"$\vert J \vert = {label}$")
                elif label in AGENT_OR_BASE_DICT.keys():
                    filtered_labels.append(AGENT_OR_BASE_DICT[label])
                else:
                    filtered_labels.append(label)
                filtered_handles.append(handle)

        plt.legend(
            filtered_handles,
            filtered_labels,
            ncol=2,
            title=None,
        )
        plt.title("throughput_recovery_fraction")
        plt.gca().title.set_visible(False)

    def plotThroughputRecoveryFractionWithBase(self, jammer_ep_step: int) -> None:
        self.log_loader.groupLogPaths(
            byNodes=True, byJammers=True, distinguish_baselines=True
        )
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )
        # self.log_data_processor.use_test_df()

        # Get stabilized throughput
        group = self.log_loader.group
        df_throughput = self.log_data_processor.getStatsForValue("throughput").copy()

        df_stabilize_throughput_before_jamming = (
            self.log_data_processor.getGroupedEpisodeStabilizedMeansForValue(
                group, "throughput", end_step=jammer_ep_step - 1
            )
        )

        df_stabilize_throughput_after_jamming = (
            self.log_data_processor.getGroupedEpisodeStabilizedMeansForValue(
                group, "throughput", start_step=jammer_ep_step + 1
            )
        )

        df_throughput_at_jamming = df_throughput[
            df_throughput["ep_step"] == jammer_ep_step
        ]

        # Create a merged DataFrame for throughput recovery calculation
        df_base = pd.merge(
            df_stabilize_throughput_before_jamming,
            df_throughput_at_jamming,
            on=["episode", "nodes", "agent_or_base", "jammers"]
            + [
                col
                for col in group
                if col not in ["episode", "nodes", "agent_or_base", "jammers"]
            ],
            suffixes=("_before", "_at"),
        )

        # Calculate recovery ratio
        df_base["throughput_recovery"] = df_base["value_at"] / df_base["value_before"]

        # Handle infinities and NaNs
        df_base = _replace_infs_nans(df_base, "throughput_recovery", 1.0, 0.0, 0.0)

        # Create a merged DataFrame for throughput recovery calculation
        df_recovery = pd.merge(
            df_stabilize_throughput_before_jamming,
            df_stabilize_throughput_after_jamming,
            on=["episode", "nodes", "agent_or_base", "jammers"]
            + [
                col
                for col in group
                if col not in ["episode", "nodes", "agent_or_base", "jammers"]
            ],
            suffixes=("_before", "_after"),
        )

        # Calculate recovery ratio
        df_recovery["throughput_recovery"] = (
            df_recovery["value_after"] / df_recovery["value_before"]
        )

        # Handle infinities and NaNs
        df_recovery = _replace_infs_nans(
            df_recovery, "throughput_recovery", 1.0, 0.0, 0.0
        )

        self._init_new_figure()
        self._set_theme()

        sns.lineplot(
            data=df_recovery,
            x="nodes",
            y="throughput_recovery",
            hue="agent_or_base",
            style="jammers",
            markers=True,
            dashes=True,
            errorbar=("ci", 95),
        )

        sns.lineplot(
            data=df_base,
            x="nodes",
            y="throughput_recovery",
            hue="agent_or_base",
            style="jammers",
            markers=True,
            dashes=True,
            palette=["black", "black"],  # Custom palette with two black colors
            errorbar=("ci", 95),
        )

        plt.xlabel(r"$\vert M \vert$")
        plt.ylabel(r"Throughput Recovery Ratio")

        # Customize the legend with LaTeX formatting as in other methods
        handles, labels = plt.gca().get_legend_handles_labels()
        filtered_handles = []
        filtered_labels = []
        for handle, label in zip(handles, labels):
            if label not in ["agent_or_base", "jammers"]:
                if label.isdigit():
                    filtered_labels.append(rf"$\vert J \vert = {label}$")
                elif label in AGENT_OR_BASE_DICT.keys():
                    filtered_labels.append(AGENT_OR_BASE_DICT[label])
                else:
                    filtered_labels.append(label)
                filtered_handles.append(handle)

        plt.legend(
            filtered_handles,
            filtered_labels,
            ncol=2,
            title=None,
        )
        plt.title("throughput_recovery_with_base")
        plt.gca().title.set_visible(False)

    def plotThroughputRecoveryEpStep(self) -> None:
        self.log_loader.groupLogPaths(
            byNodes=True, byJammers=True, distinguish_baselines=True
        )
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )

        # Get stabilized throughput
        df_stepForValue = self.log_data_processor.getStatsForValue("throughput")
        df_stepForValue = df_stepForValue[df_stepForValue["jammers"] == 1]
        sns.lineplot(
            data=df_stepForValue,
            x="ep_step",
            y="value",
            hue="agent_or_base",
            style="nodes",
            markers=True,
            dashes=True,
            errorbar=None,
            markevery=100,
        )

        plt.title("throughput_recovery_ep_step")
        plt.gca().title.set_visible(False)

    def plotThroughputRecoveryFromAtJammed(self, jammer_ep_step: int) -> None:
        self.log_loader.groupLogPaths(
            byNodes=True, byJammers=True, distinguish_baselines=True
        )
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )
        self.log_data_processor.filter_custom_by_numb_jammers((1, 3))
        # Get throughput at jamming (exactly at jammer_ep_step)
        df_throughput = self.log_data_processor.getStatsForValue("throughput").copy()
        df_at_jamming = df_throughput[df_throughput["ep_step"] == jammer_ep_step]
        df_at_jamming = df_at_jamming.drop(
            columns=["ep_step", "value_name"], errors="ignore"
        )

        # Get stabilized throughput
        group = self.log_loader.group

        df_stabilize_throughput_before_jamming = (
            self.log_data_processor.getGroupedEpisodeStabilizedMeansForValue(
                group, "throughput", end_step=jammer_ep_step - 1
            )
        )

        df_stabilize_throughput_after_jamming = (
            self.log_data_processor.getGroupedEpisodeStabilizedMeansForValue(
                group, "throughput", start_step=jammer_ep_step
            )
        )

        # First merge: before jamming and after jamming
        df_temp = pd.merge(
            df_stabilize_throughput_before_jamming,
            df_stabilize_throughput_after_jamming,
            on=["episode", "nodes", "agent_or_base", "jammers"]
            + [
                col
                for col in group
                if col not in ["episode", "nodes", "agent_or_base", "jammers"]
            ],
            suffixes=("_before", "_after"),
        )

        # Second merge: merge the result with at jamming
        # Rename 'value' column in df_at_jamming to 'value_at'
        df_at_jamming = df_at_jamming.rename(columns={"value": "value_at"})

        # Second merge: merge the result with at jamming
        df_recovery = pd.merge(
            df_temp,
            df_at_jamming,
            on=["episode", "nodes", "agent_or_base", "jammers"]
            + [
                col
                for col in group
                if col not in ["episode", "nodes", "agent_or_base", "jammers"]
            ],
        )

        # Compute recovery ratio: post-recovery throughput / at-jamming throughput
        df_recovery["throughput_recovery"] = (
            df_recovery["value_after"] / df_recovery["value_at"]
        )

        # Clean up NaNs and infs
        df_recovery = _replace_infs_nans(
            df_recovery, "throughput_recovery", 1.0, 0.0, 0.0
        )

        # Plot
        self._init_new_figure()
        self._set_theme()

        sns.lineplot(
            data=df_recovery,
            x="nodes",
            y="throughput_recovery",
            hue="agent_or_base",
            style="jammers",
            markers=True,
            dashes=True,
            errorbar=("ci", 95),
        )

        plt.xlabel(r"$\vert M \vert$")
        plt.ylabel("Throughput Recovery Ratio")
        plt.title("Throughput Recovery (After / At Jamming)")

        # Clean legend
        handles, labels = plt.gca().get_legend_handles_labels()
        filtered_handles, filtered_labels = [], []
        for handle, label in zip(handles, labels):
            if label not in ["agent_or_base", "jammers"]:
                if label.isdigit():
                    filtered_labels.append(rf"$\vert J \vert = {label}$")
                elif label in AGENT_OR_BASE_DICT:
                    filtered_labels.append(AGENT_OR_BASE_DICT[label])
                else:
                    filtered_labels.append(label)
                filtered_handles.append(handle)

        plt.legend(filtered_handles, filtered_labels, ncol=2, title=None)
        plt.title("throughput_recovery_at_jamming")
        plt.gca().title.set_visible(False)

    def plotMeanStabilizationStep(self, jammer_ep_steps: int = 0) -> None:
        """
        Plots the mean stabilization step.
        """
        # Group data
        self.log_loader.groupLogPaths(
            byNodes=True, byJammers=True, distinguish_baselines=True
        )
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )
        self.log_data_processor.filterByJammerEpSteps(jammer_ep_steps=jammer_ep_steps)

        # Get stabilized throughput
        group = self.log_loader.group
        df_stabilized_throughput = (
            self.log_data_processor.getGroupedEpisodeStabilizedStepsForValue(
                group, "throughput"
            )
        )

        # Subtract jammer_ep_steps from stabilization_step values only where "jammers" > 0
        df_stabilized_throughput.loc[
            (df_stabilized_throughput["value_name"] == "stabilization_step")
            & (df_stabilized_throughput["jammers"] > 0),
            "value",
        ] -= jammer_ep_steps

        plt.figure()
        self._set_theme()
        sns.lineplot(
            data=df_stabilized_throughput,
            x="nodes",
            y="value",
            hue="agent_or_base",
            style="jammers",
            markers=True,
            dashes=True,
            errorbar=("ci", 95),
        )

        # Apply scientific notation
        apply_scientific_notation(plt.gca())

        # Update axis labels with LaTeX
        plt.xlabel(r"$\vert M \vert$")  # Number of nodes
        plt.ylabel(r"$t$")

        # Customize the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        # Filter out the titles for "agent_or_base" and "jammers"
        filtered_handles = []
        filtered_labels = []
        for handle, label in zip(handles, labels):
            if label not in ["agent_or_base", "jammers"]:  # Skip the titles
                if label.isdigit():
                    filtered_labels.append(
                        rf"$\vert J \vert = {label}$"
                    )  # Replace jammers with |\mathcal{J}|
                elif label in AGENT_OR_BASE_DICT.keys():
                    filtered_labels.append(AGENT_OR_BASE_DICT[label])
                else:
                    filtered_labels.append(label)  # Keep other labels as is
                filtered_handles.append(handle)  # Keep the corresponding handle

        # Group the legend entries
        # Separate "Agent" and "Baseline" from the jammers
        agent_base_handles = []
        agent_base_labels = []
        jammers_handles = []
        jammers_labels = []

        for handle, label in zip(filtered_handles, filtered_labels):
            if label in AGENT_OR_BASE_DICT.values():
                agent_base_handles.append(handle)
                agent_base_labels.append(label)
            else:
                jammers_handles.append(handle)
                jammers_labels.append(label)

        # Combine the handles and labels in the desired order
        final_handles = jammers_handles + agent_base_handles
        final_labels = jammers_labels + agent_base_labels

        # Create the legend without the subtitle
        plt.legend(
            final_handles,
            final_labels,
            ncol=2,  # Two columns
            title=None,  # Remove legend title
        )

        plt.xticks(df_stabilized_throughput["nodes"].unique().astype(int))

        plt.title("stabilization_step")
        plt.gca().title.set_visible(False)

    def plotMeanSettledStep(self) -> None:
        """
        Plots the mean settle step.
        """
        # Group data
        self.log_loader.groupLogPaths(True, True, True, True)
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )

        # Get grouped ADT
        group = self.log_loader.group
        self.log_data_processor.getGroupedAverageDistanceTravelled(group)

        # Get settled step
        df_settled_step = (
            self.log_data_processor.getGroupedEpisodeStabilizedStepsForValue(
                group, "cum_distance"
            )
        )

        plt.figure()
        self._set_theme()
        sns.lineplot(
            data=df_settled_step,
            x="nodes",
            y="value",
            hue="agent_or_base",
            style="jammers",
            markers=True,
            dashes=True,
        )

        plt.title("Mean Settle Step")
        plt.xlabel("Number of Nodes")
        plt.ylabel("Step")
        plt.legend(title="Agent or Base")
        plt.xticks(df_settled_step["nodes"].unique().astype(int))

    def printMeanStabilizedThroughputs(
        self, jammer_ep_steps: int = 0, store_dir: str = None
    ) -> None:
        """
        Print the mean stabilized throughputs.
        """
        # Group data
        self.log_loader.groupLogPaths(
            byNodes=True, byJammers=True, distinguish_baselines=True
        )
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )

        self.log_data_processor.filterByJammerEpSteps(jammer_ep_steps=jammer_ep_steps)
        # Get grouped stabilized throughput
        group = self.log_loader.group
        df_stabilized_throughput = (
            self.log_data_processor.getGroupedEpisodeStabilizedMeansForValue(
                group, "throughput"
            )
        )
        scaling_factor = 1e6  # Scale to Mbps
        df_stabilized_throughput["value"] /= scaling_factor

        # Print the raw data
        print("Raw stabilized throughput data:")
        print(df_stabilized_throughput)

        # Calculate and print the means for each group
        mean_values = (
            df_stabilized_throughput.groupby(["agent_or_base", "nodes", "jammers"])[
                "value"
            ]
            .mean()
            .reset_index()
        )
        print("\nCalculated mean stabilized throughputs:")
        print(mean_values)

        name = "stable_throughput"
        if store_dir is not None:
            store_path = Path(store_dir) / (name + ".csv")
            # Optionally, save the means to a CSV file
            mean_values.to_csv(store_path, index=False)

    def plotMeanStabilizedThroughputs(self, jammer_ep_steps: int = 0) -> None:
        """
        Plots the mean stabilized throughputs.
        """
        # Group data
        self.log_loader.groupLogPaths(
            byNodes=True, byJammers=True, distinguish_baselines=True
        )
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )

        self.log_data_processor.filterByJammerEpSteps(jammer_ep_steps=jammer_ep_steps)
        # Get grouped stabilized throughput
        group = self.log_loader.group
        df_stabilized_throughput = (
            self.log_data_processor.getGroupedEpisodeStabilizedMeansForValue(
                group, "throughput"
            )
        )

        self._init_new_figure()
        self._set_theme()
        sns.lineplot(
            data=df_stabilized_throughput,
            x="nodes",
            y="value",
            hue="agent_or_base",
            style="jammers",
            markers=True,
            dashes=True,
        )

        # Apply scientific notation
        apply_scientific_notation(plt.gca())

        # Update axis labels with LaTeX
        plt.xlabel(r"$\vert M \vert$")  # Number of nodes
        plt.ylabel(r"$\tau_{\text{stable}} \, \text{[bit/s]}$")

        # Customize the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        # Filter out the titles for "agent_or_base" and "jammers"
        filtered_handles = []
        filtered_labels = []
        for handle, label in zip(handles, labels):
            if label not in ["agent_or_base", "jammers"]:  # Skip the titles
                if label.isdigit():
                    filtered_labels.append(
                        rf"$\vert J \vert = {label}$"
                    )  # Replace jammers with |\mathcal{J}|
                elif label in AGENT_OR_BASE_DICT.keys():
                    filtered_labels.append(AGENT_OR_BASE_DICT[label])
                else:
                    filtered_labels.append(label)  # Keep other labels as is
                filtered_handles.append(handle)  # Keep the corresponding handle

        # Group the legend entries
        # Separate "Agent" and "Baseline" from the jammers
        agent_base_handles = []
        agent_base_labels = []
        jammers_handles = []
        jammers_labels = []

        for handle, label in zip(filtered_handles, filtered_labels):
            if label in AGENT_OR_BASE_DICT.values():
                agent_base_handles.append(handle)
                agent_base_labels.append(label)
            else:
                jammers_handles.append(handle)
                jammers_labels.append(label)

        # Combine the handles and labels in the desired order
        final_handles = jammers_handles + agent_base_handles
        final_labels = jammers_labels + agent_base_labels

        # Create the legend without the subtitle
        plt.legend(
            final_handles,
            final_labels,
            ncol=2,  # Two columns
            title=None,  # Remove legend title
        )

        # Set the title but make it invisible
        plt.title("mean_stabilized_throughput")
        plt.gca().title.set_visible(False)

        plt.xticks(df_stabilized_throughput["nodes"].unique().astype(int))

    def printMeanThroughputs(
        self, jammer_ep_steps: int = 0, store_dir: str = None
    ) -> None:
        """
        Print the mean throughputs.
        """
        # Group data
        self.log_loader.groupLogPaths(
            byNodes=True, byJammers=True, distinguish_baselines=True
        )
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )

        self.log_data_processor.filterByJammerEpSteps(jammer_ep_steps=jammer_ep_steps)
        # Get grouped stabilized throughput
        group = self.log_loader.group
        df_throughput = self.log_data_processor.getGroupedEpisodeMeansForValue(
            group, "throughput"
        )

        scaling_factor = 1e6  # Scale to Mbps
        df_throughput["value"] /= scaling_factor

        # Print the raw data
        print("Raw stabilized throughput data:")
        print(df_throughput)

        # Calculate and print the means for each group
        mean_values = (
            df_throughput.groupby(["agent_or_base", "nodes", "jammers"])["value"]
            .mean()
            .reset_index()
        )
        print("\nCalculated mean stabilized throughputs:")
        print(mean_values)

        name = "mean_throughputs"
        if store_dir is not None:
            store_path = Path(store_dir) / (name + ".csv")
            # Optionally, save the means to a CSV file
            mean_values.to_csv(store_path, index=False)

    def plotMeanThroughputs(self, jammer_ep_steps: int = 0) -> None:
        """
        Plots the mean throughputs.
        """
        # Group data
        self.log_loader.groupLogPaths(
            byNodes=True, byJammers=True, distinguish_baselines=True
        )
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )

        self.log_data_processor.filterByJammerEpSteps(jammer_ep_steps=jammer_ep_steps)
        # Get grouped stabilized throughput
        group = self.log_loader.group
        df_throughput = self.log_data_processor.getGroupedEpisodeMeansForValue(
            group, "throughput"
        )

        self._init_new_figure()
        self._set_theme()
        sns.lineplot(
            data=df_throughput,
            x="nodes",
            y="value",
            hue="agent_or_base",
            style="jammers",
            markers=True,
            dashes=True,
        )

        # Apply scientific notation
        apply_scientific_notation(plt.gca())

        # Update axis labels with LaTeX
        plt.xlabel(r"$\vert M \vert$")  # Number of nodes
        plt.ylabel(r"$\tau_{\text{stable}} \, \text{[bit/s]}$")

        # Customize the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        # Filter out the titles for "agent_or_base" and "jammers"
        filtered_handles = []
        filtered_labels = []
        for handle, label in zip(handles, labels):
            if label not in ["agent_or_base", "jammers"]:  # Skip the titles
                if label.isdigit():
                    filtered_labels.append(
                        rf"$\vert J \vert = {label}$"
                    )  # Replace jammers with |\mathcal{J}|
                elif label in AGENT_OR_BASE_DICT.keys():
                    filtered_labels.append(AGENT_OR_BASE_DICT[label])
                else:
                    filtered_labels.append(label)  # Keep other labels as is
                filtered_handles.append(handle)  # Keep the corresponding handle

        # Group the legend entries
        # Separate "Agent" and "Baseline" from the jammers
        agent_base_handles = []
        agent_base_labels = []
        jammers_handles = []
        jammers_labels = []

        for handle, label in zip(filtered_handles, filtered_labels):
            if label in AGENT_OR_BASE_DICT.values():
                agent_base_handles.append(handle)
                agent_base_labels.append(label)
            else:
                jammers_handles.append(handle)
                jammers_labels.append(label)

        # Combine the handles and labels in the desired order
        final_handles = jammers_handles + agent_base_handles
        final_labels = jammers_labels + agent_base_labels

        # Create the legend without the subtitle
        plt.legend(
            final_handles,
            final_labels,
            ncol=2,  # Two columns
            title=None,  # Remove legend title
        )

        # Set the title but make it invisible
        plt.title("mean_throughput")
        plt.gca().title.set_visible(False)

        plt.xticks(df_throughput["nodes"].unique().astype(int))

    def plotMeanStabilizedThroughputsOverJammersByAttacker(
        self, legend: bool = False
    ) -> None:
        """
        Plots the mean stabilized throughputs.
        """
        # Group data
        self.log_loader.groupLogPaths(
            byJammers=True, distinguish_baselines=True, byAttackerModel=True
        )
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )

        # Get grouped stabilized throughput
        group = self.log_loader.group
        df_stabilized_throughput = (
            self.log_data_processor.getGroupedEpisodeStabilizedMeansForValue(
                group, "throughput"
            )
        )

        self._init_new_figure()
        self._set_theme()
        sns.lineplot(
            data=df_stabilized_throughput,
            x="jammers",
            y="value",
            hue="agent_or_base",
            style="attackerModel",
            markers=True,
            dashes=True,
        )

        # Update axis labels with LaTeX
        plt.xlabel(r"$\vert J \vert$")  # Number of nodes
        plt.ylabel(r"$\tau \, \text{[bit/s]}$")

        # Apply scientific notation
        apply_scientific_notation(plt.gca())

        # Customize the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        # Filter out the titles for "agent_or_base" and "jammers"
        filtered_handles = []
        filtered_labels = []
        for handle, label in zip(handles, labels):
            if label not in ["agent_or_base", "attackerModel"]:  # Skip the titles
                if label in JAMMER_DICT.keys():
                    filtered_labels.append(JAMMER_DICT[label])
                elif label in AGENT_OR_BASE_DICT.keys():
                    filtered_labels.append(AGENT_OR_BASE_DICT[label])
                else:
                    filtered_labels.append(label)  # Keep other labels as is
                filtered_handles.append(handle)  # Keep the corresponding handle

        # Group the legend entries
        # Separate "Agent" and "Baseline" from the jammers
        agent_base_handles = []
        agent_base_labels = []
        attackermodels_handles = []
        attackermodels_labels = []

        for handle, label in zip(filtered_handles, filtered_labels):
            if label in ["Agent", "Baseline"]:
                agent_base_handles.append(handle)
                agent_base_labels.append(label)
            else:
                attackermodels_handles.append(handle)
                attackermodels_labels.append(label)

        # Combine the handles and labels in the desired order
        final_handles = attackermodels_handles + agent_base_handles
        final_labels = attackermodels_labels + agent_base_labels

        # Create the legend without the subtitle
        plt.legend(
            final_handles,
            final_labels,
            ncol=2,  # Two columns
            title=None,  # Remove legend title
        )
        plt.xticks(df_stabilized_throughput["jammers"].unique().astype(int))

    def printAttackerModels(
        self, jammer_ep_steps: int = 0, store_dir: str = None
    ) -> None:
        """
        Plots the mean stabilized throughputs.
        """
        # Group data
        self.log_loader.groupLogPaths(
            byNodes=True,
            byJammers=True,
            distinguish_baselines=True,
            byAttackerModel=True,
        )
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )
        self.log_data_processor.filterByJammerEpSteps(jammer_ep_steps=jammer_ep_steps)
        # Get grouped stabilized throughput
        group = self.log_loader.group
        df_stabilized_throughput = (
            self.log_data_processor.getGroupedEpisodeStabilizedMeansForValue(
                group, "throughput"
            )
        )

        scaling_factor = 1e6  # Scale to Mbps
        df_stabilized_throughput["value"] /= scaling_factor

        # Print the raw data
        print("Raw stabilized throughput data:")
        print(df_stabilized_throughput)

        # Calculate and print the means for each group
        mean_values = (
            df_stabilized_throughput.groupby(
                ["agent_or_base", "attackerModel", "nodes"]
            )["value"]
            .mean()
            .reset_index()
        )
        print("\nCalculated mean stabilized throughputs:")
        print(mean_values)

        # Optionally, save the means to a CSV file
        name = "attacker_models"
        if store_dir is not None:
            store_path = Path(store_dir) / (name + ".csv")
            # Optionally, save the means to a CSV file
            mean_values.to_csv(store_path, index=False)

    def plotMeanStabilizedThroughputsOverNodesByAttacker(
        self, jammer_ep_steps: int = 0
    ) -> None:
        """
        Plots the mean stabilized throughputs.
        """
        # Group data
        self.log_loader.groupLogPaths(
            byNodes=True,
            byJammers=True,
            distinguish_baselines=True,
            byAttackerModel=True,
        )
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )
        self.log_data_processor.filterByJammerEpSteps(jammer_ep_steps=jammer_ep_steps)
        # Get grouped stabilized throughput
        group = self.log_loader.group
        df_stabilized_throughput = (
            self.log_data_processor.getGroupedEpisodeStabilizedMeansForValue(
                group, "throughput"
            )
        )

        # df_stabilized_throughput = df_stabilized_throughput[
        #     df_stabilized_throughput["jammers"] == 1
        # ]

        self._init_new_figure()
        self._set_theme()
        sns.lineplot(
            data=df_stabilized_throughput,
            x="nodes",
            y="value",
            hue="agent_or_base",
            style="attackerModel",
            markers=True,
            dashes=True,
        )

        # Update axis labels with LaTeX
        plt.xlabel(r"$\vert M \vert$")  # Number of nodes
        plt.ylabel(r"$\tau_{\text{stable}} \, \text{[bit/s]}$")

        # Apply scientific notation
        apply_scientific_notation(plt.gca())

        # Customize the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        # Filter out the titles for "agent_or_base" and "jammers"
        filtered_handles = []
        filtered_labels = []
        for handle, label in zip(handles, labels):
            if label not in ["agent_or_base", "attackerModel"]:  # Skip the titles
                if label in JAMMER_DICT.keys():
                    filtered_labels.append(JAMMER_DICT[label])
                else:
                    filtered_labels.append(label)  # Keep other labels as is
                filtered_handles.append(handle)  # Keep the corresponding handle

        # Group the legend entries
        # Separate "Agent" and "Baseline" from the jammers
        agent_base_handles = []
        agent_base_labels = []
        attackermodels_handles = []
        attackermodels_labels = []

        for handle, label in zip(filtered_handles, filtered_labels):
            if label in ["Agent", "Baseline"]:
                agent_base_handles.append(handle)
                agent_base_labels.append(AGENT_OR_BASE_DICT[label])
            else:
                attackermodels_handles.append(handle)
                attackermodels_labels.append(label)

        # Combine the handles and labels in the desired order
        final_handles = attackermodels_handles + agent_base_handles
        final_labels = attackermodels_labels + agent_base_labels

        # Create the legend without the subtitle
        plt.legend(
            final_handles,
            final_labels,
            ncol=2,  # Two columns
            title=None,  # Remove legend title
        )
        plt.xticks(df_stabilized_throughput["nodes"].unique().astype(int))

        # Set the title but make it invisible
        plt.title("attacker_models")
        plt.gca().title.set_visible(False)

    def plotJammerEffectOverNodes(self, jammer_ep_steps: int = 150) -> None:
        """
        Plots the mean stabilized throughputs.
        """
        # --- Prepare throughput data ---
        self.log_loader.groupLogPaths(
            byNodes=True, byJammers=True, distinguish_baselines=True
        )
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )
        self.log_data_processor.resetCache()
        df_before = self.log_data_processor.getStatsForValue("throughput").copy()
        df_before = df_before[df_before["ep_step"] == jammer_ep_steps]
        df_before["before_or_after"] = "TBR"  # Throughput Before Relocation

        group = self.log_loader.group
        self.log_data_processor.filterByJammerEpSteps(jammer_ep_steps=jammer_ep_steps)
        df_after = self.log_data_processor.getGroupedEpisodeStabilizedMeansForValue(
            group, "throughput"
        ).copy()
        df_after["before_or_after"] = "TBA"  # Throughput After Relocation

        df_plot = pd.concat([df_before, df_after], ignore_index=True)

        # --- Prepare TTS data ---
        self.log_loader.groupLogPaths(
            byNodes=True, byJammers=True, distinguish_baselines=True
        )
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )
        group = self.log_loader.group
        df_step = self.log_data_processor.getGroupedEpisodeStabilizedStepsForValue(
            group, "throughput"
        ).copy()
        df_step["TTS"] = df_step["value"] - jammer_ep_steps

        # --- Plot ---
        import matplotlib as mpl

        plt.figure(figsize=(5, 3))
        self._set_theme()
        ax = plt.gca()

        # Throughput (left axis)
        sns.lineplot(
            data=df_plot,
            x="nodes",
            y="value",
            hue="before_or_after",
            style="agent_or_base",
            markers=True,
            dashes=True,
            ax=ax,
            errorbar=("ci", 95),
            linewidth=2,
            legend=True,  # Enable legend here
        )

        # TTS (right axis)
        ax2 = ax.twinx()
        tts_palette = sns.color_palette(
            "Set2", n_colors=df_step["agent_or_base"].nunique()
        )
        sns.lineplot(
            data=df_step,
            x="nodes",
            y="TTS",
            hue="agent_or_base",
            ax=ax2,
            errorbar=("ci", 95),
            marker="o",
            dashes=True,
            palette=tts_palette,
            legend=True,  # Enable legend here
            linewidth=2,
        )

        # Axis labels
        ax.set_xlabel(r"$\vert M \vert$")
        ax.set_ylabel(r"$\tau \, \text{[bit/s]}$")
        ax2.set_ylabel(r"$t$")

        # Scientific notation
        apply_scientific_notation(ax)
        ax2.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        # X-ticks
        ax.set_xticks(df_plot["nodes"].unique().astype(int))

        # # --- Legend ---
        # handles1, labels1 = ax.get_legend_handles_labels()
        # handles2, labels2 = ax2.get_legend_handles_labels()
        # legend_entries = []
        # legend_labels = []
        # for h, l in zip(handles1, labels1):
        #     if l in ["TBR", "TBA"]:
        #         legend_entries.append(h)
        #         legend_labels.append(
        #             "Throughput Before Relocation"
        #             if l == "TBR"
        #             else "Throughput After Relocation"
        #         )
        # for h, l in zip(handles2, labels2):
        #     legend_entries.append(h)
        #     legend_labels.append(f"TTS ({l})")
        # ax.legend(
        #     legend_entries,
        #     legend_labels,
        #     ncol=1,
        #     title=None,
        #     loc="upper left",
        #     frameon=True,
        # )

        # Grid: only on left axis
        ax.grid(True, which="both", axis="both")
        ax2.grid(False)

        ax.set_title("")

    def plotMeanTotalAverageDistanceTravelled(self) -> None:
        # Group data
        self.log_loader.groupLogPaths(True, True, True, True)
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )

        # Get grouped adt
        group = self.log_loader.group
        self.log_data_processor.getGroupedAverageDistanceTravelled(group)

        # Get only total ADT (from last ep_step in each episode)
        df_total_distance_mean = self.log_data_processor.getTotalDistancePerEpisode(
            group
        )

        plt.figure()
        self._set_theme()
        sns.lineplot(
            data=df_total_distance_mean,
            x="nodes",
            y="value",
            hue="agent_or_base",
            style="jammers",
            markers=True,
            dashes=True,
        )

        plt.title("Mean Total Average Distance Travelled")
        plt.xlabel("Number of Nodes")
        plt.ylabel("Mean ADT")
        plt.legend(title="Agent or Base")
        plt.xticks(df_total_distance_mean["nodes"].unique().astype(int))

    def plotCombinedNodeMeans(self, metrics_to_include: list[str]) -> None:
        """
        Plots multiple metrics in a single-column layout using Seaborn's FacetGrid.

        Args:
            metrics_to_include (list[str]): List of metrics to include. Options are:
                - "stabilization_step"
                - "settled_step"
                - "stabilized_throughputs"
                - "total_distance"
        """
        data_frames = []
        self.log_loader.groupLogPaths(True, True, True, True)
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )

        # Get stabilized throughput
        group = self.log_loader.group

        if "stabilization_step" in metrics_to_include:
            self.log_data_processor.resetCache()
            df_stabilized_step = (
                self.log_data_processor.getGroupedEpisodeStabilizedStepsForValue(
                    group, "throughput"
                )
            )
            df_stabilized_step = df_stabilized_step.copy()
            df_stabilized_step["plot_type"] = "Mean Stabilization Step"
            data_frames.append(df_stabilized_step)
            self.log_data_processor.clear_cache()

        if "settled_step" in metrics_to_include:
            self.log_data_processor.resetCache()

            # Get grouped ADT
            self.log_data_processor.getGroupedAverageDistanceTravelled(group)

            # Get settled step
            df_settled_step = (
                self.log_data_processor.getGroupedEpisodeStabilizedStepsForValue(
                    group, "cum_distance"
                )
            )
            df_settled_step = df_settled_step.copy()
            df_settled_step["plot_type"] = "Mean Settled Step"
            data_frames.append(df_settled_step)
            self.log_data_processor.clear_cache()

        if "stabilized_throughputs" in metrics_to_include:
            self.log_data_processor.resetCache()

            df_stabilized_throughput = (
                self.log_data_processor.getGroupedEpisodeStabilizedMeansForValue(
                    group, "throughput"
                )
            )
            df_stabilized_throughput = df_stabilized_throughput.copy()
            df_stabilized_throughput["plot_type"] = "Mean Stabilized Throughput"
            data_frames.append(df_stabilized_throughput)
            self.log_data_processor.clear_cache()

        if "total_distance" in metrics_to_include:
            self.log_data_processor.resetCache()
            self.log_data_processor.getGroupedAverageDistanceTravelled(group)

            # Get only total ADT (from last ep_step in each episode)
            df_total_distance_mean = self.log_data_processor.getTotalDistancePerEpisode(
                group
            )
            df_total_distance_mean = df_total_distance_mean.drop(columns=["ep_step"])
            df_total_distance_mean = df_total_distance_mean.copy()
            df_total_distance_mean["plot_type"] = "Mean ADT"
            data_frames.append(df_total_distance_mean)
            self.log_data_processor.clear_cache()

        # Combine all data into a single DataFrame
        combined_df = pd.concat(data_frames)

        self._set_theme()
        # Create a FacetGrid with independent y-axes for each row
        g = sns.FacetGrid(
            combined_df,
            row="plot_type",  # Each row corresponds to a plot type
            sharex=True,  # Share x-axis across subplots
            sharey=False,  # Do not share y-axis (since y-values may differ)
            aspect=4 / 3,
            height=2,
        )

        # Map a lineplot to each subplot
        g.map_dataframe(
            sns.lineplot,
            x="nodes",
            y="value",
            hue="agent_or_base",  # Color by "agent_or_base"
            style="jammers",  # Use jammers for line styles
            markers=True,
            dashes=True,
        )
        plt.xticks(combined_df["nodes"].unique().astype(int))

        # Enable LaTeX rendering globally
        plt.rcParams["text.usetex"] = True  # Use LaTeX for text rendering
        plt.rcParams["text.latex.preamble"] = (
            r"\usepackage{amsmath, amssymb}"  # Optional: Add LaTeX packages
        )

        # Customize the legend
        handles, labels = g.axes.flat[
            0
        ].get_legend_handles_labels()  # Get legend handles and labels

        # Filter out the titles for "agent_or_base" and "jammers"
        filtered_handles = []
        filtered_labels = []
        for handle, label in zip(handles, labels):
            if label not in ["agent_or_base", "jammers"]:  # Skip the titles
                if label.isdigit():
                    filtered_labels.append(
                        f"$|\\mathcal{{J}}| = {label}$"
                    )  # Replace jammers with |\mathcal{J}|
                else:
                    filtered_labels.append(label)  # Keep other labels as is
                filtered_handles.append(handle)  # Keep the corresponding handle

        # Group the legend entries
        # Separate "Agent" and "Baseline" from the jammers
        agent_base_handles = []
        agent_base_labels = []
        jammers_handles = []
        jammers_labels = []

        for handle, label in zip(filtered_handles, filtered_labels):
            if label in ["Agent", "Baseline"]:
                agent_base_handles.append(handle)
                agent_base_labels.append(label)
            else:
                jammers_handles.append(handle)
                jammers_labels.append(label)

        # Combine the handles and labels in the desired order
        final_handles = jammers_handles + agent_base_handles
        final_labels = jammers_labels + agent_base_labels

        # Create the legend with two columns: one for "Agent"/"Baseline" and one for jammers
        plt.legend(
            final_handles,
            final_labels,
            bbox_to_anchor=(0.5, -0.5),
            loc="upper center",
            ncol=2,  # Use two columns
            title=None,  # Remove the legend title
        )

        # Add space at the bottom of the entire figure
        plt.subplots_adjust(
            bottom=0.2 + 0.08 * (4 - len(metrics_to_include))
        )  # Add 20% padding at the bottom of the figure

        g.set_axis_labels(
            r"$\vert \mathcal{N} \vert$", ""
        )  # x-axis label is $\vert N \vert$, y-axis label is empty

        # Customize y-axis labels for each subplot with LaTeX expressions
        y_labels = {
            "Mean Stabilization Step": r"$Step$",
            "Mean Settled Step": r"$Step$",
            "Mean Stabilized Throughput": r"$\text{Throughput } \mathcal{\tau}$",
            "Mean ADT": r"$\text{ADT}$",
        }
        # Apply y-axis labels and scientific notation to each subplot
        for ax, title in zip(g.axes.flat, g.row_names):
            ax.set_ylabel(y_labels.get(title, title))
            ax.yaxis.set_major_formatter(
                ticker.ScalarFormatter(useMathText=True)
            )  # Use scientific notation
            ax.ticklabel_format(
                axis="y", style="sci", scilimits=(0, 0)
            )  # Force scientific notation

        # Apply y-axis labels to each subplot
        for ax, title in zip(g.axes.flat, g.row_names):
            ax.set_ylabel(y_labels.get(title, title))

        # Customize subplot titles
        subplot_titles = ["a)", "b)", "c)", "d)"]  # Add more if needed
        for i, ax in enumerate(g.axes_dict.values()):

            ax.set_title(subplot_titles[i])

        plt.rcParams["text.usetex"] = False  # Disable LaTeX
        plt.rcParams["mathtext.default"] = "regular"  # Use MathTextS

    def plotCombinedStepMeans(self, metrics_to_include: list[str]) -> None:
        data_frames = []
        self.log_loader.groupLogPaths(True, True, True, True)
        self.log_data_processor.loadGroupedDataFrame(
            self.log_loader.group, self.log_loader.grouped_log_paths
        )
        if "ADT" in metrics_to_include:
            value_name = "cum_distance"
            self.log_data_processor.resetCache()

            # Calculate average distance
            self.log_data_processor.getGroupedAverageDistanceTravelled(
                self.log_loader.group
            )
            df_adt = self.log_data_processor.getStatsForValue(value_name)

            df_adt = df_adt.copy()
            df_adt["plot_type"] = "ADT per Step"
            data_frames.append(df_adt)

        if "throughput" in metrics_to_include:
            value_name = "throughput"
            self.log_data_processor.resetCache()

            # Calculate average distance
            self.log_data_processor.getGroupedAverageDistanceTravelled(
                self.log_loader.group
            )
            df_throughput = self.log_data_processor.getStatsForValue(value_name)

            df_throughput = df_throughput.copy()
            df_throughput["plot_type"] = "Throughput per Step"
            data_frames.append(df_throughput)

        # Combine all data into a single DataFrame
        combined_df = pd.concat(data_frames)

        self._set_theme()
        unique_nodes = combined_df["nodes"].nunique()  # Get the unique number of nodes
        custom_palette = sns.color_palette("Set1", unique_nodes)

        # Create a FacetGrid with independent y-axes for each row
        g = sns.FacetGrid(
            combined_df,
            row="plot_type",  # Each row corresponds to a plot type
            sharex=True,  # Share x-axis across subplots
            sharey=False,  # Do not share y-axis (since y-values may differ)
            aspect=4 / 3,
            height=3,
        )

        # Map a lineplot to each subplot
        g.map_dataframe(
            sns.lineplot,
            x="ep_step",
            y="value",
            hue="nodes",
            errorbar=None,
            style="agent_or_base",
            markers=True,
            dashes=False,
            palette=custom_palette,
            markevery=200,
        )

        # Enable LaTeX rendering globally
        plt.rcParams["text.usetex"] = True  # Use LaTeX for text rendering
        plt.rcParams["text.latex.preamble"] = (
            r"\usepackage{amsmath, amssymb}"  # Optional: Add LaTeX packages
        )

        # Customize the legend
        handles, labels = g.axes.flat[
            0
        ].get_legend_handles_labels()  # Get legend handles and labels

        # Filter out the titles for "agent_or_base" and "jammers"
        filtered_handles = []
        filtered_labels = []
        for handle, label in zip(handles, labels):
            if label not in ["agent_or_base", "nodes"]:  # Skip the titles
                if label.isdigit():
                    filtered_labels.append(
                        f"$|\\mathcal{{N}}| = {label}$"
                    )  # Replace jammers with |\mathcal{J}|
                else:
                    filtered_labels.append(label)  # Keep other labels as is
                filtered_handles.append(handle)  # Keep the corresponding handle

        # Group the legend entries
        # Separate "Agent" and "Baseline" from the jammers
        agent_base_handles = []
        agent_base_labels = []
        node_handles = []
        node_labels = []

        for handle, label in zip(filtered_handles, filtered_labels):
            if label in ["Agent", "Baseline"]:
                agent_base_handles.append(handle)
                agent_base_labels.append(label)
            else:
                node_handles.append(handle)
                node_labels.append(label)

        # Combine the handles and labels in the desired order
        final_handles = node_handles + agent_base_handles
        final_labels = node_labels + agent_base_labels

        # Create the legend with two columns: one for "Agent"/"Baseline" and one for jammers
        plt.legend(
            final_handles,
            final_labels,
            bbox_to_anchor=(0.5, -0.25),
            loc="upper center",
            ncol=3,  # Use two columns
            title=None,  # Remove the legend title
        )

        # Add space at the bottom of the entire figure
        plt.subplots_adjust(bottom=0.2)  # Add 20% padding at the bottom of the figure

        g.set_axis_labels(
            r"$\text{step }\mathcal{t}$", ""
        )  # x-axis label is $\vert N \vert$, y-axis label is empty

        # Customize y-axis labels for each subplot with LaTeX expressions
        y_labels = {
            "ADT per Step": r"$ADT$",
            "Throughput per Step": r"$\text{Throughput } \mathcal{\tau}$",
        }

        # Apply y-axis labels and scientific notation to each subplot
        for ax, title in zip(g.axes.flat, g.row_names):
            ax.set_ylabel(y_labels.get(title, title))
            ax.yaxis.set_major_formatter(
                ticker.ScalarFormatter(useMathText=True)
            )  # Use scientific notation
            ax.ticklabel_format(
                axis="y", style="sci", scilimits=(0, 0)
            )  # Force scientific notation

        # Apply y-axis labels to each subplot
        for ax, title in zip(g.axes.flat, g.row_names):
            ax.set_ylabel(y_labels.get(title, title))

        # Customize subplot titles
        subplot_titles = ["a)", "b)"]  # Add more if needed
        for i, ax in enumerate(g.axes_dict.values()):
            ax.set_title(subplot_titles[i])

        plt.rcParams["text.usetex"] = False  # Disable LaTeX
        plt.rcParams["mathtext.default"] = "regular"  # Use MathTextS

    # * Special metrics
    def _getNormalizedGain(self, df_throughputs: pd.DataFrame) -> pd.DataFrame:
        """Adds a column with the normalized gain calculated.

        Args:
            df_throughputs (pd.DataFrame): DataFrame with Throughputs

        Returns:
            pd.DataFrame: DataFrame with normalized gain calculated
        """
        # Filter data for jammers transition from 0 to 1
        df_throughputs = df_throughputs[df_throughputs["jammers"].isin([0, 1])]

        # Pivot data for throughput recovery
        normalize_gain_data = df_throughputs.pivot_table(
            index=["episode", "agent_or_base", "nodes", "users"],
            columns="jammers",
            values="value",
        ).reset_index()

        # Rename columns for clarity
        normalize_gain_data.columns = [
            "episode",
            "agent_or_base",
            "nodes",
            "users",
            "throughput_0",
            "throughput_1",
        ]

        # Initialize the normalized_gain column
        normalize_gain_data["normalized_gain"] = 0.0

        # Get baseline throughput_0 for each node/user combination
        baseline_data = normalize_gain_data[
            normalize_gain_data["agent_or_base"] == "Baseline"
        ]

        # Merge the baseline throughput into the main dataframe for alignment
        normalize_gain_data = normalize_gain_data.merge(
            baseline_data[["nodes", "users", "throughput_0"]].rename(
                columns={"throughput_0": "baseline_throughput_0"}
            ),
            on=["nodes", "users"],
            how="left",
        )

        # Calculate normalized_gain for all rows
        normalize_gain_data["normalized_gain"] = (
            normalize_gain_data["baseline_throughput_0"]
            - normalize_gain_data["throughput_1"]
        ) / normalize_gain_data["baseline_throughput_0"]

        # Handle infinities and NaNs using the helper function
        normalize_gain_data = _replace_infs_nans(
            normalize_gain_data, "normalized_gain", 1, 0, 0
        )

        return normalize_gain_data

    def _getThroughputRecoveryFraction(
        self, df_throughputs: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculates the throughput recovery fraction.

        Args:
            df_throughputs (pd.DataFrame): DataFrame with Throughputs

        Returns:
            pd.DataFrame: DataFrame with throughput recovery fraction calculated
        """
        # Filter data for jammers transition from 0 to 1
        df_throughputs = df_throughputs[df_throughputs["jammers"].isin([0, 1])]

        # Pivot data for throughput recovery
        recovery_data = df_throughputs.pivot_table(
            index=["episode", "agent_or_base", "nodes", "users"],
            columns="jammers",
            values="value",
        ).reset_index()

        # Rename columns for clarity
        recovery_data.columns = [
            "episode",
            "agent_or_base",
            "nodes",
            "users",
            "throughput_0",
            "throughput_1",
        ]

        # Calculate recovery as a ratio, avoiding division by zero or infinity issues
        recovery_data["throughput_recovery"] = (
            recovery_data["throughput_1"] / recovery_data["throughput_0"]
        )

        # Handle infinities and NaNs using the helper function
        recovery_data = _replace_infs_nans(
            recovery_data, "throughput_recovery", 1, 0, 0
        )

        return recovery_data

    # * Store the visualization
    def storePlots(self, storage_dir: str) -> None:
        """Stores all plots in memory to the storage directory with the graph name set to the plot title.

        Args:
            storage_dir (str): Directory where the graphs are to be stored.
        """
        ut.create_dirs(storage_dir)
        for i, figure in enumerate(plt.get_fignums()):
            fig = plt.figure(figure)
            if fig._suptitle:
                title = fig._suptitle.get_text()
            elif fig.axes:
                title = fig.axes[0].get_title()
            else:
                title = f"figure_{i}"
            title = title.replace(
                " ", "_"
            )  # Replace spaces with underscores for filenames
            file_path = os.path.join(storage_dir, title)
            fig.savefig(file_path + ".png", dpi=450, bbox_inches="tight")
            fig.savefig(file_path + ".svg", bbox_inches="tight")

    # * Set figures
    def _init_new_figure(self, with_legend_space_below: bool = False) -> None:

        figure_width = 3.5  # ieee column width
        figure_height = 2.6

        if with_legend_space_below:
            figure_height += 0.3  # add space for legend

        plt.figure(figsize=(figure_width, figure_height), constrained_layout=True)


# * Helper methods
def findLogFilesFromPattern(*log_patterns: str) -> list:
    """
    Given a log file pattern, this function will return all the paths to the .csv
    log files matching the log_patterns.

    Args:
        log_patterns (str): pattern to specify log files

    Returns:
        list: log file paths matching the log patterns
    """
    log_paths = []

    for pattern in log_patterns:
        matched_paths = glob.glob(pattern, recursive=True)
        for path in matched_paths:
            if os.path.isdir(path):
                csv_files = glob.glob(os.path.join(path, "*.csv"))
                log_paths.extend(csv_files)
            elif path.endswith(".csv"):
                log_paths.append(path)

    converted_paths = ut.convertPathsToPyPath(*log_paths)

    return converted_paths


def apply_scientific_notation(ax: plt.Axes) -> None:
    """
    Applies scientific notation formatting to both x-axis and y-axis of the given axis.

    Args:
        ax (plt.Axes): The axis to format.
    """
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.xaxis.get_offset_text().set_fontsize(
        10
    )  # Optional: Adjust font size for offset text
    ax.ticklabel_format(
        axis="x", style="sci", scilimits=(0, 0)
    )  # Force scientific notation for x-axis

    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.yaxis.get_offset_text().set_fontsize(
        10
    )  # Optional: Adjust font size for offset text
    ax.ticklabel_format(
        axis="y", style="sci", scilimits=(0, 0)
    )  # Force scientific notation for y-axis


def find_stabilizing_index(
    series: pd.Series, epsilon_factor: float = 0.01
) -> Tuple[int, bool]:
    """Finds the step where the series stabilizes using binary search.
    Works with any Series index, returning the actual index value where stabilization occurs.

    Args:
        series (pd.Series): Series of throughput values (float) per step.
        epsilon_factor (float): Fraction of standard deviation to set epsilon.

    Returns:
        Tuple[int, bool]:
            - int: Index value from which the series stabilizes, or last index if none found.
            - bool: Flag indicating whether stabilization was found (True) or the last index was returned (False).
    """
    # Get the original indices as a list
    indices = series.index.tolist()
    ep_length = len(indices)
    std_value = series.std()
    epsilon = epsilon_factor * std_value

    def is_stable(position: int) -> bool:
        if position >= ep_length:
            return False
        # Use iloc for position-based slicing
        std_current = series.iloc[position:].std()
        mean_current = series.iloc[position:].mean()
        midpoint = position + (ep_length - position) // 2
        std_half = series.iloc[midpoint:].std()
        mean_half = series.iloc[midpoint:].mean()
        return (
            abs(std_current - std_half) < epsilon
            and abs(mean_current - mean_half) < epsilon
        )

    def binary_search(left: int, right: int) -> tuple[int, bool]:
        if left >= right:
            # Return the actual index value and stabilization flag
            if is_stable(left):
                return indices[left], True
            else:
                return indices[ep_length - 1], False
        mid = (left + right) // 2
        if is_stable(mid):
            return binary_search(left, mid)
        return binary_search(mid + 1, right)

    return binary_search(0, ep_length - 1)


def _replace_infs_nans(
    df: pd.DataFrame,
    column: str,
    pos_inf_val: float,
    neg_inf_val: float,
    nan_val: float,
) -> pd.DataFrame:
    """Replaces +inf, -inf, and NaN values in a specified column with given values.

    Args:
        df (pd.DataFrame): DataFrame containing the column.
        column (str): Column name to perform replacements on.
        pos_inf_val (float): Value to replace +inf with.
        neg_inf_val (float): Value to replace -inf with.
        nan_val (float): Value to replace NaNs with.

    Returns:
        pd.DataFrame: DataFrame with replaced values.
    """
    df[column] = df[column].replace(np.inf, pos_inf_val)
    df[column] = df[column].replace(-np.inf, neg_inf_val)
    df[column] = df[column].fillna(nan_val)
    return df


def smooth_moving_average(data: pd.Series, window_size: int) -> pd.Series:
    """
    Smooths the data using a moving average.

    Args:
        data (pd.Series): The data to smooth.
        window_size (int): The size of the moving window.

    Returns:
        pd.Series: Smoothed data.
    """
    return data.rolling(window=window_size, center=True).mean()


JAMMER_DICT: dict = {
    "Static_GreedyJammers": "GSA",
    "Static_ClusterJammers": "LCA",
    "Static_TrafficJammers": "TJ",
}

AGENT_OR_BASE_DICT: dict = {"Agent": "PADRE", "Baseline": "GTM"}


def parse_string(
    input_string: str,
) -> Tuple[int, Union[int, Tuple[int, int]], Union[int, Tuple[int, int]]]:
    """
    Parses a string to extract the number of nodes, jammers, and users.

    Args:
        input_string (str): The input string to parse.

    Returns:
        Tuple[int, Union[int, Tuple[int, int]], Union[int, Tuple[int, int]]]:
        A tuple containing the number of nodes, jammers, and users.
    """
    # Define the regex pattern to extract the values
    pattern = r"(\d+)_\(?(\d+)(?:,\s*(\d+))?\)?_\((\d+),\s*(\d+)\)"
    match = re.match(pattern, input_string)

    if match:
        # Extract values from the regex groups
        numbOfNodes = int(match.group(1))
        numbOfJammers = (
            int(match.group(2))
            if not match.group(3)
            else (int(match.group(2)), int(match.group(3)))
        )
        numbOfUsers = (int(match.group(4)), int(match.group(5)))

        return numbOfNodes, numbOfJammers, numbOfUsers
    else:
        raise ValueError("Input string does not match the expected format.")
