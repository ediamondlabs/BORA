import os
from pathlib import Path
import yaml
import itertools
from typing import Dict, Any, List, Tuple, Optional, Union
import glob
from copy import deepcopy

import Utils.training_utils as ut


class ConfigGenerator:
    """Generate configuration files by combining parameter sets and lists."""

    @staticmethod
    def _find_paths_with_type(
        data: Dict[str, Any],
        target_type: Union[type, Tuple[type, ...]],
        parent_key: str = "",
    ) -> List[str]:
        """Find all keys with values of a specific type, including nested ones.

        Args:
            data: Dictionary to search through
            target_type: Type or tuple of types to look for
            parent_key: Parent key string for nested dictionaries

        Returns:
            List of dot-separated paths to values of the target type
        """
        paths = []
        for key, value in data.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, target_type):
                paths.append(full_key)
            elif isinstance(value, dict):
                paths.extend(
                    ConfigGenerator._find_paths_with_type(value, target_type, full_key)
                )
        return paths

    @staticmethod
    def _find_list_paths(data: Dict[str, Any], parent_key: str = "") -> List[str]:
        """Find all keys containing lists, including nested ones."""
        return ConfigGenerator._find_paths_with_type(data, list, parent_key)

    @staticmethod
    def _find_set_paths(data: Dict[str, Any], parent_key: str = "") -> List[str]:
        """Find all keys containing sets, including nested ones."""
        return ConfigGenerator._find_paths_with_type(data, set, parent_key)

    @staticmethod
    def _find_tuple_paths(data: Dict[str, Any], parent_key: str = "") -> List[str]:
        """Find all keys containing tuples, including nested ones."""
        return ConfigGenerator._find_paths_with_type(data, tuple, parent_key)

    @staticmethod
    def _get_nested_value(data: Dict[str, Any], path: str) -> Any:
        """Get a nested value from a dictionary given a dot-separated path.

        Args:
            data: Dictionary to extract value from
            path: Dot-separated path to the value

        Returns:
            The value at the specified path

        Raises:
            KeyError: If the path doesn't exist in the dictionary
        """
        current = data
        for key in path.split("."):
            current = current[key]
        return current

    @staticmethod
    def _set_nested_value(data: Dict[str, Any], path: str, value: Any) -> None:
        """Set a nested value in a dictionary given a dot-separated path.

        Args:
            data: Dictionary to modify
            path: Dot-separated path where the value should be set
            value: Value to set

        Raises:
            KeyError: If the path (except for the last key) doesn't exist
        """
        keys = path.split(".")
        current = data
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = value

    @staticmethod
    def _validate_list_lengths(
        data: Dict[str, Any], list_paths: List[str]
    ) -> Optional[int]:
        """Validate that all lists have the same length and return that length.

        Args:
            data: Dictionary containing the lists
            list_paths: List of paths to list values

        Returns:
            Common length of lists if all have the same length, otherwise None

        Raises:
            AssertionError: If lists don't have the same length
        """
        if not list_paths:
            return 0

        lengths = [
            len(ConfigGenerator._get_nested_value(data, path)) for path in list_paths
        ]

        if len(set(lengths)) > 1:
            raise AssertionError(
                f"Lists must have the same length. Got lengths: {lengths}"
            )

        return lengths[0]

    @staticmethod
    def _deep_copy_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a deep copy of a configuration dictionary.

        Args:
            config: Dictionary to copy

        Returns:
            A deep copy of the dictionary
        """
        return deepcopy(config)

    @staticmethod
    def _extend_log_name_and_set_values(
        config: Dict[str, Any],
        paths: List[str],
        values: Tuple[Any, ...],
        exclude_from_name: List[str] = ["numbOfNodes", "ep_length", "RL_model_name"],
    ) -> str:
        """Set values in config and create a name extension based on those values.

        Args:
            config: Configuration dictionary to modify
            paths: List of paths where values should be set
            values: Values to set at the given paths
            exclude_from_name: Path substrings to exclude from the name extension

        Returns:
            String extension for log name
        """
        extension = ""

        for path, value in zip(paths, values):
            ConfigGenerator._set_nested_value(config, path, value)

            # Only certain values are included in the log name
            should_exclude = any(
                exclude_item in path for exclude_item in exclude_from_name
            )
            if not should_exclude:
                log_name_addition = (
                    process_string(value, max_length=4)
                    if isinstance(value, str)
                    else value
                )
                extension += f"_{log_name_addition}"

        return extension

    @staticmethod
    def generate_configs(
        base_config: Dict[str, Any], output_dir: str, file_name: str = "config"
    ) -> None:
        """Generate configuration files by combining parameter sets and lists.

        Args:
            base_config: Base configuration dictionary
            output_dir: Directory to save generated configurations
            file_name: Base name for generated files

        Raises:
            AssertionError: If lists in the configuration don't have the same length
        """

        base_config = deepcopy(base_config)

        # Create output dir if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all paths with sets and lists
        set_paths = ConfigGenerator._find_set_paths(base_config)
        list_paths = ConfigGenerator._find_list_paths(base_config)
        tuple_paths = ConfigGenerator._find_tuple_paths(base_config)

        if base_config.get("mode", "train") == "train":
            # Path to the log name that needs to be updated
            log_name_path = "storing_and_logging.log_name"

        # Validate and get list length if lists exist
        list_length = ConfigGenerator._validate_list_lengths(base_config, list_paths)

        # Extract list values if they exist
        list_values = []
        if list_length > 0:
            list_values = [
                ConfigGenerator._get_nested_value(base_config, path)
                for path in list_paths
            ]

        # Get all combinations of set values
        set_values = [
            ConfigGenerator._get_nested_value(base_config, path) for path in set_paths
        ]
        all_set_combinations = (
            list(itertools.product(*set_values)) if set_values else [()]
        )

        # Translate tuple paths to lists
        for path in tuple_paths:
            value = ConfigGenerator._get_nested_value(base_config, path)
            ConfigGenerator._set_nested_value(base_config, path, list(value))

        file_idx = 1

        # Generate configurations for each combination
        for i, set_combination in enumerate(all_set_combinations):
            # Start with a clean copy of the base configuration
            config_copy = ConfigGenerator._deep_copy_config(base_config)

            if base_config.get("mode", "train") == "train":
                # Get the original log name
                base_log_name = ConfigGenerator._get_nested_value(
                    config_copy, log_name_path
                )

            # Update values and extend log name for set parameters
            set_extension = ConfigGenerator._extend_log_name_and_set_values(
                config_copy, set_paths, set_combination
            )

            if list_length == 0:
                if config_copy.get("mode", "train") == "train":
                    # Simple case: only set combinations
                    new_log_name = base_log_name + set_extension
                    ConfigGenerator._set_nested_value(
                        config_copy, log_name_path, new_log_name
                    )

                ConfigGenerator.makeTuplesIntoLists(config_copy)
                config_path = output_path / f"{file_name}{file_idx}.yaml"
                with open(config_path, "w") as file:
                    yaml.dump(config_copy, file, default_flow_style=False)

                print(f"Generated: {config_path}")
                file_idx += 1
            else:
                # More complex case: set combinations × list combinations
                for j in range(list_length):
                    # Create a fresh copy for each list combination
                    iteration_config = ConfigGenerator._deep_copy_config(config_copy)

                    # Get values for this list index
                    list_combination = tuple(values[j] for values in list_values)

                    # Update values and extend log name for list parameters
                    list_extension = ConfigGenerator._extend_log_name_and_set_values(
                        iteration_config, list_paths, list_combination
                    )

                    if iteration_config.get("mode", "train") == "train":
                        # Set the final log name
                        new_log_name = base_log_name + set_extension + list_extension
                        ConfigGenerator._set_nested_value(
                            iteration_config, log_name_path, new_log_name
                        )

                    ConfigGenerator.makeTuplesIntoLists(iteration_config)
                    # Save the configuration
                    config_path = output_path / f"{file_name}{file_idx}.yaml"
                    with open(config_path, "w") as file:
                        yaml.dump(iteration_config, file, default_flow_style=False)

                    print(f"Generated: {config_path}")
                    file_idx += 1

    @staticmethod
    def makeTuplesIntoLists(data: Dict[str, Any]) -> None:
        """In a given dictionary, transforms all tuple values into lists

        Args:
            data (Dict[str, Any]): Dictionary with values to transform
        """
        tuple_paths = ConfigGenerator._find_tuple_paths(data)
        # Translate tuple paths to lists
        for path in tuple_paths:
            value = ConfigGenerator._get_nested_value(data, path)
            ConfigGenerator._set_nested_value(data, path, list(value))


class BatchFileHelper:
    """
    A class to generate job scripts for reinforcement learning tasks.
    """

    @staticmethod
    def create_rl_job_scripts_from_search(
        config_dir: str = "src/ubelix_train", hours: int = 72
    ) -> None:
        file_paths: dict = BatchFileHelper.get_file_paths(config_dir)
        python_file_names = file_paths_to_file_name(file_paths["py"])
        yaml_file_names = file_paths_to_file_name(file_paths["yaml"])
        max_count, remaining_indices = BatchFileHelper.find_highest_count()
        count = max_count
        for py_file_name in python_file_names:
            if remaining_indices == []:
                count += 1
                file_count = count
            else:
                file_count = remaining_indices.pop(0)
            BatchFileHelper.create_rl_job_scripts_from_file_name(
                bash_file_index=file_count,
                file_name=py_file_name,
                hours=hours,
                is_config=False,
            )

        for yaml_file_name in yaml_file_names:
            if remaining_indices == []:
                count += 1
                file_count = count
            else:
                file_count = remaining_indices.pop(0)
            config = ut.loadConfig(
                config_dir=config_dir,
                config_name=(
                    yaml_file_name + ".yaml"
                    if not yaml_file_name.endswith(".yaml")
                    else yaml_file_name
                ),
            )
            BatchFileHelper.create_rl_job_scripts_from_file_name(
                bash_file_index=file_count,
                file_name=yaml_file_name,
                hours=hours,
                is_config=True,
                mode=config.get("mode", "train"),
            )

    @staticmethod
    def create_rl_job_scripts_from_params(
        count: int,
        file_name: str,
        start_count: int = 0,
        hours: int = 72,
        is_config: bool = True,
        mode: str = "train",
    ) -> None:
        """
        Creates job scripts for reinforcement learning tasks.

        Args:
            count (int): The number of job scripts to create.
            file_name (str): The base name of the configuration files.
            start_count (int, optional): The starting index for the job scripts. Defaults to 0.
            hours (int, optional): The duration of the job in hours. Defaults to 72.
        """
        # Ensure the directory exists
        os.makedirs("job_scripts", exist_ok=True)

        for i in range(count):
            index = i + start_count

            BatchFileHelper.create_rl_job_scripts_from_file_name(
                bash_file_index=index,
                file_name=f"{file_name}{i+1}",
                hours=hours,
                is_config=is_config,
                mode=mode,
            )

    @staticmethod
    def create_rl_job_scripts_from_file_name(
        bash_file_index: int,
        file_name: str,
        hours: int = 72,
        is_config: bool = True,
        mode: str = "train",
    ) -> None:
        """
        Creates job scripts for reinforcement learning tasks.

        Args:
            count (int): The number of job scripts to create.
            file_name (str): The base name of the configuration files.
            start_count (int, optional): The starting index for the job scripts. Defaults to 0.
            hours (int, optional): The duration of the job in hours. Defaults to 72.
        """
        mode = "train" if mode == "retrain" else mode
        config_command = mode + ".py"

        # Ensure the directory exists
        os.makedirs("job_scripts", exist_ok=True)

        script_file_name = f"job_scripts/python_rl_job{bash_file_index}.sh"
        with open(script_file_name, "w", encoding="utf-8", newline="\n") as file:
            file_content = (
                "#!/bin/bash\n"
                f'#SBATCH --job-name="{file_name}"\n'
                "#SBATCH --partition=epyc2,bdw\n"
                f"#SBATCH --time={hours}:00:00\n\n"
                "#SBATCH --account=projects\n"
                "# Your code below this line\n"
                "eval \"$(~/micromamba shell hook -s bash)\"\n"
                "micromamba activate manet\n"
                
            )
            config_file = (
                f"python src/{config_command} src/ubelix_train/{file_name}.yaml\n"
            )
            python_file = f"python src/ubelix_train/{file_name}.py\n"
            file_content += config_file if is_config else python_file
            file.write(file_content)
            print(f"Generated: {script_file_name}")

    @staticmethod
    def get_batch_commands(count: int, start_count: int = 0) -> None:
        for i in range(count):
            print(f"sbatch job_scripts/python_rl_job{i+start_count}.sh")

    @staticmethod
    def get_file_paths(config_dir: str = "src/ubelix_train") -> dict:
        """
        Find all files matching the given pattern and sort them by their extensions (.py and .yaml).

        Args:
            pattern (str): The glob pattern to search for files.

        Returns:
            dict: A dictionary with keys 'py' and 'yaml', containing lists of file paths.
        """
        config_dir = os.path.join(config_dir, "*")
        # Use glob to find files matching the pattern
        files = glob.glob(config_dir, recursive=True)

        # Filter and sort files by extension
        sorted_files = {
            "py": sorted([file for file in files if file.endswith(".py")]),
            "yaml": sorted([file for file in files if file.endswith(".yaml")]),
        }

        return sorted_files

    @staticmethod
    def find_highest_count() -> Tuple[int, List[int]]:
        """
        Finds the highest index of job script files in the 'job_scripts' folder and
        returns a list of missing indices below the highest index.

        Returns:
            Tuple[int, List[int]]: The highest index and a list of missing indices.
        """
        folder = "job_scripts"
        pattern = "python_rl_job*.sh"
        files = glob.glob(os.path.join(folder, pattern))

        # Extract indices from file names
        indices = []
        for file in files:
            file_name = Path(file).stem  # Get the file name without extension
            if file_name.startswith("python_rl_job"):
                try:
                    index = int(file_name.replace("python_rl_job", ""))
                    indices.append(index)
                except ValueError:
                    continue

        if not indices:
            return 0, []

        highest_index = max(indices)
        all_indices = set(range(1, highest_index + 1))
        missing_indices = sorted(all_indices - set(indices))

        return highest_index, missing_indices


def abbreviate(word: str, max_length: int = 5) -> str:
    """Generate a readable abbreviation for a word, preserving all acronyms.
    Acronyms (all uppercase) stay unchanged; others are abbreviated.

    Args:
        word (str): The word to abbreviate.
        max_length (int, optional): The number of characters the word shouldn't exceed. Defaults to 5.

    Returns:
        str: Abbreviated string.

    Example:
        sequential >> seqe
        VS_COW >> VS_COW
        flow >> flow
        gurobi >> gurb
    """
    # If the word is fully uppercase (an acronym), keep it as-is up to max_length
    if word.isupper():
        return word[:max_length]

    # If word is already short enough, return it
    if len(word) <= max_length:
        return word

    # For non-acronyms: Keep first letter, filter vowels, preserve structure
    vowels = "aeiou"
    abbr = word[0]
    last_char = ""

    for char in word[1:]:
        # Add consonants or meaningful transitions
        if char not in vowels or (last_char in vowels and len(abbr) < max_length):
            abbr += char
        last_char = char

    # Truncate to max_length
    return abbr[:max_length]


# Combined with split/join
def process_string(text: str, max_length: int = 5) -> str:
    """Process a string by abbreviating its parts separted by an underscore.

    Args:
        text (str): A string, here string used to specify an configuration parameter. E.g. "sequential_max_flow"
        max_length (int, optional): Max length of a str after abbreviation. Defaults to 5.

    Returns:
        str: Abbreviated parts. E.g. "seqe_max_flow"
    """
    parts = text.split("_")
    abbr_parts = [abbreviate(part, max_length) for part in parts]
    return "_".join(abbr_parts)


def file_paths_to_file_name(paths: list) -> list[str]:
    """Transform file paths to file names

    Args:
        paths (list): list of paths as str

    Returns:
        list[str]: list of file_names as strings
    """
    return [Path(path).stem for path in paths]
