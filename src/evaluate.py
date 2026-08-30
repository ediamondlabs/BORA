import argparse
from evaluate_MANETAgents import evaluate_from_config


def main():
    parser = argparse.ArgumentParser(description="Filter agent files")
    parser.add_argument("path_to_config", type=str, help="Path to config file")

    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    evaluate_from_config(**kwargs)


if __name__ == "__main__":
    main()
