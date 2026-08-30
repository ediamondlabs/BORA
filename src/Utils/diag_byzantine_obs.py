"""
Diagnostic: does the obs_capacity attack actually affect the observation vector
seen by the policy at evaluation time?

Run with:
    /home/aki/Documents/Studium/Git_PADRE/.venv/bin/python src/Utils/diag_byzantine_obs.py

Steps:
1. Create the environment exactly as run_byzantine_sweep.py does (obs_type="real", 1 Byzantine node).
2. Reset and run a few steps.
3. At each step print:
   a. The type of each node in unwrapped_env.nodes
   b. The type of each node in Real_ObservationWrapper.wrapped_env.nodes
   c. The value of getattr(node, "observed_capacity_factor", 1.0) for each
   d. Obs vector slice for the Byzantine node (dims 72-89) in attack vs no-attack

This tells us definitively whether the attack is reaching the policy input.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np

# ---- config (match the sweep exactly) ----
MODEL_PATH = "out/models/29_04/PPO_5_1_10_100k_no_obs_attack/rl_model_29999340_steps.zip"
CONFIG_PATH = "src/ubelix_train/29_04_no_obs_attack.yaml"
N_STEPS = 5
SEED = 42

# ---- imports ----
import yaml
from src.Utils import training_utils as ut
from src.MANET import ObservationWrappers as ow


def make_env(attack_type, capacity_factor, numbOfByzantineNodes):
    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    nested = ut.loadNestedConfig(raw)

    env = ut.createEnv(
        env_constructor=ut.getEnvironmentFromName(nested["env_constructor"]),
        numbOfNodes=nested["numbOfNodes"],
        numbOfJammers=nested["numbOfJammers"],
        numbOfUsers=nested["numbOfUsers"],
        seed=SEED,
        networkConnectedAtStart=nested.get("networkConnectedAtStart", True),
        jammersSpawnNextToUsers=nested.get("jammersSpawnNextToUsers", True),
        attackerModel=ut.getAttackerFromName(nested.get("attacker_model", "Static_ClusterJammers")),
        allow_early_ep_finish=nested.get("allow_early_ep_finish", False),
        ep_length=nested.get("ep_length", 100_000),
        steps_till_jammer_active=nested.get("steps_till_jammer_active", 200),
        routing_func=None,
        step_size=nested.get("step_size", 1),
        numbOfByzantineNodes=numbOfByzantineNodes,
        byzantine_attack_type=attack_type,
        byzantine_obs_capacity_factor=capacity_factor,
    )
    return env


def wrap_env_for_node(env, node_idx, obs_config):
    base_wrapper = ow.Real_ObservationWrapper(
        env=env.unwrapped,
        config=obs_config,
        node_idx=node_idx,
        state_loss_rate=0.0,
    )
    outer = ow.VS_iCOW(base_wrapper)
    return base_wrapper, outer


def print_node_types(label, nodes):
    types = [type(n).__name__ for n in nodes]
    factors = [getattr(n, "observed_capacity_factor", "N/A (not Byzantine)") for n in nodes]
    attack_types = [getattr(n, "attack_types", None) for n in nodes]
    print(f"  [{label}]")
    for i, (t, f, at) in enumerate(zip(types, factors, attack_types)):
        print(f"    node {i}: type={t}, observed_capacity_factor={f}, attack_types={at}")


def run_diagnostic():
    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    nested = ut.loadNestedConfig(raw)
    obs_config = nested.get("obs_config", {})

    print("=" * 70)
    print("CASE A: no Byzantine nodes")
    print("=" * 70)
    env_clean = make_env("greyhole", 1.0, 0)
    base_clean, outer_clean = wrap_env_for_node(env_clean, 0, obs_config)
    env_clean.reset(seed=SEED)
    base_clean.initializeAllComponents()

    print(f"\nAfter reset (ep_step={env_clean.ep_step}):")
    print("  unwrapped_env.nodes:")
    print_node_types("unwrapped", env_clean.unwrapped.nodes)
    print("  Real_ObservationWrapper.nodes:")
    print_node_types("wrapper", base_clean.nodes)

    obs_clean_list = []
    for step in range(N_STEPS):
        obs = outer_clean.get_wrapper_attr("observation")()
        obs_clean_list.append(obs.copy())
        action = np.zeros((env_clean.numbOfNodes, 2))
        env_clean.unwrapped.step(action)
        print(f"\nStep {step+1} (ep_step now={env_clean.ep_step}):")
        print("  wrapper.nodes after updateComponents:")
        print_node_types("wrapper", base_clean.nodes)

    print("\n" + "=" * 70)
    print("CASE B: 1 Byzantine node, obs_capacity, factor=2.0")
    print("=" * 70)
    env_byz = make_env("obs_capacity", 2.0, 1)
    base_byz, outer_byz = wrap_env_for_node(env_byz, 0, obs_config)
    env_byz.reset(seed=SEED)
    base_byz.initializeAllComponents()

    print(f"\nAfter reset (ep_step={env_byz.ep_step}):")
    print("  unwrapped_env.nodes:")
    print_node_types("unwrapped", env_byz.unwrapped.nodes)
    print("  Real_ObservationWrapper.nodes:")
    print_node_types("wrapper", base_byz.nodes)

    obs_byz_list = []
    for step in range(N_STEPS):
        obs = outer_byz.get_wrapper_attr("observation")()
        obs_byz_list.append(obs.copy())
        action = np.zeros((env_byz.numbOfNodes, 2))
        env_byz.unwrapped.step(action)
        print(f"\nStep {step+1} (ep_step now={env_byz.ep_step}):")
        print("  wrapper.nodes after updateComponents:")
        print_node_types("wrapper", base_byz.nodes)

    print("\n" + "=" * 70)
    print("OBS VECTOR COMPARISON (node 4 block = dims 72-89)")
    print("=" * 70)
    for step in range(N_STEPS):
        obs_c = obs_clean_list[step]
        obs_b = obs_byz_list[step]
        diff = np.abs(obs_b - obs_c)
        print(f"\nStep {step}:")
        print(f"  clean obs[72:90]: {obs_c[72:90]}")
        print(f"  byz   obs[72:90]: {obs_b[72:90]}")
        print(f"  max |diff|: {diff.max():.6f}")
        if diff.max() > 1e-6:
            print("  DIFFERENT: attack IS visible in observation")
        else:
            print("  IDENTICAL: attack NOT visible in observation")


if __name__ == "__main__":
    run_diagnostic()
