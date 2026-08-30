"""
Verify that obs-manipulation attacks actually change the observation vector.
Run from resilient-backbone/ root:
    python src/Utils/verify_obs_attacks.py
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import MANET.Environments as envs
import MANET.ObservationWrappers as ow


def make_wrapped(byz_nodes, attack_type, **byz_kwargs):
    env = envs.Dynamic_MANETEnv(5, 1, 10)
    env.set_ep_length(500)
    env.set_steps_till_activate_jammers(50)
    env.set_byzantine_params(
        numbOfByzantineNodes=byz_nodes,
        byzantine_attack_type=attack_type,
        **byz_kwargs,
    )
    env.reset(seed=42)
    return ow.VS_iCOW(ow.Base_ObservationWrapper(env))


CASES = {
    "no_byzantine":       dict(byz_nodes=0, attack_type="greyhole"),
    "obs_noise_4.0":      dict(byz_nodes=1, attack_type="obs_noise",           byzantine_obs_noise_std=4.0),
    "obs_capacity_5x":    dict(byz_nodes=1, attack_type="obs_capacity",        byzantine_obs_capacity_factor=5.0),
    "obs_demand_5x":      dict(byz_nodes=1, attack_type="obs_demand",          byzantine_obs_demand_factor=5.0),
    "obs_int_lie_0":      dict(byz_nodes=1, attack_type="obs_interference_lie",byzantine_obs_interference_factor=0.0),
    "obs_user_dir_[4,0]": dict(byz_nodes=1, attack_type="obs_user_dir",        byzantine_obs_user_dir_offset=[4.0, 0.0]),
}

envs_map = {name: make_wrapped(**kwargs) for name, kwargs in CASES.items()}

# Collect observations at reset (step 0) and after 100 steps
results = {}
for name, wrapper in envs_map.items():
    obs_reset, _ = wrapper.reset(seed=42)
    obs_step = obs_reset.copy()
    for _ in range(100):
        obs_step, *_ = wrapper.step(wrapper.action_space.sample())
    results[name] = {"reset": obs_reset, "step100": obs_step}

baseline_reset = results["no_byzantine"]["reset"]
baseline_step  = results["no_byzantine"]["step100"]

print(f"\n{'Attack':<25} {'max_diff@reset':>15} {'max_diff@step100':>16} {'obs_changed':>12}")
print("-" * 72)
for name in CASES:
    if name == "no_byzantine":
        continue
    dr = np.max(np.abs(results[name]["reset"]   - baseline_reset))
    ds = np.max(np.abs(results[name]["step100"] - baseline_step))
    print(f"{name:<25} {dr:>15.6f} {ds:>16.6f} {str(dr > 1e-8 or ds > 1e-8):>12}")

# Also verify Byzantine node type in environment
print("\nByzantine node check:")
for name, kwargs in CASES.items():
    if kwargs["byz_nodes"] == 0:
        continue
    w = envs_map[name]
    env = w.wrapped_env.unwrapped_env
    byz = env.nodes[-1]
    print(f"  {name}: type={type(byz).__name__}  attack_types={getattr(byz, 'attack_types', 'N/A')}")
