"""
Definitive test of whether the PADRE policy relies on peer-broadcast observations.

Two complementary methods:
  1. Gradient saliency  — |da/dobs| averaged over many rollout steps.
     Near-zero gradients on peer-feature dimensions prove the policy
     output is insensitive to those inputs at the level of the neural network.
  2. Peer-masking ablation — zero out all peer node features, run a full
     evaluation, and compare throughput to the unmasked baseline.
     If throughput is statistically identical, the policy demonstrably
     does not need peer broadcasts to act.

Usage (from resilient-backbone/):
    python src/Utils/peer_feature_sensitivity.py \
        --model "out/models/29_04/PPO_5_1_10_100k_no_obs_attack/rl_model_29999340_steps.zip" \
        --config "src/ubelix_train/29_04_no_obs_attack.yaml" \
        --n-obs 5000 \
        --n-eps 30 \
        --output "out/visualizations/peer_sensitivity"
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from Utils import training_utils as ut
from MANET.ObservationWrappers import NODE_KEYS


# ── Feature layout helpers ────────────────────────────────────────────────────

def get_feature_layout(env):
    """Return per-node feature size and per-feature dim widths from a live env.

    Returns
    -------
    feature_widths : list[int]
        Width (in obs dims) of each enabled NODE_KEY feature.
    feature_labels : list[str]
        Human-readable name for each enabled NODE_KEY feature.
    node_feature_size : int
        Total obs dims belonging to one node.
    n_nodes : int
    obs_size : int
    """
    # Walk the env chain to find the Configurable_ObservationWrapper which
    # exposes a `wrapped_env` attribute pointing to the Base_ObservationWrapper.
    base_wrapper = None
    candidate = env
    for _ in range(20):  # safety bound against infinite loops
        if hasattr(candidate, "wrapped_env"):
            base_wrapper = candidate.wrapped_env
            break
        if hasattr(candidate, "env"):
            candidate = candidate.env
        elif hasattr(candidate, "envs"):
            # SubprocVecEnv — not expected here
            break
        else:
            break

    if base_wrapper is None:
        raise RuntimeError("Could not locate Base_ObservationWrapper in env chain.")

    node_config = base_wrapper.config.get("nodes", {})

    feature_widths = []
    feature_labels = []
    for key in NODE_KEYS:
        if node_config.get(key, False):
            obs_array = base_wrapper.NODE_OBSERVATION_FUNCTIONS[key](
                use_replacement_value=True
            )
            # obs_array has shape (n_nodes, dim_per_node_for_this_feature)
            width_per_node = obs_array.shape[1] if obs_array.ndim == 2 else 1
            feature_widths.append(width_per_node)
            feature_labels.append(key)

    node_feature_size = sum(feature_widths)
    n_nodes = base_wrapper.unwrapped_env.numbOfNodes
    # Total obs = node_obs + user_obs + general_obs
    # For the 29_04 config, user and general obs are all disabled, so obs_size = n_nodes * node_feature_size
    obs_size = base_wrapper.get_observation_size()

    return feature_widths, feature_labels, node_feature_size, n_nodes, obs_size


def build_dim_metadata(feature_widths, feature_labels, node_feature_size, n_nodes, obs_size):
    """Build arrays that label every obs dimension.

    Returns
    -------
    dim_node_idx : np.ndarray[int]  shape (obs_size,)
        Which node (0..n_nodes-1) each obs dim belongs to.
        -1 for user/general dims that don't belong to any node.
    dim_feature_name : list[str]  length obs_size
        Feature name for each obs dim.
    """
    dim_node_idx = np.full(obs_size, -1, dtype=int)
    dim_feature_name = [""] * obs_size

    node_obs_total = n_nodes * node_feature_size

    for node_i in range(n_nodes):
        base_offset = node_i * node_feature_size
        feat_offset = 0
        for feat_idx, (width, label) in enumerate(zip(feature_widths, feature_labels)):
            for d in range(width):
                global_dim = base_offset + feat_offset + d
                dim_node_idx[global_dim] = node_i
                suffix = f"[{d}]" if width > 1 else ""
                dim_feature_name[global_dim] = f"n{node_i}/{label}{suffix}"
            feat_offset += width

    # Remaining dims (user obs, general obs) stay at -1 / ""
    for d in range(node_obs_total, obs_size):
        dim_feature_name[d] = f"other[{d - node_obs_total}]"

    return dim_node_idx, dim_feature_name


# ── Method 1: Gradient saliency ───────────────────────────────────────────────

def compute_gradient_saliency(model, observations, n_nodes, node_feature_size, obs_size):
    """Compute mean |∂action_i / ∂obs_dim| for each action output and obs dim.

    Parameters
    ----------
    observations : np.ndarray  shape (T, obs_size)

    Returns
    -------
    saliency : np.ndarray  shape (n_actions, obs_size)
        Mean absolute gradient of each action output w.r.t. each obs dim.
    """
    policy = model.policy
    policy.eval()

    n_actions = n_nodes * 2  # 2D action per node
    saliency_accum = np.zeros((n_actions, obs_size), dtype=np.float64)
    T = len(observations)

    for t, obs in enumerate(observations):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        obs_t.requires_grad_(True)

        # Forward: extract features → mlp_extractor → action_net
        features = policy.extract_features(obs_t, policy.pi_features_extractor)
        latent_pi, _ = policy.mlp_extractor(features)
        action_mean = policy.action_net(latent_pi)  # shape (1, n_actions)

        for a_idx in range(n_actions):
            if obs_t.grad is not None:
                obs_t.grad.zero_()
            action_mean[0, a_idx].backward(retain_graph=(a_idx < n_actions - 1))
            if obs_t.grad is not None:
                saliency_accum[a_idx] += obs_t.grad[0].abs().detach().cpu().numpy()

    saliency = saliency_accum / T
    return saliency


def own_vs_peer_saliency(saliency, n_nodes, node_feature_size, obs_size):
    """For each node's actions, compute total saliency on own vs peer features.

    Returns
    -------
    own_sal : np.ndarray  shape (n_nodes,)
    peer_sal : np.ndarray  shape (n_nodes,)
    """
    own_sal = np.zeros(n_nodes)
    peer_sal = np.zeros(n_nodes)

    for node_i in range(n_nodes):
        own_start = node_i * node_feature_size
        own_end = own_start + node_feature_size
        own_dims = np.arange(own_start, own_end)
        peer_dims = np.concatenate([
            np.arange(0, own_start),
            np.arange(own_end, n_nodes * node_feature_size),
        ])
        # Action dims for this node: [2*node_i, 2*node_i+1]
        a0, a1 = 2 * node_i, 2 * node_i + 1
        own_sal[node_i] = saliency[a0][own_dims].sum() + saliency[a1][own_dims].sum()
        peer_sal[node_i] = saliency[a0][peer_dims].sum() + saliency[a1][peer_dims].sum()

    return own_sal, peer_sal


# ── Method 2: Peer-masking ablation ───────────────────────────────────────────

class PeerMaskWrapper:
    """Wraps an existing eval environment to zero out peer node features.

    For each step, before returning the observation to the policy, all
    feature dimensions that do NOT belong to "focal_node" are replaced
    with zeros in the node-feature section of the obs vector.

    When focal_node=-1 (all-mask mode), ALL node features are zeroed,
    leaving only user and general obs (if any).  This tests whether the
    policy relies on node features at all.
    """

    def __init__(self, base_env, node_feature_size, n_nodes, focal_node: int = -1):
        self.env = base_env
        self.node_feature_size = node_feature_size
        self.n_nodes = n_nodes
        self.focal_node = focal_node
        self._build_mask()

    def _build_mask(self):
        total_node_dims = self.n_nodes * self.node_feature_size
        self.mask = np.ones(self.env.observation_space.shape[0], dtype=np.float32)

        if self.focal_node == -1:
            # Zero all node features
            self.mask[:total_node_dims] = 0.0
        else:
            # Zero all node features except focal node's block
            self.mask[:total_node_dims] = 0.0
            start = self.focal_node * self.node_feature_size
            end = start + self.node_feature_size
            self.mask[start:end] = 1.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs * self.mask, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs * self.mask, reward, terminated, truncated, info

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space


def run_masked_evaluation(model, masked_env, n_episodes: int, ep_length: int):
    """Run n_episodes with the masked environment and return mean throughput."""
    all_tp = []
    for ep in range(n_episodes):
        obs, _ = masked_env.reset()
        ep_tp = []
        for _ in range(ep_length):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = masked_env.step(action)
            # Throughput is in info or in the reward wrapper; use reward as proxy
            # (SquaredReward ∝ throughput²; use raw step info if available)
            ep_tp.append(reward)
            if terminated or truncated:
                break
        all_tp.append(np.mean(ep_tp))
    return np.mean(all_tp), np.std(all_tp)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_saliency_own_vs_peer(own_sal, peer_sal, output_dir: str):
    n_nodes = len(own_sal)
    x = np.arange(n_nodes)
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, own_sal, width, label="Own features", color="#2196F3")
    ax.bar(x + width / 2, peer_sal, width, label="Peer features", color="#FF5722")

    ax.set_xlabel("Node index")
    ax.set_ylabel("Mean |∂action / ∂obs| (summed over action dims)")
    ax.set_title("Gradient saliency: own vs peer feature dependence")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Node {i}" for i in range(n_nodes)])
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "saliency_own_vs_peer.pdf")
    fig.savefig(path)
    path_png = path.replace(".pdf", ".png")
    fig.savefig(path_png, dpi=150)
    plt.close(fig)
    print(f"Saved: {path_png}")


def plot_saliency_heatmap(saliency, dim_feature_name, n_nodes, node_feature_size,
                          feature_labels, feature_widths, output_dir: str):
    """Heatmap: actions × feature groups (averaged over nodes for peer features)."""
    n_actions = saliency.shape[0]

    # Build per-feature aggregated saliency: for each NODE_KEY feature,
    # sum own-node dims and peer-node dims separately.
    # Rows: action outputs (node 0 ax, ay; node 1 ax, ay; ...)
    # Cols: own_<feature> and peer_<feature>

    col_labels = []
    col_own_mask = []  # True if this column is an own-feature column

    for label, width in zip(feature_labels, feature_widths):
        col_labels.append(f"own/{label}")
        col_own_mask.append(True)
    for label, width in zip(feature_labels, feature_widths):
        col_labels.append(f"peer/{label}")
        col_own_mask.append(False)

    n_cols = len(col_labels)
    agg = np.zeros((n_actions, n_cols), dtype=np.float64)

    for node_i in range(n_nodes):
        a0, a1 = 2 * node_i, 2 * node_i + 1

        # own features (block at node_i * node_feature_size)
        own_base = node_i * node_feature_size
        feat_offset = 0
        for f_idx, (label, width) in enumerate(zip(feature_labels, feature_widths)):
            own_dims = np.arange(own_base + feat_offset, own_base + feat_offset + width)
            agg[a0, f_idx] += saliency[a0][own_dims].sum()
            agg[a1, f_idx] += saliency[a1][own_dims].sum()
            feat_offset += width

        # peer features (all other nodes)
        for peer_j in range(n_nodes):
            if peer_j == node_i:
                continue
            peer_base = peer_j * node_feature_size
            feat_offset = 0
            for f_idx, (label, width) in enumerate(zip(feature_labels, feature_widths)):
                peer_dims = np.arange(peer_base + feat_offset, peer_base + feat_offset + width)
                col_peer_idx = len(feature_labels) + f_idx
                agg[a0, col_peer_idx] += saliency[a0][peer_dims].sum()
                agg[a1, col_peer_idx] += saliency[a1][peer_dims].sum()
                feat_offset += width

    # Normalize each row to [0,1] for visual clarity
    row_max = agg.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    agg_norm = agg / row_max

    row_labels = []
    for node_i in range(n_nodes):
        row_labels.append(f"n{node_i} ax")
        row_labels.append(f"n{node_i} ay")

    fig, ax = plt.subplots(figsize=(max(12, n_cols * 0.7), max(5, n_actions * 0.45)))
    im = ax.imshow(agg_norm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(n_actions))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title("Gradient saliency heatmap (row-normalised)\nYellow = highest saliency, white = zero")

    # Draw vertical separator between own and peer columns
    sep = len(feature_labels) - 0.5
    ax.axvline(sep, color="black", linewidth=2)
    ax.text(sep / 2, -1.5, "OWN", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(sep + (n_cols - sep) / 2, -1.5, "PEER", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#FF5722")

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04, label="Normalised gradient magnitude")
    fig.tight_layout()

    path = os.path.join(output_dir, "saliency_heatmap.pdf")
    fig.savefig(path, bbox_inches="tight")
    path_png = path.replace(".pdf", ".png")
    fig.savefig(path_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path_png}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Peer-feature sensitivity analysis for PADRE policy.")
    p.add_argument("--model", required=True,
                   help="Path to trained SB3 model .zip file.")
    p.add_argument("--config", required=True,
                   help="Path to training YAML config used to train the model.")
    p.add_argument("--n-obs", type=int, default=5000,
                   help="Number of observations to collect for gradient saliency (default 5000).")
    p.add_argument("--n-eps", type=int, default=10,
                   help="Number of episodes for masked ablation (default 10).")
    p.add_argument("--output", default="out/visualizations/peer_sensitivity",
                   help="Output directory for plots.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-ablation", action="store_true",
                   help="Skip the masking ablation (faster, saliency only).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # ── Load config and create environment ───────────────────────────────────
    print("Loading config...")
    config = ut.loadConfig(
        os.path.dirname(args.config), os.path.basename(args.config)
    )

    obs_wrapper_cls = ut.getObservationWrapperFromName(config.get("obs_wrapper", "VS_iCOW"))
    rew_wrapper_cls = ut.getRewardWrapperFromName(config.get("rew_wrapper", "SquaredReward"))
    env_cls = ut.getEnvironmentFromName(config.get("env", "Static_MANETEnv"))
    attacker_cls = ut.getAttackerModelFromName(config.get("attackerModel", "Static_ClusterJammers"))
    routing_fn = ut.getRoutingFunction(config.get("routing_func", "sequential_max_flow"))

    env = ut.createAndWrapEnv(
        env_constructor=env_cls,
        numbOfNodes=config.get("numbOfNodes", 5),
        numbOfJammers=config.get("numbOfJammers", 1),
        numbOfUsers=config.get("numbOfUsers", 10),
        seed=args.seed,
        networkConnectedAtStart=config.get("networkConnectedAtStart", True),
        jammersSpawnNextToUsers=config.get("jammersSpawnNextToUsers", True),
        allow_early_ep_finish=config.get("allow_early_ep_finish", True),
        attackerModel=attacker_cls,
        steps_till_jammer_active=config.get("steps_till_jammer_active", 200),
        step_jammers_start_moving=config.get("step_jammers_start_moving", 100),
        obs_wrapper=obs_wrapper_cls,
        obs_config=config.get("obs_config"),
        rew_wrapper=rew_wrapper_cls,
        ep_length=config.get("ep_length", 100000),
        obs_type="full",
        routing_func=routing_fn,
        step_size=config.get("step_size", 2),
        numbOfByzantineNodes=0,
    )

    # ── Introspect feature layout ─────────────────────────────────────────────
    print("Introspecting feature layout...")
    feature_widths, feature_labels, node_feature_size, n_nodes, obs_size = get_feature_layout(env)
    dim_node_idx, dim_feature_name = build_dim_metadata(
        feature_widths, feature_labels, node_feature_size, n_nodes, obs_size
    )

    print(f"  Nodes: {n_nodes}, obs dims per node: {node_feature_size}, total obs: {obs_size}")
    print(f"  Enabled features per node: {feature_labels}")
    print(f"  Feature widths: {feature_widths}")
    print()
    print("  Own-vs-peer split (structural, not semantic):")
    print(f"    Own  = all {node_feature_size} dims in node i's own block")
    print(f"    Peer = all {(n_nodes-1)*node_feature_size} dims from other nodes' blocks")
    print(f"    Peer fraction of obs: {(n_nodes-1)*node_feature_size / obs_size * 100:.0f}%")
    print()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model from {args.model} ...")
    model = PPO.load(args.model, env=env)

    # ── Collect observations for saliency ────────────────────────────────────
    print(f"Collecting {args.n_obs} observations for gradient saliency...")
    collected_obs = []
    obs, _ = env.reset(seed=args.seed)
    while len(collected_obs) < args.n_obs:
        collected_obs.append(obs.copy())
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    collected_obs = np.array(collected_obs[:args.n_obs])
    print(f"  Collected {len(collected_obs)} observations.")

    # ── Gradient saliency ────────────────────────────────────────────────────
    print("Computing gradient saliency...")
    saliency = compute_gradient_saliency(
        model, collected_obs, n_nodes, node_feature_size, obs_size
    )

    # Aggregate over action outputs
    mean_saliency_per_dim = saliency.mean(axis=0)

    # Own vs peer per node
    own_sal, peer_sal = own_vs_peer_saliency(saliency, n_nodes, node_feature_size, obs_size)

    print()
    print("=== Gradient saliency: own vs peer (summed over action dims) ===")
    print(f"{'Node':<8} {'Own':<18} {'Peer':<18} {'Own / (Own+Peer)':<20} {'Verdict'}")
    print("-" * 72)
    for i in range(n_nodes):
        total = own_sal[i] + peer_sal[i]
        frac = own_sal[i] / total if total > 0 else float("nan")
        verdict = "OWN-DOMINANT" if frac > 0.95 else ("MIXED" if frac > 0.5 else "PEER-DOMINANT")
        print(f"  {i:<6} {own_sal[i]:<18.6f} {peer_sal[i]:<18.6f} {frac:<20.4f} {verdict}")

    print()
    print("Per-feature mean saliency (averaged over all nodes and action dims):")
    print(f"{'Feature':<30} {'Own mean':<18} {'Peer mean':<18}")
    print("-" * 66)

    feat_own = np.zeros(len(feature_labels))
    feat_peer = np.zeros(len(feature_labels))
    feat_offset_list = np.cumsum([0] + feature_widths[:-1])

    for f_idx, (label, width, base_off) in enumerate(
            zip(feature_labels, feature_widths, feat_offset_list)):
        own_total = 0.0
        peer_total = 0.0
        for node_i in range(n_nodes):
            own_dims = np.arange(
                node_i * node_feature_size + base_off,
                node_i * node_feature_size + base_off + width
            )
            peer_dims_list = []
            for node_j in range(n_nodes):
                if node_j == node_i:
                    continue
                peer_dims_list.append(np.arange(
                    node_j * node_feature_size + base_off,
                    node_j * node_feature_size + base_off + width
                ))
            peer_dims = np.concatenate(peer_dims_list) if peer_dims_list else np.array([], dtype=int)

            own_total += mean_saliency_per_dim[own_dims].sum()
            if peer_dims.size:
                peer_total += mean_saliency_per_dim[peer_dims].sum()

        feat_own[f_idx] = own_total / n_nodes
        feat_peer[f_idx] = peer_total / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
        print(f"  {label:<28} {feat_own[f_idx]:<18.8f} {feat_peer[f_idx]:<18.8f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print()
    print("Generating plots...")
    plot_saliency_own_vs_peer(own_sal, peer_sal, args.output)
    plot_saliency_heatmap(
        saliency, dim_feature_name, n_nodes, node_feature_size,
        feature_labels, feature_widths, args.output
    )

    # ── Masking ablation ──────────────────────────────────────────────────────
    if not args.skip_ablation:
        print()
        print("=== Peer-masking ablation ===")
        ep_length = min(config.get("ep_length", 100000), 2000)  # cap for speed

        # Baseline: no masking
        print(f"Running baseline ({args.n_eps} eps)...")
        baseline_mean, baseline_std = run_masked_evaluation(
            model, env, args.n_eps, ep_length
        )
        print(f"  Baseline reward:   {baseline_mean:.4f} ± {baseline_std:.4f}")

        # Peer-masked: zero out all node features except own block (per focal node)
        # We test "all peers masked" for the focal node = 0 (representative)
        masked_rewards = {}
        for focal in range(n_nodes):
            masked_env = PeerMaskWrapper(env, node_feature_size, n_nodes, focal_node=focal)
            m, s = run_masked_evaluation(model, masked_env, args.n_eps, ep_length)
            masked_rewards[focal] = (m, s)
            print(f"  Peer-masked node {focal}: {m:.4f} ± {s:.4f}  "
                  f"(delta = {m - baseline_mean:+.4f})")

        # All-masked: zero out ALL node features
        all_masked_env = PeerMaskWrapper(env, node_feature_size, n_nodes, focal_node=-1)
        all_m, all_s = run_masked_evaluation(model, all_masked_env, args.n_eps, ep_length)
        print(f"  All-node-features masked: {all_m:.4f} ± {all_s:.4f}  "
              f"(delta = {all_m - baseline_mean:+.4f})")

        print()
        print("Interpretation:")
        avg_masked = np.mean([v[0] for v in masked_rewards.values()])
        delta_peer = avg_masked - baseline_mean
        delta_all = all_m - baseline_mean

        if abs(delta_peer) < 0.5 * baseline_std:
            print("  Peer masking produces negligible throughput change.")
            print("  CONCLUSION: Policy does NOT rely on peer features.")
        else:
            print("  Peer masking produces measurable throughput change.")
            print("  CONCLUSION: Policy DOES use peer features to some degree.")

        if abs(delta_all) > 2 * baseline_std:
            print("  Zeroing all node features degrades throughput significantly.")
            print("  CONCLUSION: Policy relies on at least some node features (own or peer).")
        else:
            print("  Zeroing all node features has negligible effect.")
            print("  CONCLUSION: Policy operates without ANY node features (unusual).")

    print()
    print(f"All outputs saved to: {args.output}/")


if __name__ == "__main__":
    main()
