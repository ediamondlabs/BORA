"""
Stackelberg Byzantine-attacker gym wrapper.

The honest PADRE policy is frozen. On each call to step():
  1. Byzantine PPO agent outputs action in [-1, 1]^(N_byz * 2).
  2. Action is reshaped to (N_byz, 2) and scaled: offset = action * max_offset.
  3. Each Byzantine node's obs_user_dir_offset is set to its offset slice.
  4. The honest policy predicts its action from the current obs.
  5. base_env.step(honest_action) advances the simulation.
  6. Byzantine agent receives reward = -throughput (disruption objective).

obs_scope controls what the Byzantine agent observes:

  "full"  (oracle-info upper bound, default):
      Full CTDE obs vector — all nodes' broadcasts, same as honest policy.

  "local" (Stage A — weakest attacker):
      Each Byzantine node sees only its own per-node obs slice.
      obs_space shape: (N_byz * per_node_dim,).

  "heard" (Stage B):
      Each Byzantine node sees its own obs + the obs slices of all
      in-range neighbors (nodes whose broadcasts it can receive, i.e.
      graph predecessors).  Padded to max_neighbors = n_nodes - 1 with zeros.
      obs_space shape: (N_byz * n_nodes * per_node_dim,).

  "coordinated" (Stage C):
      Same as "heard", but additionally ALL other Byzantine nodes' obs
      are filled in (regardless of graph range) via a covert channel.
      Honest neighbor slots use the same graph-predecessor convention as
      Stage B; remaining slots for out-of-range Byzantine peers are placed
      after heard neighbors.  Shape is identical to Stage B.
      obs_space shape: (N_byz * n_nodes * per_node_dim,).

Requirements on base_env:
  Must have been created with:
    numbOfByzantineNodes = n_byz
    byzantine_attack_type = "obs_user_dir"  (default) or "obs_full"
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from MANET.Components import MANETNode


class ByzantineAttackerEnv(gym.Env):
    """Gym environment that presents the Byzantine PPO agent's MDP.

    Parameters
    ----------
    base_env:
        Fully wrapped PADRE environment (obs + reward wrappers applied).
    honest_policy:
        Frozen PADRE policy (SB3 PPO instance).
        Must support .predict(obs, deterministic=True).
    n_byz:
        Number of Byzantine nodes.  Must equal base_env's numbOfByzantineNodes.
    max_offset:
        Action scaling factor.  Agent action in [-1,1] → offset in
        [-max_offset, max_offset].
    obs_scope:
        "full"  — full CTDE obs vector (oracle-info upper bound, default).
        "local" — each Byzantine node sees only its own per-node obs slice.
        "heard" — own obs + obs of current in-range neighbors (zero-padded).
    attack_type:
        "obs_user_dir" — falsify only the 2D user-direction offset (default, backward-compat).
        "obs_full"     — falsify the entire per-node obs vector; action space is
                         (n_byz * per_node_obs_dim,).
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        base_env: gym.Env,
        honest_policy,
        n_byz: int,
        max_offset: float = 4.0,
        obs_scope: str = "full",
        attack_type: str = "obs_user_dir",
    ) -> None:
        super().__init__()
        if obs_scope not in ("full", "local", "heard", "coordinated"):
            raise ValueError(
                f"obs_scope must be 'full', 'local', 'heard', or 'coordinated', got {obs_scope!r}"
            )
        if attack_type not in ("obs_user_dir", "obs_full"):
            raise ValueError(
                f"attack_type must be 'obs_user_dir' or 'obs_full', got {attack_type!r}"
            )

        self.base_env = base_env
        self.honest_policy = honest_policy
        self.n_byz = n_byz
        self.max_offset = float(max_offset)
        self.obs_scope = obs_scope
        self.attack_type = attack_type

        self._obs: np.ndarray | None = None
        self._byz_indices: list[int] = []

        # Per-node obs layout (populated in _init_local_obs_layout).
        self._n_nodes: int | None = None
        self._per_node_obs_dim: int | None = None
        self._node_obs_total: int | None = None
        # For "heard": node-object → row-index lookup, rebuilt each reset.
        self._node_to_row: dict | None = None

        # Probe per-node layout if needed (non-full obs_scope OR obs_full attack).
        if obs_scope != "full" or attack_type == "obs_full":
            per_node_dim = self._init_local_obs_layout()
        else:
            per_node_dim = None

        # Action space: (n_byz * per_node_dim,) for obs_full, else (n_byz * 2,).
        if attack_type == "obs_full":
            action_dim = n_byz * per_node_dim
        else:
            action_dim = n_byz * 2
        self.action_space: spaces.Box = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32,
        )

        if obs_scope == "full":
            self.observation_space: spaces.Box = base_env.observation_space
        else:
            if obs_scope == "local":
                obs_dim = n_byz * per_node_dim
            else:  # "heard"/"coordinated": own row + up to (n_nodes-1) neighbor rows, padded
                obs_dim = n_byz * self._n_nodes * per_node_dim
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
            )

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        self._obs, info = self.base_env.reset(seed=seed, options=options)
        self._refresh_byz_indices()
        self._zero_offsets()
        if self.obs_scope in ("local", "heard", "coordinated"):
            self._node_to_row = {
                node: idx
                for idx, node in enumerate(self.base_env.unwrapped.nodes)
            }
        return self._get_byz_obs(), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        unwrapped = self.base_env.unwrapped

        if self.attack_type == "obs_full":
            offsets = (
                np.clip(action, -1.0, 1.0).reshape(self.n_byz, self._per_node_obs_dim)
                * self.max_offset
            )
            for i, node_idx in enumerate(self._byz_indices):
                unwrapped.nodes[node_idx].obs_full_offset = offsets[i].astype(np.float64)
        else:
            offsets = (
                np.clip(action, -1.0, 1.0).reshape(self.n_byz, 2) * self.max_offset
            )
            for i, node_idx in enumerate(self._byz_indices):
                unwrapped.nodes[node_idx].obs_user_dir_offset = offsets[i].astype(
                    np.float64
                )

        honest_action, _ = self.honest_policy.predict(
            self._obs, deterministic=True
        )

        self._obs, _, terminated, truncated, info = self.base_env.step(
            honest_action
        )

        # Scale to Mbps to match the honest policy's reward magnitude (RewardWrappers / 1e6).
        throughput = float(unwrapped.network.getThroughput()) / 1e6
        reward = -throughput

        return self._get_byz_obs(), reward, terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self) -> None:
        self.base_env.close()

    # ------------------------------------------------------------------
    # Obs extraction
    # ------------------------------------------------------------------

    def _get_byz_obs(self) -> np.ndarray:
        if self.obs_scope == "full":
            return self._obs

        node_obs = self._obs[:self._node_obs_total].reshape(
            self._n_nodes, self._per_node_obs_dim
        )

        if self.obs_scope == "local":
            return node_obs[self._byz_indices].flatten().astype(np.float32)

        if self.obs_scope == "coordinated":
            return self._get_byz_obs_coordinated(node_obs)

        # "heard": own obs + zero-padded heard neighbor obs
        d = self._per_node_obs_dim
        result = np.zeros((self.n_byz, self._n_nodes * d), dtype=np.float32)
        graph = self.base_env.unwrapped.network.graph

        for i, byz_idx in enumerate(self._byz_indices):
            byz_node = self.base_env.unwrapped.nodes[byz_idx]
            # Own obs always occupies slot 0.
            result[i, :d] = node_obs[byz_idx]
            # Heard: nodes that can reach this Byzantine node (graph predecessors),
            # restricted to MANETNode objects (excludes users).
            heard = [
                n for n in graph.predecessors(byz_node)
                if isinstance(n, MANETNode) and n is not byz_node
            ]
            for slot, heard_node in enumerate(heard, start=1):
                row = self._node_to_row.get(heard_node)
                if row is not None:
                    result[i, slot * d : (slot + 1) * d] = node_obs[row]

        return result.flatten()

    def _get_byz_obs_coordinated(self, node_obs: np.ndarray) -> np.ndarray:
        """Stage C: heard obs + covert channel to ALL other Byzantine nodes.

        Slot 0: own obs (same as Stage B).
        Slots 1..: heard honest/Byzantine predecessors in graph order (same as
        Stage B), then any remaining Byzantine peers not already placed.
        Shape is identical to Stage B — (n_byz * n_nodes * d,).
        """
        d = self._per_node_obs_dim
        result = np.zeros((self.n_byz, self._n_nodes * d), dtype=np.float32)
        graph = self.base_env.unwrapped.network.graph

        for i, byz_idx in enumerate(self._byz_indices):
            byz_node = self.base_env.unwrapped.nodes[byz_idx]
            result[i, :d] = node_obs[byz_idx]

            heard = [
                n for n in graph.predecessors(byz_node)
                if isinstance(n, MANETNode) and n is not byz_node
            ]
            slot = 1
            placed_rows = {byz_idx}
            for heard_node in heard:
                if slot >= self._n_nodes:
                    break
                row = self._node_to_row.get(heard_node)
                if row is not None:
                    result[i, slot * d:(slot + 1) * d] = node_obs[row]
                    placed_rows.add(row)
                    slot += 1

            # Coordination: fill remaining slots with Byzantine peers not yet placed.
            for other_byz_idx in self._byz_indices:
                if slot >= self._n_nodes:
                    break
                if other_byz_idx in placed_rows:
                    continue
                result[i, slot * d:(slot + 1) * d] = node_obs[other_byz_idx]
                placed_rows.add(other_byz_idx)
                slot += 1

        return result.flatten()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_local_obs_layout(self) -> int:
        """Probe the wrapper chain to find per-node obs dimension.

        Walks the env stack to find the Base_ObservationWrapper, resets
        base_env once, calls get_selected_node_observations() to read shape.
        Returns per_node_obs_dim and caches layout attributes.
        """
        env = self.base_env
        cow = None
        while env is not None:
            if hasattr(env, "wrapped_env") and hasattr(
                env.wrapped_env, "get_selected_node_observations"
            ):
                cow = env
                break
            env = getattr(env, "env", None)

        if cow is None:
            raise RuntimeError(
                "Could not find Base_ObservationWrapper in env wrapper chain. "
                "Cannot determine per-node obs dimension."
            )

        self._obs, _ = self.base_env.reset()
        node_obs_matrix = cow.wrapped_env.get_selected_node_observations()
        self._n_nodes = node_obs_matrix.shape[0]
        self._per_node_obs_dim = node_obs_matrix.shape[1]
        self._node_obs_total = self._n_nodes * self._per_node_obs_dim
        return self._per_node_obs_dim

    def _refresh_byz_indices(self) -> None:
        n_total = len(self.base_env.unwrapped.nodes)
        self._byz_indices = list(range(n_total - self.n_byz, n_total))

    def _zero_offsets(self) -> None:
        unwrapped = self.base_env.unwrapped
        for node_idx in self._byz_indices:
            if self.attack_type == "obs_full":
                unwrapped.nodes[node_idx].obs_full_offset = np.zeros(
                    self._per_node_obs_dim, dtype=np.float64
                )
            else:
                unwrapped.nodes[node_idx].obs_user_dir_offset = np.zeros(
                    2, dtype=np.float64
                )


class AdvHonestEnv(ByzantineAttackerEnv):
    """Adversarial honest-training environment — the role-inverse of
    ByzantineAttackerEnv.

    Here the Byzantine policy is FROZEN (a loaded, trained attacker) and the
    honest policy is the one being trained: the gym action/observation are the
    HONEST joint action/observation, so a standard PPO trains the honest swarm.
    On every step the frozen Byzantine policy is run on the Byzantine
    observation to set the per-node falsification offset, exactly as the
    Byzantine attacker would at deployment. Used to train an adversarially
    robust honest policy against the learned (e.g. ``obs_full``) attacker.

    All Byzantine-observation extraction and offset-setting machinery is
    inherited from ByzantineAttackerEnv; only the action direction is flipped.
    """

    def __init__(
        self,
        base_env: gym.Env,
        byz_policy,
        n_byz: int,
        max_offset: float = 4.0,
        obs_scope: str = "heard",
        attack_type: str = "obs_full",
    ) -> None:
        super().__init__(
            base_env,
            honest_policy=None,
            n_byz=n_byz,
            max_offset=max_offset,
            obs_scope=obs_scope,
            attack_type=attack_type,
        )
        self.byz_policy = byz_policy
        # The trainable honest PPO sees the honest joint obs/action, not the
        # Byzantine ones that the parent class exposes.
        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space

    def reset(self, seed=None, options=None):
        self._obs, info = self.base_env.reset(seed=seed, options=options)
        self._refresh_byz_indices()
        self._zero_offsets()
        if self.obs_scope in ("local", "heard", "coordinated"):
            self._node_to_row = {
                node: idx
                for idx, node in enumerate(self.base_env.unwrapped.nodes)
            }
        return self._obs, info

    def step(self, honest_action):
        # Frozen Byzantine policy sets this step's falsification offset.
        byz_obs = self._get_byz_obs()
        byz_action, _ = self.byz_policy.predict(byz_obs, deterministic=True)
        unwrapped = self.base_env.unwrapped
        if self.attack_type == "obs_full":
            offsets = (
                np.clip(byz_action, -1.0, 1.0).reshape(self.n_byz, self._per_node_obs_dim)
                * self.max_offset
            )
            for i, node_idx in enumerate(self._byz_indices):
                unwrapped.nodes[node_idx].obs_full_offset = offsets[i].astype(np.float64)
        else:
            offsets = (
                np.clip(byz_action, -1.0, 1.0).reshape(self.n_byz, 2) * self.max_offset
            )
            for i, node_idx in enumerate(self._byz_indices):
                unwrapped.nodes[node_idx].obs_user_dir_offset = offsets[i].astype(np.float64)

        # The honest action advances the environment; the honest training reward
        # (from the env's reward wrapper) is returned unchanged.
        self._obs, reward, terminated, truncated, info = self.base_env.step(honest_action)
        return self._obs, reward, terminated, truncated, info
