# MANETNodesRelocation — Byzantine-Resilient Node Relocation

This README mirrors the structure of [`tutorial.ipynb`](tutorial.ipynb) — the notebook is
the full, runnable walkthrough; this page is the condensed map. Each section below
corresponds to a section of the tutorial (and to [`docs/Getting Started.md`](docs/Getting%20Started.md)).

## Table of Contents

- [Concept](#concept)
- [Byzantine Attacks](#byzantine-attacks)
- [Installation](#installation)
- [Tutorial](#tutorial)
- [1. Training Agents](#1-training-agents)
- [2. Retraining Agents](#2-retraining-agents)
- [3. Evaluating Agents](#3-evaluating-agents)
- [4. Filtering Agents](#4-filtering-agents)
- [5. Command-Line Training](#5-command-line-training)
- [6. Config & Batch Generation](#6-config--batch-generation)
- [7. Full Parameter List](#7-full-parameter-list)
- [Documentation](#documentation)
- [Tests](#tests)
- [Huggingface - RL Agents and Datasets](#huggingface---rl-agents-and-datasets)

## Concept

**Problem:** A well-connected Mobile Ad Hoc Network (MANET) is used as a backbone for
end-user communication. One or more malicious attackers try to jam the network,
interrupting communication. In addition, some MANET nodes may be **Byzantine** — they
broadcast a *falsified local-state vector* to their neighbours during in-band signalling
to mislead the relocation policy.

**Solution:** MANET nodes are relocated using a data-driven approach (Reinforcement
Learning). The base relocation strategy is **PADRE** (Proactive Decentralized node
Relocation; Leuenberger et al. 2025), trained with Centralized Training / Decentralized
Execution (CTDE). This repository extends PADRE with a study of **Byzantine observation
manipulation**: how a node can perturb its broadcast state to maximally degrade global
throughput, and how a learned attacker (**BORA**) compares to fixed heuristic attacks.

`MANETEnv` is a Gymnasium environment that simulates the scenario, so any RL agent (e.g.
Stable-Baselines3) can be trained on it. The helper scripts described below make training,
retraining, evaluating and filtering models straightforward — you only set the parameters.

## Byzantine Attacks

Attacks are configured via `byzantine_attack_type` in the training/eval YAML (one value
or a list). The authoritative list is `VALID_ATTACK_TYPES` in
[`src/MANET/Components.py`](src/MANET/Components.py).

**Observation-manipulation attacks (primary research focus):**

| `byzantine_attack_type` | Effect | Parameter |
|---|---|---|
| `obs_user_dir` | Lies about direction to closest user/sender/receiver | `byzantine_obs_user_dir_offset` `[dx, dy]` |
| `obs_capacity` / `obs_capacity_deflate` | Scales reported in/out capacity to attract/repel nodes | `byzantine_obs_capacity_factor` |
| `obs_demand` / `obs_demand_deflate` | Scales reported sender/receiver demand ("demand sinkhole") | `byzantine_obs_demand_factor` |
| `obs_interference_lie` | Hides a nearby jammer by scaling reported interference | `byzantine_obs_interference_factor` (`0.0` = fully hidden) |
| `obs_noise` | Adds Gaussian noise to the broadcast state vector | `byzantine_obs_noise_std` |
| `obs_replay` | Rebroadcasts a stale state from N steps ago | `byzantine_obs_replay_delay` (int steps) |
| `obs_full` | Learned full-vector perturbation produced by the BORA attacker | — |

**Routing-layer attacks (legacy / deferred, still supported):** `greyhole`, `sinkhole`,
`selective_jamming`, `position_spoofing`.

See [`src/examples/evaluate/example_obs_manipulation_eval_config.yaml`](src/examples/evaluate/example_obs_manipulation_eval_config.yaml)
for a worked example covering every obs-manipulation knob.

## Installation

The project targets Python 3.11.x — make sure to use the correct interpreter.
Dependencies are managed with Poetry. Either install via a pip that supports PEP 517/518
(`pip install .`) or, preferably, install Poetry (`pip install poetry`) and run
`poetry install` in the root folder. Poetry is the safer option, as `pip install .` can
cause trouble recognizing certain modules. Using `poetry install` also installs the
developer dependencies needed for the `MkDocs` site and the `pytest` tests.

To use the OptimumFinder (a minor part of the project) install the `ipopt` optimizer. If
you use conda, run from the same conda env:
```bash
conda install -c conda-forge ipopt
```

Quick smoke test — train an agent from an
[example config](src/examples/train/example_train_config.yaml):
```bash
python src/train.py src/examples/train/example_train_config.yaml
```
If the GUI opens and the progress bar appears, the install works — you can kill the process.

## Tutorial

[`tutorial.ipynb`](tutorial.ipynb) is the full runnable walkthrough of training,
retraining, evaluating, filtering, and batch generation, with explanations of the
environment and evaluation parameters. The same content rendered as a page lives in
[`docs/Getting Started.md`](docs/Getting%20Started.md), and runnable example scripts for
every workflow are under [`src/examples/`](src/examples/). The sections below summarise it.

## 1. Training Agents

Train via a config file or a direct function call (tutorial §2):

```python
import train_MANETAgent as trainer

# From a YAML config
trainer.train_agent_from_config(config_path="src/examples/train/example_train_config.yaml")

# Or with parameters directly
trainer.train_agent(RL_model_name="PPO", env="Static_MANETEnv", obs_wrapper="VS_iCOW", ...)
```

**Key component options** (tutorial §2.4):

| Component | Choices |
|---|---|
| **Environments** | `Static_MANETEnv`, `Less_Static_MANETEnv`, `Dynamic_MANETEnv` |
| **Attacker (jammer) models** | `Static_ClusterJammers`, `Static_TrafficJammers`, moving variants (`Attackers.py`) |
| **Observation wrappers** | `FS`/`VS` × `COW`/`iCOW` (e.g. `VS_iCOW`); see `ObservationWrappers.py` |
| **Reward wrappers** | `LinearReward`, `SquaredReward`, `MovementCost`, `EvaluationReward` |
| **Routing functions** | exact MCMF (`PuLP`/`Gurobi`/`Xpress`) and faster `sequential_max_flow` |
| **Byzantine attacks** | set `numbOfByzantineNodes` + `byzantine_attack_type` (see [Byzantine Attacks](#byzantine-attacks)) |

Monitor training with TensorBoard on `out/logs/` (tutorial §2.6).

## 2. Retraining Agents

Continue training an existing model (tutorial §3):

```python
trainer.retrain_agent_from_config(config_path="src/examples/retrain/example_retrain_config.yaml")
# or
trainer.retrain_agent(dir_of_agent_to_load="path/to/your/model", ...)
```

## 3. Evaluating Agents

Evaluate trained agents (and non-RL baselines) under chosen conditions (tutorial §4):

```python
from evaluate_MANETAgents import evaluate, evaluate_from_config

evaluate_from_config("src/examples/evaluate/example_eval_config.yaml")
```

or from the terminal:

```bash
python src/evaluate.py src/examples/evaluate/example_eval_config.yaml
```

An eval config sets `agent_patterns` (which models to load), optional `baselines_with_cs`
(baseline agents + condition strategies), and `eval_params` (environment, `obs_type`,
`state_loss_rate`, and any Byzantine attack). To evaluate under attack, see
[`example_obs_manipulation_eval_config.yaml`](src/examples/evaluate/example_obs_manipulation_eval_config.yaml).

## 4. Filtering Agents

Use `evaluate` to rank and keep the best-performing models or agents from a set of
candidates (tutorial §5) — point `agent_patterns` at the candidate models and inspect the
resulting scores.

## 5. Command-Line Training

`train.py` accepts a config file, a folder containing one, or an existing model to
retrain (tutorial §6):

```bash
# From a config file (or a folder containing one)
python src/train.py src/examples/train/example_train_config.yaml

# Retrain an existing model
python src/train.py path/to/your/model/rl_model_x_steps.zip --ep_length 11000
```

See `python src/train.py --help` for all command-line arguments.

## 6. Config & Batch Generation

Generate many configs and cluster job scripts programmatically (tutorial §7):

```python
from src.Utils.batch_utils import ConfigGenerator as CG, BatchFileHelper as BFH

# Cartesian-product of parameter dicts → multiple training configs
CG.generate_configs(train_dict)

# Create Slurm job scripts for the generated configs
BFH.create_rl_job_scripts_from_params(...)
```

The Slurm scripts that produced the **thesis** results — and the order to run them in —
are indexed in [`job_scripts/README.md`](job_scripts/README.md). The learned (BORA)
attacker is trained/evaluated via
[`src/Utils/train_learned_byzantine.py`](src/Utils/train_learned_byzantine.py), and an
attack-type sweep + harm-curve plot via
[`src/Utils/run_byzantine_sweep.py`](src/Utils/run_byzantine_sweep.py).

## 7. Full Parameter List

The complete set of training, retraining, and command-line options is tabulated in
[`docs/Getting Started.md` §8](docs/Getting%20Started.md) (tutorial §8), and the component
distribution model in tutorial §9.

## Documentation

API docs are built with `MkDocs`. Build and serve locally with:
```bash
mkdocs build
mkdocs serve
```

## Tests

Behaviour is covered by `pytest` tests with ~70% coverage of the
[main components](src/MANET/). Run them with:
```bash
pytest tests/
```

## Huggingface - RL Agents and Datasets

The RL agents and resulting datasets are hosted on Huggingface:

- [Models](https://huggingface.co/Thesis-MANETNodesRelocation/MANETRelocator)
- [Datasets](https://huggingface.co/datasets/Thesis-MANETNodesRelocation/MANETRelocator)

Both can be downloaded directly or with Git. Models:
```bash
git clone https://huggingface.co/Thesis-MANETNodesRelocation/MANETRelocator
```
Datasets (run from a different directory than the models):
```bash
git clone https://huggingface.co/datasets/Thesis-MANETNodesRelocation/MANETRelocator
```

Copy agents from `upload-v3.2-stable/best_agents/**` to `out/best_agents/**`, and datasets
from `upload-v3.2-stable/datasets/**` to `out/evaluations/**` (extract the `.7z` archives
first). The visualization scripts under [`src/Utils/visualize/`](src/Utils/visualize/)
then regenerate the thesis plots — run them from the `resilient-backbone/` root so the
relative `out/` paths resolve.
