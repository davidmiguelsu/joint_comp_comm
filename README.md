# Joint Communication–Computing Energy Optimization

This repository provides the reference implementation and codebase for the paper *"Joint Communication-Computation Orchestration via UE Deadline Partitioning in MEC-Enabled CF-mMIMO Architecture"* by Georgios Kontos, David Rodrigues, Prodromos Makris, Konstantinos Christodoulopoulos, and Emmanouel Varvarigos.

It includes comprehensive configuration variables, scripts for generating the Cell-Free (CF) massive MIMO environment (including channel realizations and user setups), tools for evaluating Spectral Efficiency (SE) utilizing Minimum Mean Square Error (MMSE) combining, scripts to model the Multi-Access Edge Computing (MEC) network, and implementations of the reference heuristics $H_c$ and $H_e$.

The framework jointly allocates an end-to-end deadline between an **uplink
communication phase** (served by a cell-free massive-MIMO network) and an
**edge-computation phase** (served by a pool of MEC servers), and minimizes the
total infrastructure + computation energy subject to per-UE deadlines.

For each user (UE) `k`, a split fraction `a_k ∈ [0, 1]` assigns `a_k · D_k`
seconds to communication and `(1 − a_k) · D_k` seconds to computation, where
`D_k` is the overall UE end-to-end deadline. The goal is to choose the split vector `a = (a_1, …, a_K)`
that minimizes energy while minimizing deadline violations.

## Repository structure

| File                      | Role                                                                                                                                                    |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.yaml`           | Single source of truth for**both system models** — the cell-free network and the edge-server pool — plus task profiles and evaluation settings. |
| `cellfree_mimo.py`      | Cell-free massive-MIMO channel generation, estimation, and spectral-efficiency computation.                                                             |
| `comm_energy_model.py`  | **Communication energy model** (AP power, antenna power).                                                                                         |
| `edge_compute_model.py` | **Computation energy model** + the **edge-computation heuristic H_e** (server selection + DVFS).                                            |
| `comm_heuristic.py`     | **Communication heuristic H_c** (reuse-aware AP clustering).                                                                                      |
| `joint_evaluator.py`    | Builds the network+task snapshot and evaluates a split vector`a`: runs both heuristics and returns energy + violation (if any).                       |
| `run_sweep.py`          | Evaluates a uniform split`a` over a range of values (or a single `a`).                                                                              |
| `run_vector.py`         | Evaluates a custom per-UE split vector and prints a per-UE breakdown.                                                                                   |
| `run_comparison.py`     | Results table comparing **Fair vs. Random Search**.                                                                                              |

## Installation

Requires Python 3.9+ and the packages listed in `requirements.txt`
(`numpy`, `scipy`, `PyYAML`):

```bash
pip install -r requirements.txt
```

## Quick Start Example

If you have just downloaded the repository, follow this  workflow to get familiar with the framework:

1. **Review the Configuration:**
   Start by opening `config.yaml`. This file contains the complete system parameters (cell-free network, MEC servers, and user task profiles). You can leave it as default to replicate the results of the paper.
2. **Generate the Environment & Run a Baseline:**
   The easiest way to see the framework in action is to evaluate a simple uniform deadline split (e.g., `a = 0.5`, meaning exactly 50% of the deadline is given to communication and 50% to computation for all users).

   Run the following command:

   ```bash
   python run_sweep.py --a 0.5
   ```

   *What happens?* On the first run, the code generates the Cell-Free MIMO channels and user task profiles based on `config.yaml`, and caches this environment snapshot in the `data/` folder. It then evaluates both the communication ($H_c$) and computation ($H_e$) heuristics, returning the system's total energy consumption and (potential) violations.
3. **Compare Strategies:**
   The comparison script evaluates a uniform baseline against a random search optimization over the split vector space:

   ```bash
   python run_comparison.py
   ```

   It outputs a summary table.

## Detailed Usage

All runners read the scenario from `config.yaml`. The first run builds and
caches the network/task snapshot under `data/` (created automatically); later
runs reuse it.

**Uniform split sweep** — per-plane energy (communication / computation),
penalty, feasibility, and the per-plane resource decisions (active APs,
assigned cores, cumulated frequency) for each split value:

```bash
python run_sweep.py              # sweep a = 0.1 .. 0.9
python run_sweep.py --a 0.5      # single uniform split for all UEs (fair scenario)
python run_sweep.py --start 0.2 --stop 0.8 --step 0.05
```

**Custom per-UE split vector** — one value per UE (length must equal `K`);
prints a per-UE breakdown (upload/execution times, achieved vs. required SE,
serving cluster size, assigned server/cores/frequency), a per-server
utilization table, and an energy summary split by component:

```bash
python run_vector.py --a 0.5,0.6,0.3,0.9,...   # K comma-separated values
python run_vector.py --file my_split.txt        # or read values from a file
```

**Comparison table** — Fair vs. Random Search:

```bash
python run_comparison.py
```

## Configuration

All scenario parameters live in `config.yaml`:

- `network:` — cell-free system model (area, number of APs `L`, UEs `K`,
  antennas per AP `N`, bandwidth, pilots, channel realizations, power
  control, …).
- `task_profiles:` — UE workload mix (data size and processing intensity).
- `deadline_model:` — nominal upload/compute baselines and skew range used to
  reverse-engineer per-UE end-to-end deadlines.
- `edge_servers:` — edge-computation system model (per-server cores, frequency
  range, energy coefficient `kappa`).
- `ablation:` — per-plane heuristic selection (`v6`/`unaware` for the
  communication plane, `greedy`/`first_fit` for the computation plane).
- `comm_energy_params:` / `antenna_power_params:` / `gops_params:` —
  communication energy-model coefficients.
- `evaluation:` — penalty coefficient `lambda_penalty` used to score a split.
- `random_search:` — number of samples, seed, and split bounds for the random
  baseline in `run_comparison.py`.

## Reproducibility

The network channel, task set, and random baseline are all seeded (see
`random_search.seed` in `config.yaml` and the seed arguments to
`get_or_build_snapshot` in `joint_evaluator.py`). The first run builds and
caches a snapshot under `data/` (created automatically); subsequent runs reuse
it.

## License

This project is released under the [MIT License](LICENSE). You are free to
use, modify, and redistribute the code; if you use it in academic or
scientific work, please cite the paper below.

## Citation

```bibtex
@article{kontos_joint_comm_comp,
  title   = {Joint Communication-Computation Orchestration via UE Deadline Partitioning in MEC-Enabled CF-mMIMO Architecture},
  author  = {Georgios Kontos and David Rodrigues and Prodromos Makris and Konstantinos Christodoulopoulos and Emmanouel Varvarigos},
  journal = {<venue>},
  year    = {<year>}
}
```
