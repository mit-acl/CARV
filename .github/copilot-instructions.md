**Repository Overview**

- **Purpose:** Constraint-Aware Refinement for Verification (CARV) — closed-loop reachability, certified/robust controller training and analysis using PyTorch and vendorized provable-robustness libs.
- **Major components:** `nfl_robustness_training/` (experiments, training, analysis scripts), `carv/` (control/constraints primitives), `third_party/` (vendored `auto_LiRPA`, `CROWN-IBP`, `nfl_veripy`).

**Quick Start / Common workflows**

- **Run an example analyzer / quick smoke:** `python3 nfl_robustness_training/src/tester.py` (the script contains quick test/animate entrypoints). Use `--animate` to create GIF/MP4 in `animation_output/`.
- **Run experiments:** examine `nfl_robustness_training/src/run_experiments.py` (it loads params and runs analyses). Most scripts accept `-h`/`--help`; prefer running them locally with trimmed horizons for iteration.
- **Train robust models:** use `nfl_robustness_training/src/train.py`. The script uses `auto_LiRPA` and supports `--bound_type` (`IBP`, `CROWN-IBP`, `CROWN`), `--data`, `--device` and scheduler options.

**Project-specific conventions**

- **Analyzer / ReachableSet pattern:** High-level flows use an `Analyzer` class (see `nfl_robustness_training/src/utils/robust_training_utils.py`) that orchestrates partitioning, refinement and plotting. When changing reachability logic, update `Analyzer` and inspect callers in `run_experiments.py` and `tester.py`.
- **Closed-loop systems and controllers:** `carv/cl_systems/` and `nfl_robustness_training/src/cl_systems` (imported as `cl_systems`) centralize controller definitions and `ClosedLoopDynamics`. Add new controllers there and update loader code used in `run_experiments.py`.
- **Vendored libs:** `third_party/auto_LiRPA`, `third_party/CROWN-IBP`, and `third_party/nfl_veripy` are included. Changes here may need local rebuilds or pip installs; avoid editing unless necessary.
- **Data & outputs:** Persistent experiment outputs are stored under `experimental_data/` and `animation_output/`. Large runs dump pickles into `experimental_data/` — use small horizons for development.

**Integrations & dependencies**

- **PyTorch + auto_LiRPA:** Code relies on `torch` and `auto_LiRPA.BoundedModule/BoundedTensor`. Many files call `model.compute_bounds(...)` and mix IBP/CROWN strategies (see `train.py`).
- **CROWN-IBP:** Training and verification combine IBP and CROWN logic; search for `CROWN-IBP` to find the mixing logic.
- **nfl_veripy:** Used for helper analysers/dynamics; functions live under `third_party/nfl_veripy` and are imported as `nfl_veripy.*`.

**Editing and debugging tips (practical, repo-specific)**

- **Fast iteration:** Reduce `time_horizon` and `partition` sizes in `run_experiments.py` or pass trimmed params into `Analyzer` when developing algorithms.
- **Repro tip:** Many scripts set `device='cpu'` for deterministic runs — switch to `cuda` only after correctness checks.
- **Logging:** `run_experiments.py` calls `suppress_unecessary_logs()` early — if you need more output, search for that call and temporarily comment it.
- **Visual outputs:** To debug plotting/animation, run `python3 nfl_robustness_training/src/tester.py --animate` and inspect `animation_output/`. GIF/MP4 creation relies on `imageio`/`imageio-ffmpeg`.

**Files to read first (high signal-to-noise)**

- `nfl_robustness_training/src/run_experiments.py` — primary experiment runner and parameter handling.
- `nfl_robustness_training/src/train.py` — robust training with auto_LiRPA; shows bound mixing patterns.
- `nfl_robustness_training/src/tester.py` and `nfl_robustness_training/src/TTT_alg.py` — small, self-contained examples used for visualization and algorithm prototyping.
- `nfl_robustness_training/src/utils/robust_training_utils.py` — `Analyzer`, `calculate_reachable_sets`, partition helpers (core algorithms).
- `carv/cl_systems/` — controller definitions and closed-loop wrappers.

**What to avoid / common pitfalls**

- **Editing vendored third-party code** without tests: changes in `third_party/` affect core computations and are hard to validate; prefer wrapping or local overrides.
- **Large default horizons:** many defaults compute long horizons and can run for minutes/hours — shorten parameters while developing.
- **Implicit assumptions about shapes/devices:** many callers assume tensors are on CPU or CUDA in specific places. Use small experiments to confirm device placement.

**If you need to implement a change**

- Start a branch, run `python3 nfl_robustness_training/src/tester.py` to smoke test, then run a trimmed `run_experiments.py` case.
- When adding new controllers, update `carv/cl_systems` and ensure `load_controller(...)` (used by `run_experiments.py`) can find it.
- If modifying analyzer internals, add a small unit-like script under `nfl_robustness_training/src/` that runs the new code with `time_horizon <= 10` and commit that alongside the change.

If any part of this file is unclear or you want more granular instructions for a particular area (training, analyzer internals, or controller integration), tell me which area to expand and I'll iterate.
