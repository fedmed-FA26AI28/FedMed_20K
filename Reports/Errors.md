# Master Project Error & Issue Registry

> **Repository:** `FedMedAI_Study`  
> **Status:** Active & Maintained  
> **Location:** `Reports/Errors.md`  
> **Scope:** Internal Project Issues (Codebase, Algorithms, Architectures, Pipelines & Math)  
> **Last Updated:** 2026-10-25  

This document is the **internal issue and fix log** for the `FedMedAI_Study` project. It records all bugs, algorithmic discrepancies, numerical instabilities, architectural hazards, and pipeline errors originating directly from the project's own codebase, along with their root causes, fixes, and verification proofs.

---

## Quick Navigation & Issue Matrix

| Issue ID | Affected Module | Severity | Core Problem | Status |
| :--- | :--- | :--- | :--- | :--- |
| [**ERR-001**](#err-001-pathological-partition-sample-dropping--class-starvation) | `datasets/partition.py` | 🔴 Critical | Class starvation and dropped samples in pathological partition | `✅ RESOLVED` |
| [**ERR-002**](#err-002-rng-state-mutation-causing-train-partition-divergence) | `datasets/partition.py` | 🟡 Medium | Training splits differed when toggling `local_test` | `✅ RESOLVED` |
| [**ERR-003**](#err-003-partition-generation-bottleneck-via-repeated-pil-transforms) | `datasets/partition.py` | 🟡 Medium | 35s+ partition creation from repeated `__getitem__` calls | `✅ RESOLVED` |
| [**ERR-004**](#err-004-single-sample-batch-tensor-collapse-in-batchnorm-layers) | `datasets/medmnist_code.py` | 🟠 High | `ValueError: Expected batch_size (1) to match (0)` on small shards | `✅ RESOLVED` |
| [**ERR-005**](#err-005-client-lr-scheduler-ephemeral-scope--server-override) | `client/client.py`, `server/server.py` | 🟠 High | Client LR stuck at 0.001 despite `ReduceLROnPlateau` | `✅ RESOLVED` |
| [**ERR-006**](#err-006-flower-server-early-stopping-loop-termination-failure) | `server/server.py` | 🟡 Medium | Server failed to stop when early stopping condition was met | `✅ RESOLVED` |
| [**ERR-007**](#err-007-fednova-negative-variance--nan-loss-on-int64-tracking-buffers) | `algorithms/fednova.py`, `models/cnn.py` | 🔴 Critical | FedNova produced `NaN` test loss under non-IID partitions | `✅ RESOLVED` |
| [**ERR-008**](#err-008-scaffold-round-2-collapse-under-adaptive-optimizers-option-1-flaw) | `client/client.py`, `algorithms/scaffold.py` | 🔴 Critical | SCAFFOLD collapsed to 8.30% accuracy under Adam in Round 2 | `✅ RESOLVED` |
| [**ERR-009**](#err-009-scaffold-velocity-amplification-under-sgd-momentum-10x-drift) | `client/client.py` | 🔴 Critical | Loss exploded to $10^{23}$ when combining SCAFFOLD with SGD momentum | `✅ RESOLVED` |
| [**ERR-010**](#err-010-silent-fedbn-dormancy-local_bn--0-on-baseline-architecture) | `models/baseline.py`, `algorithms/proposed.py` | 🟠 High | FedBN was inactive (`local_bn=0`) because baseline CNN lacked BN layers | `✅ RESOLVED` |
| [**ERR-011**](#err-011-proposed-strategy-drift-conflict-on-unbalanced-partitions) | `algorithms/proposed.py` | 🟠 High | Unweighted SCAFFOLD $c$ update caused divergence under FedNova scaling | `✅ RESOLVED` |
| [**ERR-012**](#err-012-silent-architectural-mismatch-via-hardcoded-model-fallback) | `algorithms/base.py`, `algorithms/fed*.py` | 🟠 High | Strategies defaulted to `CNN(num_classes=8)` masking wrong model/classes | `✅ RESOLVED` |
| [**ERR-013**](#err-013-local-model-overfitting-in-deep-federated-training-400-epochs) | `datasets/medmnist_code.py` | 🟠 High | Test loss diverged in rounds 40–80 due to missing data augmentation | `✅ RESOLVED` |
| [**ERR-014**](#err-014-flower-server-distributed-evaluation-indefinite-hang) | `server/server.py` | 🟠 High | Server hung forever if a client stalled during `evaluate_round` | `✅ RESOLVED` |
| [**ERR-015**](#err-015-evaluation-metric-ambiguity-macro-average-vs-micro-average) | `algorithms/base.py`, `monitoring/metrics.py`| 🟡 Medium | Ambiguity between client fairness (BMCTA) and total accuracy (BTA) | `✅ RESOLVED` |

---

## Detailed Issue Reports & Solutions

### ERR-001: Pathological Partition Sample Dropping & Class Starvation
- **Affected Module:** [`datasets/partition.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/datasets/partition.py)
- **Session / Date:** `fe15b7e7-4f37-4407-883f-ac53754cf865` | 2026-10-10
- **Severity:** 🔴 Critical
- **Symptoms:**
  Generating pathological partitions on datasets with 9 classes (PathMNIST) or 11 classes (OrganAMNIST) for 12 clients crashed with:
  ```text
  AssertionError: Partition sample count mismatch! Expected 89996, got 82450.
  ```
- **Root Cause:**
  The FedSLD pathological rule requires each client to hold samples from exactly 2 classes ($2 \times 12 = 24$ total assignments). When randomly assigning classes without replacement, the sampling loop occasionally exhausted available classes before all 12 clients received their second class, leaving unassigned class indices completely discarded.
- **Resolution / Fix:**
  Implemented a rejection-sampling loop (up to 1,000 attempts) with a deterministic round-robin fallback guarantee in [`datasets/partition.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/datasets/partition.py):
  ```python
  # Rejection-sampling retry loop to guarantee valid 2-class allocation
  for attempt in range(1000):
      class_assignments = assign_two_classes_per_client(num_clients, num_classes)
      if all_classes_covered(class_assignments, num_classes):
          break
  else:
      # Deterministic fallback guaranteeing zero dropped samples
      class_assignments = deterministic_coverage_assignment(num_clients, num_classes)
  ```
- **Verification:** Verified 100% sample preservation across all MedMNIST datasets with exactly 2 classes per client.

---

### ERR-002: RNG State Mutation Causing Train Partition Divergence
- **Affected Module:** [`datasets/partition.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/datasets/partition.py)
- **Session / Date:** `2c0fd6de-8b6f-48bb-850d-12b69f0a164e` | 2026-10-17
- **Severity:** 🟡 Medium
- **Symptoms:**
  Executing simulations with `--local_test` produced slightly different training client sample counts and convergence trajectories compared to `--no-local_test`, even when using the identical random seed (`--seed 42`).
- **Root Cause:**
  When `test_dataset` was passed, test partition generation called `np.random.shuffle()` inside the same loop using NumPy's global PRNG state. This consumed random numbers and shifted the PRNG sequence for subsequent training client allocations.
- **Resolution / Fix:**
  Decoupled training and test partitioning into two independent deterministic phases using dedicated PRNG instances:
  - **Phase 1 (Train):** Uses `rng = np.random.RandomState(seed)` and records exact client class shard assignments.
  - **Phase 2 (Test):** Uses `test_rng = np.random.RandomState(seed)` and applies the recorded assignments to test splits.
- **Verification:** Training partitions proved 100% bit-for-bit identical whether `local_test` is enabled or disabled.

---

### ERR-003: Partition Generation Bottleneck via Repeated PIL Transforms
- **Affected Module:** [`datasets/partition.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/datasets/partition.py)
- **Session / Date:** `2c0fd6de-8b6f-48bb-850d-12b69f0a164e` | 2026-10-17
- **Severity:** 🟡 Medium
- **Symptoms:**
  Partition generation took 35–45 seconds every time a simulation script started up on PathMNIST.
- **Root Cause:**
  Label indexing used naive item iteration: `labels = [dataset[i][1] for i in range(len(dataset))]`. This called `dataset.__getitem__()` 90,000 times, triggering full torchvision PIL image decoding and tensor conversions 4 to 5 times.
- **Resolution / Fix:**
  Created `_extract_labels()` helper that directly accesses the underlying `.labels` or `.targets` NumPy arrays in memory without loading images:
  ```python
  def _extract_labels(dataset):
      if hasattr(dataset, "labels"):
          return np.asarray(dataset.labels).flatten()
      if hasattr(dataset, "targets"):
          return np.asarray(dataset.targets).flatten()
      return np.array([dataset[i][1] for i in range(len(dataset))])
  ```
- **Verification:** Partition creation time dropped from 35s to **<0.01s** (>4,000× speedup).

---

### ERR-004: Single-Sample Batch Tensor Collapse in BatchNorm Layers
- **Affected Module:** [`datasets/medmnist_code.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/datasets/medmnist_code.py)
- **Session / Date:** `a8b38674-d84c-4b1f-af26-8675947d1460` | 2026-10-03
- **Severity:** 🟠 High
- **Symptoms:**
  Clients with small non-IID shards crashed during local training with:
  ```text
  ValueError: Expected input batch_size (1) to match target batch_size (0)
  # or PyTorch BatchNorm:
  ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, ...])
  ```
- **Root Cause:**
  When a client's dataset size divided by batch size left a remainder of 1, DataLoader emitted a final batch of size 1. Batch normalization requires at least 2 samples per batch during training to compute running variance ($N - 1 = 0$).
- **Resolution / Fix:**
  Configured `drop_last=True` for client training DataLoaders whenever `len(dataset) > batch_size`. Added defensive guards before model forward passes.
- **Verification:** Clients with small partitions (e.g., 33 samples) trained smoothly without batch collapse.

---

### ERR-005: Client LR Scheduler Ephemeral Scope & Server Override
- **Affected Modules:** [`client/client.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/client/client.py), [`server/server.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/server/server.py)
- **Session / Date:** `686b19be-ac7e-44e5-ae08-c3f76b72cd48` | 2026-09-30
- **Severity:** 🟠 High
- **Symptoms:**
  Client learning rates remained stuck at `0.001000` across all rounds despite configuring `torch.optim.lr_scheduler.ReduceLROnPlateau`.
- **Root Cause:**
  1. `ReduceLROnPlateau` was instantiated locally inside `_local_train()`, discarding state and patience counters after each round.
  2. The Flower server sent `learning_rate: 0.001` in `fit_config` every round, unconditionally overriding client optimizer param groups back to `0.001`.
  3. The local scheduler monitored local training loss (which converged to >98% easily) rather than global evaluation loss.
- **Resolution / Fix:**
  1. Implemented server-side step decay in [`server/server.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/server/server.py) (`_make_on_fit_config_fn`):
     $$\text{lr}_t = \text{learning\_rate} \times \left(\text{lr\_decay\_gamma}^{\lfloor (t - 1) / \text{lr\_decay\_steps} \rfloor}\right)$$
  2. Provided three decoupled modes: `client_loss`, `fixed`, and `server_decay`.
  3. In `server_decay`, the server sends the mathematically scheduled LR to clients at each round.
- **Verification:** Verified step decay curves in CSV metrics and JSON telemetry exports.

---

### ERR-006: Flower Server Early Stopping Loop Termination Failure
- **Affected Module:** [`server/server.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/server/server.py)
- **Session / Date:** `4b90e308-9a32-4b4e-be6f-1c7ab9e69c1c` | 2026-09-30
- **Severity:** 🟡 Medium
- **Symptoms:**
  When early stopping patience ran out (e.g. 5 rounds without metric improvement), the strategy flagged `should_stop = True`, but Flower server continued running all remaining rounds up to `num_rounds`.
- **Root Cause:**
  Native Flower `fl.server.Server.fit()` executes a fixed `range(1, num_rounds + 1)` loop and does not check strategy early-stop flags.
- **Resolution / Fix:**
  Subclassed `FedMedAIServer(fl.server.Server)` in [`server/server.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/server/server.py):
  - Intercepts `self.strategy.should_stop` before and during each round.
  - Defensively halts client sampling in `configure_fit` when `should_stop == True`.
  - Immediately breaks the round loop and writes `best_model.pth` and `final_model.pth`.
- **Verification:** Verified training halts immediately at the exact round early stopping triggers.

---

### ERR-007: FedNova Negative Variance / `NaN` Loss on `int64` Tracking Buffers
- **Affected Modules:** [`algorithms/fednova.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/algorithms/fednova.py), [`models/cnn.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/models/cnn.py)
- **Session / Date:** `a8b38674-d84c-4b1f-af26-8675947d1460` | 2026-10-03
- **Severity:** 🔴 Critical
- **Symptoms:**
  1. FedNova crashed with `numpy.core._exceptions.UFuncOutputCastingError` on `int64` BatchNorm counters.
  2. Under non-IID partitions, FedNova's global model produced `loss: NaN` during `model.eval()`.
- **Root Cause:**
  1. FedNova normalized step scaling ($d_i / \tau_i \times \tau_{\text{eff}}$) was applied to all state dict keys, including integer buffers like `num_batches_tracked`.
  2. Under non-IID client sample imbalances, scaling factors exceeded $8\times$. Negative variance shifts drove `running_var` negative. During evaluation in `model.eval()`, $\sqrt{\sigma^2 + \epsilon}$ produced `NaN`.
- **Resolution / Fix:**
  1. Added `get_buffer_mask()` and `get_var_buffer_mask()` in [`models/cnn.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/models/cnn.py).
  2. Decoupled trainable parameters from buffers in [`algorithms/fednova.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/algorithms/fednova.py):
     - Trainable weights use FedNova normalized step aggregation.
     - BatchNorm buffers use standard sample-weighted averaging with variance clipped $\ge 0$ (`np.maximum(var, 0.0)`).
- **Verification:** FedNova evaluated cleanly with valid loss curves across all non-IID partitions.

---

### ERR-008: SCAFFOLD Round 2 Collapse Under Adaptive Optimizers (Option 1 Flaw)
- **Affected Modules:** [`client/client.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/client/client.py), [`algorithms/scaffold.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/algorithms/scaffold.py)
- **Session / Date:** `a8b38674-d84c-4b1f-af26-8675947d1460` | 2026-10-03
- **Severity:** 🔴 Critical
- **Symptoms:**
  SCAFFOLD trained normally in Round 1 (62.44%), but collapsed in Round 2 to **8.30%** accuracy, followed by premature early stopping at Round 3.
- **Root Cause:**
  1. **Option 1 Flaw**: Weight displacement gradient approximation $\frac{x - y}{K \eta}$ assumes plain SGD. Under Adam, steps are normalized by second moments, causing Option 1 to compute coordinates $\approx \pm 1.0$ (norm 28.14, $15\times-20\times$ larger than true gradients).
  2. In Round 2, drift was added to `p.grad` every batch, completely drowning out true gradients.
  3. Non-trainable BatchNorm buffers were erroneously assigned non-zero control variates.
  4. Early stopping check was invoked 3 times per round, depleting patience prematurely.
- **Resolution / Fix:**
  1. Implemented **SCAFFOLD Option 2** in `client/client.py`: accumulated actual uncorrupted loss gradients on GPU during backprop before applying drift. Updated $c_i^{\text{new}} = \frac{1}{K} \sum g_i(y_k)$.
  2. Zeroed control variates for non-trainable buffers.
  3. Added `_last_checked_round` guard in `FedStrategyBase` ensuring early stopping decrements at most once per round.
- **Verification:** SCAFFOLD converged to **83.98%** accuracy at Round 5 with smooth trajectories.

---

### ERR-009: SCAFFOLD Velocity Amplification Under SGD Momentum (10x Drift)
- **Affected Module:** [`client/client.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/client/client.py)
- **Session / Date:** `3a3e7cf6-f32f-41f8-8a24-91a26d12882c` | 2026-10-21
- **Severity:** 🔴 Critical
- **Symptoms:**
  When running SCAFFOLD with standard SGD momentum ($\mu = 0.9$), loss exploded to $10^{23}$ and accuracy collapsed to 34.57%.
- **Root Cause:**
  Adding drift $(c - c_i)$ directly into `p.grad` at every mini-batch causes constant drift to accumulate in PyTorch's momentum buffer $v_t = \mu v_{t-1} + g_t$. The effective step size is amplified by $\frac{1}{1 - \mu} = \frac{1}{1 - 0.9} = \mathbf{10\times}$.
- **Resolution / Fix:**
  Adopted the decoupled post-step parameter drift update in [`client/client.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/client/client.py):
  ```python
  # 1. Standard optimizer step on true loss gradients
  optimizer.step()
  # 2. Decoupled parameter drift outside momentum buffer
  p.data.add_(drift, alpha=-step_lr)
  ```
- **Verification:** SCAFFOLD with SGD momentum 0.9 converged to **85.41%** MCTA without numerical instability.

---

### ERR-010: Silent FedBN Dormancy (`local_bn = 0`) on Baseline Architecture
- **Affected Modules:** [`models/baseline.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/models/baseline.py), [`algorithms/proposed.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/algorithms/proposed.py)
- **Session / Date:** `fcfdb657-1602-45bc-bef1-1f4c9aee3f5a` | 2026-10-18
- **Severity:** 🟠 High
- **Symptoms:**
  Server log showed `[Proposed] Round X | ... | local_bn=0`. FedBN exhibited identical convergence to FedAvg.
- **Root Cause:**
  When using `--model baseline` (the 4-layer plain CNN from the FedSLD paper), the architecture contains **zero BatchNorm layers**. Thus `sum(get_bn_mask(m)) == 0`, rendering FedBN completely inactive.
- **Resolution / Fix:**
  1. Created `BaselineBN` in [`models/baseline.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/models/baseline.py) featuring `BatchNorm2d` on convolutional layers.
  2. Documented that `--model cnn` or `--model baseline_bn` must be passed to activate FedBN's domain adaptation capabilities.
- **Verification:** Log confirmed `local_bn = 10` and verified local BN parameters are retained per client.

---

### ERR-011: Proposed Strategy Drift Conflict on Unbalanced Partitions
- **Affected Module:** [`algorithms/proposed.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/algorithms/proposed.py)
- **Session / Date:** `fcfdb657-1602-45bc-bef1-1f4c9aee3f5a` | 2026-10-18
- **Severity:** 🟠 High
- **Symptoms:**
  Proposed strategy reached **80.54%** on `dirichlet_0.3` but collapsed to **25.01%** on `practical` partition.
- **Root Cause:**
  1. In `practical`, step ratios varied by $25.5\times$. FedNova scaled small client gradients by $15.34\times$, injecting high variance.
  2. Unweighted server control variate update ($c_{\text{new}} = c + \frac{1}{N}\sum \Delta c_i$) clashed with sample-weighted aggregation ($\sum p_i \dots$). Because sample weights $p_i$ ranged from 0.005 to 0.35, the residual error $\eta \tau_{\text{eff}} (c - \sum p_i c_i) \neq 0$ injected artificial drift of magnitude ~3.067 at every round.
- **Resolution / Fix:**
  1. Updated server control variate update to use sample weights: $c_{\text{new}} = c + \sum p_i \Delta c_i$.
  2. Added scaling factor safety bounds on $\frac{\tau_{\text{eff}}}{\tau_i}$.
  3. Preserved BatchNorm parameters 100% local with zero SCAFFOLD drift.
- **Verification:** Proposed strategy achieved stable multi-round convergence across skewed partitions.

---

### ERR-012: Silent Architectural Mismatch via Hardcoded Model Fallback
- **Affected Modules:** [`algorithms/base.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/algorithms/base.py), `algorithms/fed*.py`
- **Session / Date:** `9ede8b12-55be-48be-9d83-3ebe653aaa66` | 2026-10-17
- **Severity:** 🟠 High
- **Symptoms:**
  When training on 9-class PathMNIST or 11-class OrganAMNIST, or when switching to `BaselineCNN`, FedNova and SCAFFOLD strategies silently loaded incorrect BatchNorm masks and control variate lengths.
- **Root Cause:**
  `FedNovaStrategy`, `FedBNStrategy`, and `SCAFFOLDStrategy` had a dangerous hardcoded fallback: `self.model = model or CNN(num_classes=8)`. If the caller forgot to pass `model`, masks were computed against an 8-class modern CNN regardless of the actual experiment setup.
- **Resolution / Fix:**
  Removed the fallback in [`algorithms/base.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/algorithms/base.py) and strategy constructors. Replaced with an explicit requirement that raises `ValueError("Strategy requires a valid model instance")` if `model is None`.
- **Verification:** Unit tests confirmed immediate exception raised on missing model, preventing silent parameter misalignment.

---

### ERR-013: Local Model Overfitting in Deep Federated Training (400 Epochs)
- **Affected Module:** [`datasets/medmnist_code.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/datasets/medmnist_code.py)
- **Session / Date:** `a84cabec-fe4a-4a03-9044-d3177b92471f` | 2026-10-25
- **Severity:** 🟠 High
- **Symptoms:**
  In baseline training without data augmentation, test loss decreased until Round 40 (~0.43) and then diverged to 0.57–0.62 at Round 80, capping test accuracy at 88.4%.
- **Root Cause:**
  80 federated rounds with 5 local epochs equals **400 local training epochs**. In the absence of data augmentation, local models memorized individual client samples, damaging generalization upon global aggregation.
- **Resolution / Fix:**
  Integrated `RandomCrop(28, padding=2, padding_mode='reflect')` and `RandomHorizontalFlip()` on training sets in [`datasets/medmnist_code.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/datasets/medmnist_code.py). Reflection padding was used to avoid artificial zero borders on medical slides. Evaluation sets remained strictly unaugmented.
- **Verification:** Test loss steadily decreased to 0.2016, driving accuracy from 88.44% to **95.04%**.

---

### ERR-014: Flower Server Distributed Evaluation Indefinite Hang
- **Affected Module:** [`server/server.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/server/server.py)
- **Session / Date:** `7cd544f5-bdc6-4ee7-9584-ea5828737ce4` | 2026-10-22
- **Severity:** 🟠 High
- **Symptoms:**
  At Round 27, Clients 0 through 11 finished training, but Flower server hung indefinitely during distributed evaluation without error messages.
- **Root Cause:**
  In `server/server.py`, `evaluate_round(..., timeout=timeout)` was called with `timeout=None`. If any client worker experienced a stall during evaluation message handling, the Flower server waited forever.
- **Resolution / Fix:**
  Added timeout enforcement in [`server/server.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/server/server.py):
  ```python
  eval_timeout = timeout if timeout is not None else 60.0
  res = client_manager.sample(..., timeout=eval_timeout)
  ```
- **Verification:** Stalled evaluation calls time out gracefully, allowing FL rounds to proceed.

---

### ERR-015: Evaluation Metric Ambiguity (Macro-Average vs. Micro-Average)
- **Affected Modules:** [`algorithms/base.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/algorithms/base.py), [`monitoring/metrics.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/monitoring/metrics.py)
- **Session / Date:** `bf17e176-6538-45c6-972b-097242ed63f4` | 2026-10-22
- **Severity:** 🟡 Medium
- **Symptoms:**
  Confusion in metric reporting over whether `BTA` evaluated a single client or pooled test data, and how it differed from `BMCTA`.
- **Root Cause:**
  Ambiguous naming conventions across federated medical literature (e.g. FedSLD paper).
- **Resolution / Fix:**
  Formalized mathematical definitions and standardized tracking:
  1. **BMCTA (Best Mean Client Test Accuracy - Macro Average):**
     $$\text{MCTA} = \frac{1}{K} \sum_{i=1}^K \text{Acc}_i$$
     Evaluates fairness across healthcare institutions so small clinics are weighted equally to large hospitals.
  2. **BTA (Best Total Accuracy - Micro Average):**
     $$\text{Combined Acc} = \sum_{i=1}^K \frac{N_i^{\text{test}}}{N_{\text{total}}^{\text{test}}} \text{Acc}_i = \frac{\sum \text{Correct}_i}{\sum N_i^{\text{test}}}$$
     Evaluates total diagnostic utility across the pooled patient population.
- **Verification:** Metric definitions clarified in CSV exports, plot labels, and documentation.

---

## Guidelines for Logging Future Errors

When a new bug, numerical error, or algorithmic defect is discovered in future sessions, append a new entry to this document following this template:

```markdown
### ERR-XXX: <Brief Descriptive Title>
- **Affected Module:** `<file_path>`
- **Session / Date:** `<session-id>` | YYYY-MM-DD
- **Severity:** 🔴 Critical / 🟠 High / 🟡 Medium / 🔵 Low
- **Symptoms:**
  <What happened? Include exact terminal error messages, tracebacks, or anomalous metric trajectories.>
- **Root Cause:**
  <Why did it happen? Explain the mathematical, architectural, or codebase mechanism.>
- **Resolution / Fix:**
  <How was it solved? Include code snippets, configuration changes, or algorithm patches.>
- **Verification:**
  <How was the fix proven to work? Mention unit tests, benchmark runs, or log proofs.>
```
