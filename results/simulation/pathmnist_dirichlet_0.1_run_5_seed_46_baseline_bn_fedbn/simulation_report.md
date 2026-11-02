# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 14:22:25  
**Run Name:** `pathmnist_dirichlet_0.1_run_5_seed_46_baseline_bn_fedbn` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDBN** achieved **88.38%** accuracy in Round 79 (runtime: 609.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | `sgd` | No | **FEDBN** | `dirichlet_0.1` | 68.84% | **88.38%** | **84.26%** (R79) | **88.38%** (R79) | R79 | 2.0337 | 609.6s |