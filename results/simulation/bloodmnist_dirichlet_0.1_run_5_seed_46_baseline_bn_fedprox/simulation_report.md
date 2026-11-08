# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 12:17:23  
**Run Name:** `bloodmnist_dirichlet_0.1_run_5_seed_46_baseline_bn_fedprox` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDPROX** achieved **89.59%** accuracy in Round 77 (runtime: 140.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | `sgd` | No | **FEDPROX** | `dirichlet_0.1` | 89.27% | **89.59%** | **86.89%** (R78) | **89.59%** (R77) | R77 | 0.2864 | 140.8s |