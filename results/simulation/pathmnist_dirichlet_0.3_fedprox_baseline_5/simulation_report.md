# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 10:44:43  
**Run Name:** `pathmnist_dirichlet_0.3_fedprox_baseline_5` | **Model:** `baseline` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDPROX** achieved **78.47%** accuracy in Round 50 (runtime: 497.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDPROX** | `dirichlet_0.3` | 73.04% | **78.47%** | **78.71%** (R50) | 1.0896 | **78.47%** (R50) | R50 | 1.4615 | 497.9s |