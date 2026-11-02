# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 20:54:43  
**Run Name:** `pathmnist_dirichlet_0.3_fedbn_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDBN** achieved **78.20%** accuracy in Round 61 (runtime: 513.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDBN** | `dirichlet_0.3` | 54.85% | **78.20%** | **77.45%** (R61) | 1.6483 | **78.20%** (R61) | R61 | 3.6020 | 513.5s |