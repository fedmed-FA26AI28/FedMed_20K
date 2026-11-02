# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 13:23:59  
**Run Name:** `pathmnist_dirichlet_0.3_fedprox_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDPROX** achieved **81.98%** accuracy in Round 74 (runtime: 542.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDPROX** | `dirichlet_0.3` | 62.14% | **81.98%** | **81.79%** (R76) | 1.0414 | **81.98%** (R74) | R74 | 1.6928 | 542.7s |