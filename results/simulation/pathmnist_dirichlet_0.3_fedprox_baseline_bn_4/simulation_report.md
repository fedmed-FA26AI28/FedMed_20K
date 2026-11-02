# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 13:14:43  
**Run Name:** `pathmnist_dirichlet_0.3_fedprox_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDPROX** achieved **84.30%** accuracy in Round 39 (runtime: 523.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDPROX** | `dirichlet_0.3` | 82.88% | **84.30%** | **81.20%** (R39) | 0.8052 | **84.30%** (R39) | R39 | 0.8227 | 523.4s |