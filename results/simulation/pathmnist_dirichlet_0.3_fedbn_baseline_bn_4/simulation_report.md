# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 20:45:55  
**Run Name:** `pathmnist_dirichlet_0.3_fedbn_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDBN** achieved **81.85%** accuracy in Round 70 (runtime: 498.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDBN** | `dirichlet_0.3` | 81.71% | **81.85%** | **79.06%** (R80) | 1.2031 | **81.85%** (R70) | R70 | 1.0247 | 498.6s |