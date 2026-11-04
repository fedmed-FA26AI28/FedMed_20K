# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 12:04:14  
**Run Name:** `pathmnist_dirichlet_0.3_scaffold_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] SCAFFOLD** achieved **81.56%** accuracy in Round 61 (runtime: 536.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **SCAFFOLD** | `dirichlet_0.3` | 73.22% | **81.56%** | **82.39%** (R64) | 1.0840 | **81.56%** (R61) | R61 | 1.5746 | 536.8s |