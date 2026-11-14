# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 10:25:55  
**Run Name:** `bloodmnist_dirichlet_0.3_fedprox_baseline_1` | **Model:** `baseline` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDPROX** achieved **91.03%** accuracy in Round 78 (runtime: 156.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDPROX** | `dirichlet_0.3` | 91.46% | **91.03%** | **91.46%** (R78) | 0.2350 | **91.03%** (R78) | R78 | 0.2350 | 156.4s |