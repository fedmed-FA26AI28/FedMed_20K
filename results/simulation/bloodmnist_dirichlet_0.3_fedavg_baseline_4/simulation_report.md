# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 10:20:03  
**Run Name:** `bloodmnist_dirichlet_0.3_fedavg_baseline_4` | **Model:** `baseline` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDAVG** achieved **90.85%** accuracy in Round 78 (runtime: 159.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.3` | 90.84% | **90.85%** | **90.84%** (R78) | 0.2965 | **90.85%** (R78) | R78 | 0.2965 | 159.9s |