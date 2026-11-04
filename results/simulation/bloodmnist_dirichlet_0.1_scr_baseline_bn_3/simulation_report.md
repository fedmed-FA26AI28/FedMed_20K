# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 08:13:32  
**Run Name:** `dirichlet01_bn_bloodmnist_s44` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] SCR** achieved **89.18%** accuracy in Round 78 (runtime: 616.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 86.32% | **87.93%** | **89.44%** (R78) | 0.3021 | **87.93%** (R64) | R64 | 0.3751 | 624.5s |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **SCR** | `dirichlet_0.1` | 86.17% | **89.18%** | **91.21%** (R79) | 0.2956 | **89.18%** (R78) | R78 | 0.4232 | 616.6s |