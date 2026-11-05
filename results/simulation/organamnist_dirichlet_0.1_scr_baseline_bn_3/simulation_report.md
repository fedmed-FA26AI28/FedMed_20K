# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 12:53:19  
**Run Name:** `dirichlet01_bn_organamnist_s44` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] SCR** achieved **85.18%** accuracy in Round 80 (runtime: 1854.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 84.88% | **84.96%** | **84.89%** (R78) | 0.4875 | **84.96%** (R75) | R75 | 0.4862 | 1844.4s |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **SCR** | `dirichlet_0.1` | 85.18% | **85.18%** | **85.46%** (R80) | 0.5331 | **85.18%** (R80) | R80 | 0.5391 | 1854.6s |