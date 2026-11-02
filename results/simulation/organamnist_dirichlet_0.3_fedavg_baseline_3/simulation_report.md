# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 08:42:57  
**Run Name:** `organamnist_dirichlet_0.3_fedavg_baseline_3` | **Model:** `baseline` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDAVG** achieved **84.91%** accuracy in Round 71 (runtime: 240.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.3` | 84.68% | **84.91%** | **85.05%** (R78) | 0.7805 | **84.91%** (R71) | R71 | 0.8303 | 240.3s |