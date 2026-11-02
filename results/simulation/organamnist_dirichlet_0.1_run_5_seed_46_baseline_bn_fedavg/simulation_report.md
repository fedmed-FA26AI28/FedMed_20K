# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 10:53:26  
**Run Name:** `organamnist_dirichlet_0.1_run_5_seed_46_baseline_bn_fedavg` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **84.72%** accuracy in Round 78 (runtime: 443.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 84.60% | **84.72%** | **85.45%** (R78) | **84.72%** (R78) | R78 | 0.5780 | 443.4s |