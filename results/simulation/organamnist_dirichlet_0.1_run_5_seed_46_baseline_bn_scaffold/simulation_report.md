# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 11:15:07  
**Run Name:** `organamnist_dirichlet_0.1_run_5_seed_46_baseline_bn_scaffold` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] SCAFFOLD** achieved **85.02%** accuracy in Round 78 (runtime: 452.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | No | **SCAFFOLD** | `dirichlet_0.1` | 84.88% | **85.02%** | **85.93%** (R78) | **85.02%** (R78) | R78 | 0.5959 | 452.3s |