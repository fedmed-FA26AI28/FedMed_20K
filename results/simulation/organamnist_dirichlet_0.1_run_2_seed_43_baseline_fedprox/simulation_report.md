# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 08:12:21  
**Run Name:** `organamnist_dirichlet_0.1_run_2_seed_43_baseline_fedprox` | **Model:** `baseline` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDPROX** achieved **82.27%** accuracy in Round 80 (runtime: 381.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | `sgd` | No | **FEDPROX** | `dirichlet_0.1` | 82.27% | **82.27%** | **82.34%** (R78) | **82.27%** (R80) | R80 | 0.6639 | 381.6s |