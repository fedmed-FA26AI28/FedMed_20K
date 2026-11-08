# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 11:01:18  
**Run Name:** `organamnist_dirichlet_0.1_run_5_seed_46_baseline_bn_fedprox` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDPROX** achieved **84.82%** accuracy in Round 80 (runtime: 459.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | No | **FEDPROX** | `dirichlet_0.1` | 84.82% | **84.82%** | **85.54%** (R78) | **84.82%** (R80) | R80 | 0.5745 | 459.4s |