# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 12:00:10  
**Run Name:** `bloodmnist_dirichlet_0.1_run_3_seed_44_baseline_bn_scaffold` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] SCAFFOLD** achieved **87.66%** accuracy in Round 73 (runtime: 131.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | `sgd` | No | **SCAFFOLD** | `dirichlet_0.1` | 82.99% | **87.66%** | **82.25%** (R73) | **87.66%** (R73) | R73 | 0.4790 | 131.1s |