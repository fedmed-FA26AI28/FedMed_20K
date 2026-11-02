# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 12:02:17  
**Run Name:** `bloodmnist_dirichlet_0.1_run_3_seed_44_baseline_bn_proposed` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] PROPOSED** achieved **96.64%** accuracy in Round 80 (runtime: 114.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | `sgd` | No | **PROPOSED** | `dirichlet_0.1` | 96.64% | **96.64%** | **87.79%** (R80) | **96.64%** (R80) | R80 | 0.1144 | 114.7s |