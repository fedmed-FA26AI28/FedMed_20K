# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-19 10:38:35  
**Run Name:** `test_opt_run` | **Model:** `cnn` | **Seed:** `42` | **Rounds:** 1 | **Clients:** 2 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Disabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** 🏆 **FEDAVG** achieved **40.46%** accuracy in Round 1 (runtime: 26.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `cnn` | `sgd` | ✅ | **FEDAVG** | `dirichlet_0.3` | 40.46% | **40.46%** | **40.46%** (R1) | **40.46%** (R1) | R1 | 1.5201 | 26.6s |