# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 10:38:06  
**Run Name:** `bloodmnist_dirichlet_0.3_fedprox_baseline_5` | **Model:** `baseline` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDPROX** achieved **91.82%** accuracy in Round 79 (runtime: 167.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDPROX** | `dirichlet_0.3` | 91.87% | **91.82%** | **91.87%** (R79) | 0.2337 | **91.82%** (R79) | R79 | 0.2337 | 167.2s |