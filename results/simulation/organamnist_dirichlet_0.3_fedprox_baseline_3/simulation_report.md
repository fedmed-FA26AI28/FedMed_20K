# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 09:05:00  
**Run Name:** `organamnist_dirichlet_0.3_fedprox_baseline_3` | **Model:** `baseline` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDPROX** achieved **84.90%** accuracy in Round 71 (runtime: 255.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDPROX** | `dirichlet_0.3` | 84.66% | **84.90%** | **85.09%** (R71) | 0.7679 | **84.90%** (R71) | R71 | 0.8224 | 255.2s |