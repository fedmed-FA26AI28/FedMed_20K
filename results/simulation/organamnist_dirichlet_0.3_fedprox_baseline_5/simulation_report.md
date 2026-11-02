# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 09:14:10  
**Run Name:** `organamnist_dirichlet_0.3_fedprox_baseline_5` | **Model:** `baseline` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDPROX** achieved **83.50%** accuracy in Round 75 (runtime: 274.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDPROX** | `dirichlet_0.3` | 83.47% | **83.50%** | **82.89%** (R71) | 0.8383 | **83.50%** (R75) | R75 | 0.8337 | 274.7s |