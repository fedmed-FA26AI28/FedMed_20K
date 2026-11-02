# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 11:34:51  
**Run Name:** `organamnist_dirichlet_0.3_fedprox_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDPROX** achieved **87.69%** accuracy in Round 68 (runtime: 280.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDPROX** | `dirichlet_0.3` | 87.66% | **87.69%** | **87.55%** (R71) | 0.5081 | **87.69%** (R68) | R68 | 0.5033 | 280.9s |