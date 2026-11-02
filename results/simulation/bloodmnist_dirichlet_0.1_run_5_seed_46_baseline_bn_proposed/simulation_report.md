# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 12:24:34  
**Run Name:** `bloodmnist_dirichlet_0.1_run_5_seed_46_baseline_bn_proposed` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] PROPOSED** achieved **97.69%** accuracy in Round 78 (runtime: 133.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | `sgd` | No | **PROPOSED** | `dirichlet_0.1` | 97.54% | **97.69%** | **96.04%** (R79) | **97.69%** (R78) | R78 | 0.0931 | 133.8s |