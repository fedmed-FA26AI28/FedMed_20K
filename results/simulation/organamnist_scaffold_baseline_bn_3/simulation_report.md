# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-30 11:06:50  
**Run Name:** `organamnist_scaffold_baseline_bn_3` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] SCAFFOLD** achieved **86.26%** accuracy in Round 68 (runtime: 324.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **SCAFFOLD** | `practical` | 85.91% | **86.26%** | **86.85%** (R68) | 0.4559 | **86.26%** (R68) | R68 | 0.5013 | 324.8s |