# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-28 21:42:26  
**Run Name:** `organamnist_fedavg_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] FEDAVG** achieved **86.11%** accuracy in Round 74 (runtime: 261.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDAVG** | `practical` | 85.99% | **86.11%** | **86.44%** (R74) | 0.4628 | **86.11%** (R74) | R74 | 0.4843 | 261.3s |