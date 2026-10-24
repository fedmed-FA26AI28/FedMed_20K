# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-24 09:27:27  
**Run Name:** `organa_avg_prox` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `server_decay` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] FEDPROX** achieved **88.00%** accuracy in Round 59 (runtime: 757.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | Yes | **FEDAVG** | `practical` | 87.77% | **87.91%** | **86.66%** (R77) | **87.91%** (R59) | R59 | 0.3580 | 739.1s |
| **organamnist** | `baseline_bn` | `sgd` | Yes | **FEDPROX** | `practical` | 87.79% | **88.00%** | **86.67%** (R72) | **88.00%** (R59) | R59 | 0.3586 | 757.8s |