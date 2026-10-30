# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-23 13:30:09  
**Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `server_decay` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] PROPOSED** achieved **88.12%** accuracy in Round 77 (runtime: 5071.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | `sgd` | Yes | **PROPOSED** | `practical` | 86.56% | **88.12%** | **87.12%** (R77) | **88.12%** (R77) | R77 | 0.5025 | 5071.1s |