# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-24 17:11:59  
**Run Name:** `fixed_organa_proposed` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] PROPOSED** achieved **94.27%** accuracy in Round 80 (runtime: 1980.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | Yes | **PROPOSED** | `practical` | 94.27% | **94.27%** | **91.95%** (R80) | **94.27%** (R80) | R80 | 0.2016 | 1980.9s |