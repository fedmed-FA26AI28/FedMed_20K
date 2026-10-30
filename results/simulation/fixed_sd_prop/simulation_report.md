# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-24 12:52:00  
**Run Name:** `fixed_sd_prop` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] PROPOSED** achieved **81.74%** accuracy in Round 73 (runtime: 1717.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | `sgd` | Yes | **PROPOSED** | `practical` | 72.26% | **81.74%** | **79.64%** (R78) | **81.74%** (R73) | R73 | 1.3837 | 1717.9s |