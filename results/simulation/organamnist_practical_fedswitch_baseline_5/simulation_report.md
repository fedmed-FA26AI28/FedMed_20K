# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 18:49:55  
**Run Name:** `organamnist_practical_fedswitch_baseline_5` | **Model:** `baseline` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] FEDSWITCH** achieved **81.47%** accuracy in Round 78 (runtime: 280.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDSWITCH** | `practical` | 81.42% | **81.47%** | **81.68%** (R77) | 0.8401 | **81.47%** (R78) | R78 | 0.8707 | 280.9s |