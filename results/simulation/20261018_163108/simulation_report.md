# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-18 16:48:41  
**Model:** `baseline` | **Seed:** `42` | **Rounds:** 20 | **Clients:** 5 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Disabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** 🏆 **PROPOSED** achieved **80.54%** accuracy in Round 20 (runtime: 1031.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | `sgd` | ✅ | **PROPOSED** | `dirichlet_0.3` | 80.54% | **80.54%** | **80.54%** (R20) | **80.54%** (R20) | R20 | 0.6538 | 1031.8s |