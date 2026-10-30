# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-15 12:51:47  
**Model:** `baseline` | **Seed:** `41` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** 🏆 **FEDAVG** achieved **74.94%** accuracy in Round 69 (runtime: 3783.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | `sgd` | ✅ | **FEDAVG** | `practical` | 73.15% | **74.94%** | **68.57%** (R76) | **74.94%** (R69) | R69 | 0.9247 | 3783.3s |