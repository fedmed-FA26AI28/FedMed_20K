# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-21 15:27:46  
**Run Name:** `path_practical_s41_recheck` | **Model:** `baseline` | **Seed:** `41` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** 🏆 **FEDAVG** achieved **26.57%** accuracy in Round 1 (runtime: 506.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Strategy | Partition | Final Acc (%) | Best Acc (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | `sgd` | **FEDAVG** | `practical` | 26.57% | **26.57%** | R1 | 2.0911 | 506.1s |