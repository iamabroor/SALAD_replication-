# SALAD Replication — MVTec LOCO AD

**Paper:** SALAD: Semantic Aware Discriminative Logical Anomaly Detection (ICCV 2025)  
**arXiv:** [2509.02101](https://arxiv.org/abs/2509.02101)  
**Author:** Ibbi (mabb0019@student.monash.edu)  
**Date:** October 2026

---

## Overview

This repository documents a full replication of the SALAD anomaly detection method on the MVTec LOCO AD benchmark. SALAD uses three complementary branches — Appearance, Composition, and Global Statistical — fused to detect both structural and logical anomalies.

## Results Summary

| Category | Logical | Structural | Mean | Paper Mean | Gap |
|---|---|---|---|---|---|
| breakfast_box | — | — | — | 89.3 | — |
| juice_bottle | — | — | — | 99.7 | — |
| pushpins | — | — | — | 99.4 | — |
| screw_bag | — | — | — | 95.0 | — |
| splicing_connectors | — | — | — | 97.3 | — |
| **AVERAGE** | — | — | **93.80** | **96.1** | **−2.3** |

*(Seed 42; see `05_results/metrics/results_table.md` for full breakdown)*

## Repository Structure

```
SALAD_replication-/
├── 01_methodology/         Architecture diagrams, method overview
├── 02_src/source_code/     Original SALAD source code (git submodule / notes)
├── 03_scripts/             Replication scripts (training, evaluation, visualisation)
├── 04_configs/             Training hyperparameters and settings
├── 05_results/             Quantitative metrics, tables, figures
├── 06_environment/         Environment setup, dependencies
└── 07_replication_notes/   Implementation notes, issues & solutions
```

## Quick Start

```bash
# 1. Set up environment (RACE g5.2xlarge recommended)
bash 03_scripts/run_all.sh

# 2. After training, generate report
bash 03_scripts/final_report.sh

# 3. View final results
python 03_scripts/final_results.py
```

## Citation

```bibtex
@inproceedings{salad2025,
  title     = {SALAD: Semantic Aware Discriminative Logical Anomaly Detection},
  booktitle = {ICCV},
  year      = {2025},
  note      = {arXiv:2509.02101}
}
```
