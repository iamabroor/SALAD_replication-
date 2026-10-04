# SALAD Replication — MVTec LOCO AD

**Paper:** SALAD: Semantic Aware Discriminative Logical Anomaly Detection (ICCV 2025)  
**arXiv:** [2509.02101](https://arxiv.org/abs/2509.02101)  
**Author:** Ibbi (iamabroor@gmail.com)  
**Date:** October 2026

---

## Overview

This repository documents a full replication of the SALAD anomaly detection method on the MVTec LOCO AD benchmark. SALAD uses three complementary branches — Appearance, Composition, and Global Statistical — fused to detect both structural and logical anomalies.

## Results Summary

| Category | Logical | Structural | Mean | Paper Mean | Gap |
|---|---|---|---|---|---|
| breakfast_box | 89.72 | 82.46 | 86.09 | 89.3 | -3.21 |
| juice_bottle | 99.64 | 98.96 | 99.3 | 99.7 | -0.4 |
| pushpins | 99.92 | 97.85 | 98.89 | 99.4 | -0.51 |
| screw_bag | 90.15 | 95.69 | 92.92 | 95.0 | -2.08 |
| splicing_connectors | 94.9 | 97.45 | 96.18 | 97.3 | -1.12 |
| **AVERAGE** | **94.87** | **94.48** | **94.68** | **96.1** | **-1.42** |

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
# 1. Set up environment (AWS GPU A10 g5.2xlarge used)
bash 03_scripts/run_all.sh

# 2. After training, generate report
bash 03_scripts/final_report.sh

# 3. View final results
python 03_scripts/final_results.py
```
## Visualisations & Graphs

### Performance Graphs
<p align="center">
  <img src="05_results/figures/01_mean_vs_paper (1).png" width="48%">
  <img src="05_results/figures/02_logical_structural (1).png" width="48%">
  <img src="05_results/figures/03_branches_vs_paper (1).png" width="48%">
  <img src="05_results/figures/all_categories_slides (1).png" width="48%">
</p>

### Summary Steps
<p align="center">
  <img src="05_results/figures/16_summary_all_steps.png" width="48%">
  <img src="05_results/figures/16_summary_all_steps (1).png" width="48%">
  <img src="05_results/figures/16_summary_all_steps (2).png" width="48%">
  <img src="05_results/figures/16_summary_all_steps (3).png" width="48%">
  <img src="05_results/figures/16_summary_all_steps (4).png" width="48%">
  <img src="05_results/figures/16_summary_all_steps (5).png" width="48%">
</p>

### Category Galleries
<p align="center">
  <img src="05_results/figures/breakfast_box_gallery_p2.png" width="48%">
  <img src="05_results/figures/juice_bottle_gallery_p1.png" width="48%">
  <img src="05_results/figures/pushpins_gallery_p3.png" width="48%">
  <img src="05_results/figures/screw_bag_gallery_p3.png" width="48%">
  <img src="05_results/figures/splicing_connectors_gallery_p1.png" width="48%">
</p>

## Citation

```bibtex
@inproceedings{salad2025,
  title     = {SALAD: Semantic Aware Discriminative Logical Anomaly Detection},
  booktitle = {ICCV},
  year      = {2025},
  note      = {arXiv:2509.02101}
}
```
