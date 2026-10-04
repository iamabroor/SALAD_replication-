# Figures

## Generated Figures

After running `bash 03_scripts/final_report.sh`, the following figures are generated:

### Qualitative Results (per category)
Saved to `~/salad/output/figures/`

| File | Description |
|---|---|
| `breakfast_box_slides.png` | Best test image — 10-panel SALAD pipeline visualisation |
| `juice_bottle_slides.png` | Best test image — 10-panel SALAD pipeline visualisation |
| `pushpins_slides.png` | Best test image — 10-panel SALAD pipeline visualisation |
| `screw_bag_slides.png` | Best test image — 10-panel SALAD pipeline visualisation |
| `splicing_connectors_slides.png` | Best test image — 10-panel SALAD pipeline visualisation |

### Quantitative Charts
Saved to `~/salad/output/final_report/`

| File | Description |
|---|---|
| `final_results_with_seeds.png` | Bar chart: this replication vs paper per category |
| `training_curves.png` | AUROC vs training steps for all categories |
| `branch_auroc.png` | Per-branch AUROC breakdown |

### How to Generate

```bash
# Qualitative visualisations (best anomaly per category)
python 03_scripts/visualize_salad.py --category all --select best

# Pipeline walkthrough (full 10-panel for one image)
python 03_scripts/pipeline_walkthrough.py --category breakfast_box

# Final quantitative chart
python 03_scripts/final_results.py
```

## Note on Diagram

The handwritten architecture diagram by the author is in `01_methodology/methodology_diagram_handwritten.png`.
