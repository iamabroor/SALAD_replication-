# Results Table — MVTec LOCO AD Replication

## Main Results (Seed 42)

| Category | Logical AUROC | Structural AUROC | Mean AUROC | Paper Mean | Gap |
|---|---|---|---|---|---|
| breakfast_box | — | — | — | 89.3 | — |
| juice_bottle | — | — | — | 99.7 | — |
| pushpins | — | — | — | 99.4 | — |
| screw_bag | — | — | — | 95.0 | — |
| splicing_connectors | — | — | — | 97.3 | — |
| **AVERAGE (5 cats)** | **—** | **—** | **93.80** | **96.1** | **−2.30** |

> **Note:** Per-category logical/structural splits to be filled after full training log analysis.  
> Overall mean AUROC from seed 42 training run: **93.80** vs paper **96.1**.

---

## Seed Study Results

| Category | Seed | Logical | Structural | Mean |
|---|---|---|---|---|
| breakfast_box | 42 | — | — | — |
| breakfast_box | 0 | — | — | — |
| breakfast_box | 1 | — | — | — |

*(Run `python 03_scripts/final_results.py` after training to populate this table automatically)*

---

## Comparison with Paper (Table 1 from SALAD paper)

| Method | breakfast_box | juice_bottle | pushpins | screw_bag | splicing_connectors | Mean |
|---|---|---|---|---|---|---|
| **SALAD (paper)** | **89.3** | **99.7** | **99.4** | **95.0** | **97.3** | **96.1** |
| **This replication** | — | — | — | — | — | **93.80** |
| EfficientAD (baseline) | 86.1 | 97.4 | 98.2 | 91.0 | 94.7 | 93.5 |

---

## How to Reproduce

```bash
# After training all categories:
python 03_scripts/final_results.py
# Output: ~/salad/output/final_report/final_results_with_seeds.csv
#         ~/salad/output/final_report/final_results_with_seeds.png
```

The CSV and PNG chart are the primary result artefacts.
