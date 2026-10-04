# Scripts

Run in this order:

| # | Script | Purpose |
|---|---|---|
| 1 | `run_all.sh` | Train all 5 categories sequentially |
| 2 | `collect_run_info.sh` | Record hardware/software/timing info |
| 3 | `final_results.py` | Parse logs → AUROC table + bar chart |
| 4 | `visualize_salad.py` | Generate qualitative anomaly maps |
| 5 | `pipeline_walkthrough.py` | Generate 15-panel pipeline figures |
| 6 | `make_report.py` | Build full HTML report |
| 7 | `final_report.sh` | Package all figures into report |

## Usage

```bash
# Full pipeline
bash 03_scripts/run_all.sh 2>&1 | tee output/logs/master.log

# Results only (after training)
python 03_scripts/final_results.py
python 03_scripts/visualize_salad.py --results_dir output/results
python 03_scripts/pipeline_walkthrough.py --results_dir output/results
python 03_scripts/make_report.py
bash 03_scripts/final_report.sh
```
