#!/usr/bin/env bash
# Run once after ALL training has finished:  bash ~/final_report.sh
# Produces ~/salad/output/final_report/ (tables, charts, report.html) + all figures + ~/salad_final_report.tar.gz
set -uo pipefail
PY="$HOME/salad/venv/bin/python"
OUT="$HOME/salad/output"

if pgrep -f train_salad.py >/dev/null; then
  echo "Training is still running – wait until it has finished (check with: pgrep -af train_salad.py)."; exit 1
fi

echo "== 1/5 Hardware / software / timing record"
bash "$HOME/collect_run_info.sh" > /dev/null && echo "saved $OUT/run_record.txt"

echo "== 2/5 Figures for every category (best + random) – about 30 min"
for c in breakfast_box juice_bottle pushpins screw_bag splicing_connectors; do
  [ -f "$OUT/results/$c/comp_unet_final.pth" ] || { echo "skip $c (not trained)"; continue; }
  [ -f "$OUT/figures/${c}_slides.png" ] || "$PY" "$HOME/visualize_salad.py" --category "$c" --select best
  [ -f "$OUT/figures/random/${c}_slides.png" ] || "$PY" "$HOME/visualize_salad.py" --category "$c" --select random --out "$OUT/figures/random"
done
echo "== 3/5 Overview figure across all categories"
[ -f "$OUT/figures/all/all_categories_slides.png" ] || "$PY" "$HOME/visualize_salad.py" --category all --select best --n 2 --out "$OUT/figures/all"

echo "== 4/5 Tables, charts and HTML report"
"$PY" "$HOME/make_report.py" --out "$OUT/final_report"

echo "== 5/5 Package for download"
cd "$HOME/salad"
tar czf "$HOME/salad_final_report.tar.gz" output/final_report output/figures output/logs output/run_record.txt \
    output/results_table_train.csv 2>/dev/null
ls -lh "$HOME/salad_final_report.tar.gz"
echo
echo "DONE. Open the report:  xdg-open $OUT/final_report/report.html"
echo "Download:               DCV ⇅  ->  salad_final_report.tar.gz"
echo "Then STOP the workspace in RACE."
