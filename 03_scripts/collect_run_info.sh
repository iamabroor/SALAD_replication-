#!/usr/bin/env bash
# Collects hardware / software / timing / results for the SALAD replication report.
# Usage:  bash ~/collect_run_info.sh        -> writes ~/salad/output/run_record.txt (safe to run any time)
OUT="$HOME/salad/output/run_record.txt"
PY="$HOME/salad/venv/bin/python"
REPO="$HOME/salad/SALAD"
LOGS="$HOME/salad/output/logs"
{
echo "=== SALAD replication – run record (generated $(date '+%F %T %Z')) ==="
echo
echo "--- Machine ---"
echo "Instance type : $(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo 'g5.2xlarge (RACE G07)')"
echo "Region        : $(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null || echo 'us-east-1')"
echo "OS            : $(. /etc/os-release; echo "$PRETTY_NAME") / kernel $(uname -r)"
echo "CPU           : $(lscpu | awk -F: '/Model name/{gsub(/^ +/,"",$2);print $2}') – $(nproc) vCPUs"
echo "RAM           : $(free -g | awk '/Mem:/{print $2" GB"}')"
echo "Disk (/)      : $(df -h / | awk 'NR==2{print $3" used of "$2}')"
echo "Timezone      : $(date +%Z) (UTC$(date +%z))"
echo
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
nvidia-smi | grep -m1 "CUDA Version"
echo
echo "--- Software ---"
"$PY" - <<'EOF'
import sys, torch, torchvision, numpy, sklearn, timm
print("Python      :", sys.version.split()[0])
print("PyTorch     :", torch.__version__, "| CUDA runtime", torch.version.cuda, "| cuDNN", torch.backends.cudnn.version())
print("torchvision :", torchvision.__version__, "| numpy", numpy.__version__, "| scikit-learn", sklearn.__version__, "| timm", timm.__version__)
EOF
echo "SALAD commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null)"
echo
echo "--- Training time per category (from tqdm in the logs) ---"
for f in "$LOGS"/*_train.log; do
  [ -f "$f" ] || continue
  c=$(basename "$f" _train.log)
  last=$(tr '\r' '\n' < "$f" | grep -E "70000/70000|[0-9]+/70000 \[" | tail -1 | grep -oE "[0-9]+/70000 \[[0-9:]+<[0-9:?]+, *[0-9.]+it/s" )
  res=$(grep "All Logical" "$f" | tail -1)
  echo "$c : ${last:-running/not started} | final: ${res:-n/a}"
done
echo
echo "--- Intermediate AUROC every 10k steps ---"
for f in "$LOGS"/*_train.log; do
  [ -f "$f" ] || continue
  echo "[$(basename "$f" _train.log)]"
  grep -E "Intermediate image auc|Final image auc" "$f" | nl
done
echo
echo "--- Authors' released weights (eval) ---"
for f in "$LOGS"/*_eval.log; do [ -f "$f" ] && echo "$(basename "$f" _eval.log): $(grep 'All Logical' "$f" | tail -1)"; done
echo
echo "--- Results table ---"
cat "$HOME/salad/output/results_table_train.csv" 2>/dev/null || echo "(not yet)"
echo
echo "--- Disk usage ---"
du -sh "$HOME/salad/imagenet" "$REPO/data" "$HOME/salad/venv" "$HOME/salad/output" 2>/dev/null
} | tee "$OUT"
echo; echo "Saved to $OUT"
