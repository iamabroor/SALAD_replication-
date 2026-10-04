#!/usr/bin/env bash
# =============================================================================
# SALAD (ICCV 2025) – MVTec LOCO replication on a single-GPU Ubuntu machine
# (RMIT RACE Workbench / AWS g5.2xlarge). Safe to re-run: finished steps are skipped.
#
#   Usage (inside tmux so it survives disconnecting):
#     export HF_TOKEN=hf_xxx            # Hugging Face read token (ImageNet download)
#     MODE=eval  bash run_all.sh        # quick check: evaluate the authors' released weights
#     MODE=train bash run_all.sh        # full replication: 70,000 steps x 5 categories
# =============================================================================
set -euo pipefail

MODE="${MODE:-train}"                                   # train | eval
CATEGORIES="${CATEGORIES:-breakfast_box juice_bottle pushpins screw_bag splicing_connectors}"
TRAIN_STEPS="${TRAIN_STEPS:-70000}"                     # paper: 70,000 (LR drop at 0.95*70k = 66,500)
SEED="${SEED:-42}"
export WORK="${WORK:-$HOME/salad}"
REPO="$WORK/SALAD"
VENV="$WORK/venv"
PY="$VENV/bin/python"
IMAGENET="$WORK/imagenet/train"
OUT="$WORK/output"
mkdir -p "$WORK" "$OUT/logs"

log() { echo -e "\n[$(date '+%F %T')] $*"; }

# ---------------------------------------------------------------- 0. system + GPU
log "0/6 System packages and GPU"
if ! command -v tmux >/dev/null || ! command -v git >/dev/null || ! command -v unzip >/dev/null; then
  sudo apt-get update -y
  sudo apt-get install -y git tmux wget curl unzip xz-utils htop
fi
if ! nvidia-smi >/dev/null 2>&1; then
  log "No NVIDIA driver found – installing it now."
  sudo apt-get update -y
  sudo apt-get install -y ubuntu-drivers-common
  sudo ubuntu-drivers install || sudo apt-get install -y nvidia-driver-550-server
  echo
  echo ">>> Driver installed. Reboot the machine (sudo reboot), reconnect, and run this script again. <<<"
  exit 0
fi
nvidia-smi
df -h "$WORK" | tail -1

# ---------------------------------------------------------------- 1. Python env
log "1/6 Python 3.10 environment (PyTorch 2.5.1 / CUDA 12.1)"
if [ ! -x "$PY" ]; then
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  uv venv --python 3.10 "$VENV"
  uv pip install --python "$PY" \
    --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    torch==2.5.1 torchvision==0.20.1 numpy==1.26.4 opencv-python-headless==4.10.0.84 \
    segment-anything segment-anything-hq scikit-learn timm "positional-encodings[pytorch]" \
    imgaug pandas gdown matplotlib pyarrow "huggingface_hub>=0.24"
fi
export PATH="$VENV/bin:$PATH"
"$PY" -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

# ---------------------------------------------------------------- 2. code
log "2/6 SALAD code"
[ -d "$REPO" ] || git clone https://github.com/MaticFuc/SALAD.git "$REPO"
cd "$REPO"
git rev-parse HEAD | tee "$OUT/logs/salad_commit.txt"

# ---------------------------------------------------------------- 3. data + composition maps
log "3/6 MVTec LOCO + official composition maps"
LOCO="$REPO/data/mvtec_loco"
if [ ! -d "$LOCO/juice_bottle/train" ]; then
  bash download_loco.sh > /dev/null
  # if the archive extracted into a sub-folder, move categories up one level
  nested=$(find "$LOCO" -mindepth 2 -maxdepth 2 -type d -name juice_bottle | head -1 || true)
  if [ -n "$nested" ] && [ "$(dirname "$nested")" != "$LOCO" ]; then mv "$(dirname "$nested")"/* "$LOCO"/; fi
fi
[ -d "$REPO/data/mvtec_loco_composition_maps/juice_bottle" ] || bash download_composition_maps.sh
for c in $CATEGORIES; do
  for s in train/good validation/good test/good test/logical_anomalies test/structural_anomalies; do
    [ -n "$(ls -A "$LOCO/$c/$s" 2>/dev/null)" ] || { echo "Missing $LOCO/$c/$s"; exit 1; }
    [ -n "$(ls -A "$REPO/data/mvtec_loco_composition_maps/$c/$s" 2>/dev/null)" ] || { echo "Missing maps for $c/$s"; exit 1; }
  done
  echo "OK  $c  (train $(ls "$LOCO/$c/train/good" | wc -l) imgs)"
done

# ---------------------------------------------------------------- eval mode: released weights
if [ "$MODE" = "eval" ]; then
  log "EVAL: authors' released weights"
  [ -n "$(find "$REPO/results" -name teacher_final.pth 2>/dev/null | head -1)" ] || bash download_pretrained_models.sh
  for c in $CATEGORIES; do
    t=$(find "$REPO/results" -path "*/$c/teacher_final.pth" | head -1 || true)
    [ -n "$t" ] || { echo "No released weights for $c"; continue; }
    wdir=$(dirname "$(dirname "$t")")
    # known issue (GitHub #6): released breakfast_box/autoencoder_final.pth is corrupted
    "$PY" test_salad.py --category "$c" --output_dir "$wdir" 2>&1 | tee "$OUT/logs/${c}_eval.log" \
      || echo "!! $c failed – see $OUT/logs/${c}_eval.log"
  done
fi

# ---------------------------------------------------------------- 4. ImageNet (train mode only)
if [ "$MODE" = "train" ]; then
  log "4/6 ImageNet-1k train set (EfficientAD penalty loss, as in the paper)"
  if [ ! -f "$WORK/imagenet/.complete" ]; then
    [ -n "${HF_TOKEN:-}" ] || { echo "Set HF_TOKEN first (see instructions)."; exit 1; }
    mkdir -p "$IMAGENET" "$WORK/imagenet/shards_done"
    "$PY" - <<'PYEOF'
import os, io, shutil, pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
work = os.environ.get("WORK", os.path.expanduser("~/salad"))
root, done_dir = f"{work}/imagenet/train", f"{work}/imagenet/shards_done"
tmp = f"{work}/imagenet/tmp"; os.makedirs(tmp, exist_ok=True)
N = 294
def do(i):
    name = f"train-{i:05d}-of-{N:05d}.parquet"
    if os.path.exists(f"{done_dir}/{name}"): return 0
    p = hf_hub_download("ILSVRC/imagenet-1k", f"data/{name}", repo_type="dataset",
                        local_dir=f"{tmp}/{i}", token=os.environ["HF_TOKEN"])
    t = pq.read_table(p, columns=["image", "label"])
    imgs, labels = t.column("image").to_pylist(), t.column("label").to_pylist()
    for j, (im, lab) in enumerate(zip(imgs, labels)):
        d = f"{root}/{lab:04d}"; os.makedirs(d, exist_ok=True)
        with open(f"{d}/{i:03d}_{j:05d}.JPEG", "wb") as f: f.write(im["bytes"])
    shutil.rmtree(f"{tmp}/{i}", ignore_errors=True)
    open(f"{done_dir}/{name}", "w").close()
    return len(imgs)
total = 0
with ThreadPoolExecutor(6) as ex:
    for k, n in enumerate(ex.map(do, range(N))):
        total += n
        if k % 10 == 0: print(f"  shards {k+1}/{N} processed", flush=True)
print("ImageNet extraction finished")
PYEOF
    n=$(ls "$WORK/imagenet/shards_done" | wc -l)
    [ "$n" -eq 294 ] || { echo "Only $n/294 ImageNet shards done – re-run the script to resume."; exit 1; }
    touch "$WORK/imagenet/.complete"
  fi
  echo "ImageNet classes: $(ls "$IMAGENET" | wc -l)"

# ---------------------------------------------------------------- 5. training
  log "5/6 Training SALAD – $TRAIN_STEPS steps per category"
  for c in $CATEGORIES; do
    if [ -f "$OUT/results/$c/comp_unet_final.pth" ]; then echo "skip $c (already trained)"; continue; fi
    log "START $c"
    start=$(date +%s)
    "$PY" train_salad.py --category "$c" --train_steps "$TRAIN_STEPS" --seed "$SEED" \
        --imagenet_train_path "$IMAGENET" --output_dir "$OUT/results/" 2>&1 \
        | tee "$OUT/logs/${c}_train.log" \
      || { echo "!! $c failed – see $OUT/logs/${c}_train.log"; continue; }
    rm -f "$OUT/results/$c/"*_tmp.pth
    log "END $c after $(( ($(date +%s) - start) / 60 )) min"
  done
fi

# ---------------------------------------------------------------- 6. results table
log "6/6 Results (paper metric = mean of logical AUROC and structural AUROC)"
MODE="$MODE" OUT="$OUT" CATEGORIES="$CATEGORIES" "$PY" - <<'PYEOF'
import os, re, pandas as pd
PAPER = {"breakfast_box": (92.9, 85.7, 89.3), "juice_bottle": (99.7, 99.6, 99.7),
         "pushpins": (100.0, 98.8, 99.4), "screw_bag": (93.9, 96.0, 95.0),
         "splicing_connectors": (96.0, 98.6, 97.3)}
mode, out = os.environ["MODE"], os.environ["OUT"]
rows = []
for c in PAPER:
    f = f"{out}/logs/{c}_{mode}.log"
    if not os.path.exists(f): continue
    m = re.findall(r"All Logical: ([\d.]+), Struct: ([\d.]+)", open(f, errors="ignore").read())
    if not m: continue
    lg, st = map(float, m[-1])
    p = PAPER[c]
    rows.append({"category": c, "logical": round(lg, 2), "structural": round(st, 2), "mean": round((lg + st) / 2, 2),
                 "paper_logical": p[0], "paper_structural": p[1], "paper_mean": p[2],
                 "gap_mean": round((lg + st) / 2 - p[2], 2)})
if not rows:
    print("No results found yet."); raise SystemExit
df = pd.DataFrame(rows)
if len(df) == 5:
    avg = df.drop(columns="category").mean().round(2)
    df = pd.concat([df, pd.DataFrame([{"category": "AVERAGE", **avg.to_dict()}])], ignore_index=True)
print(df.to_string(index=False))
df.to_csv(f"{out}/results_table_{mode}.csv", index=False)
print(f"\nSaved {out}/results_table_{mode}.csv")
PYEOF

log "ALL DONE. Remember to STOP the workspace in RACE to save credits."
