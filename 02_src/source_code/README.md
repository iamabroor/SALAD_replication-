# Source Code

## SALAD Official Repository

The original SALAD source code is available at:
- **GitHub:** https://github.com/DecAd3/SALAD (check for public release)
- **arXiv:** https://arxiv.org/abs/2509.02101

## How This Replication Was Run

Training was conducted on the RACE HPC platform:
- **Instance:** AWS g5.2xlarge
- **GPU:** NVIDIA A10G (24 GB VRAM)
- **OS:** Ubuntu 22.04

### Clone and Setup

```bash
git clone https://github.com/DecAd3/SALAD ~/salad/SALAD
cd ~/salad/SALAD
pip install -r requirements.txt
```

### Data

```bash
# MVTec LOCO AD dataset
wget https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/mvtec_loco.tar.gz
tar xzf mvtec_loco.tar.gz -C ~/salad/SALAD/data/

# ImageNet subset (required for EfficientAD teacher training)
# See run_all.sh for download instructions
```

### Training

```bash
# Train all 5 categories
bash ~/salad/03_scripts/run_all.sh

# Or train a single category:
python train_salad.py --category breakfast_box --steps 70000
```

## Key Files in the SALAD Repo

| File | Purpose |
|---|---|
| `train_salad.py` | Main training script |
| `models/` | Teacher, Student, UNet, Discriminator architectures |
| `datasets/` | MVTec LOCO data loading |
| `eval_salad.py` | Evaluation / AUROC computation |
| `requirements.txt` | Python dependencies |

## Replication Scripts (this repo)

See `../03_scripts/` for the scripts used in this replication.
