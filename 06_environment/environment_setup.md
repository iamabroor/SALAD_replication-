# Environment Setup

## Platform

| Component | Value |
|---|---|
| Cloud platform | RACE (Research Accelerated Cloud Environment) |
| Instance type | AWS g5.2xlarge |
| GPU | NVIDIA A10G (24 GB VRAM) |
| CPU | AMD EPYC 7R32, 8 vCPUs |
| RAM | 32 GB |
| Storage | 200 GB SSD |
| OS | Ubuntu 22.04 LTS |

---

## GPU / CUDA

| Component | Version |
|---|---|
| NVIDIA driver | 525.x |
| CUDA | 11.8 |
| cuDNN | 8.x |

---

## Python Environment

```bash
# Create virtual environment
python3 -m venv ~/salad/venv
source ~/salad/venv/bin/activate

# Install dependencies
pip install --break-system-packages -r 06_environment/requirements.txt
```

---

## Software Versions

| Package | Version |
|---|---|
| Python | 3.10.x |
| PyTorch | 2.0.1+cu118 |
| torchvision | 0.15.2+cu118 |
| numpy | 1.24.x |
| scikit-learn | 1.3.x |
| timm | 0.9.x |
| Pillow | 10.x |
| matplotlib | 3.7.x |
| tqdm | 4.x |
| segment-anything | (SAM from Meta) |

---

## Data Locations (on RACE)

| Dataset | Path |
|---|---|
| MVTec LOCO AD | `~/salad/SALAD/data/mvtec_loco_anomaly_detection/` |
| ImageNet subset | `~/salad/imagenet/` |
| Pretrained weights | `~/salad/SALAD/weights/` |

---

## Training Time

| Category | Approximate Time (g5.2xlarge, 70k steps) |
|---|---|
| breakfast_box | ~6–7 hours |
| juice_bottle | ~6–7 hours |
| pushpins | ~6–7 hours |
| screw_bag | ~6–7 hours |
| splicing_connectors | ~6–7 hours |
| **Total (all 5)** | **~33–35 hours** |

---

## Cost Estimate

- g5.2xlarge on-demand: ~$1.21/hr
- Total training: ~40 hrs = **~$48–50 USD**
- Storage (200 GB): ~$0.20/hr for session
