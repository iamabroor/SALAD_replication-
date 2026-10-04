# Training Configuration & Hyperparameters

## Hardware

| Setting | Value |
|---|---|
| Instance | AWS g5.2xlarge |
| GPU | NVIDIA A10G, 24 GB VRAM |
| CPU | 8 vCPUs |
| RAM | 32 GB |
| OS | Ubuntu 22.04 |

---

## Training Settings

### Global

| Parameter | Value |
|---|---|
| Training steps | 70,000 |
| Image resolution | 256 × 256 |
| Batch size | 1 (as per paper) |
| Random seed | 42 |
| Mixed precision | fp16 (where supported) |

### Appearance Branch (EfficientAD)

| Parameter | Value |
|---|---|
| Teacher backbone | WideResNet-101 (pretrained ImageNet) |
| Teacher features | 384 channels |
| Student architecture | Mirror of teacher |
| AE architecture | Encoder-Decoder (CNN) |
| Learning rate | 1e-4 |
| Optimizer | Adam |
| Teacher frozen | Yes |

### Composition Branch

| Parameter | Value |
|---|---|
| SAM model | ViT-B (vit_b checkpoint) |
| DINO model | ViT-B/8 (vit_base_patch8_224_dino) |
| DINO patch grid | 28 × 28 (for 224px input) |
| k-means clusters (k) | 6 |
| U-Net channels | 64 base |
| Discriminator | PatchGAN style |
| Learning rate | 1e-4 |
| Optimizer | Adam |

### Global Statistical Branch

| Parameter | Value |
|---|---|
| Feature source | Teacher (WideResNet-101) |
| Distance metric | Mahalanobis |
| Fitted on | Full training set |

---

## Evaluation

| Parameter | Value |
|---|---|
| Metric | Image AUROC |
| Split | Logical / Structural / Mean |
| Test set | MVTec LOCO AD (per category) |

---

## Seed Study

Additional seeds were run to assess variance:

| Category | Seeds Run |
|---|---|
| breakfast_box | 42, 0, 1 |
| juice_bottle | 42 |
| pushpins | 42 |
| screw_bag | 42 |
| splicing_connectors | 42 |

*(See `05_results/metrics/results_table.md` for per-seed results)*
