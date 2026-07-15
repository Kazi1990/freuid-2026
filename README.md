# FREUID 2026 - Identity Document Fraud Detection

Official submission for **The FREUID Challenge 2026 - IJCAI-ECAI**.

## Quick Start

### Build Docker Image
```bash
docker build -t freuid-repro:local .
Run Inference (No Network)
docker run --rm --network none \
  -v /path/to/test/images:/data:ro \
  -v "$(pwd)/out:/submissions" \
  freuid-repro:local
Output: /submissions/submission.csv
odel Architecture
Backbone: EfficientNet-B0 (ImageNet pretrained)
Head: Dropout(0.5) -> Linear(1280, 256) -> ReLU -> Dropout(0.3) -> Linear(1)
Output: Sigmoid-activated fraud probability [0, 1]
Training
Platform: Kaggle Notebook (NVIDIA T4 GPU)
Data: FREUID 2026 official training set
Fraud Rate: 42.32%
Input Size: 224x224
Normalization: ImageNet mean/std
Loss: BCEWithLogitsLoss
Optimizer: Adam
Inference
Batch size: 64
├── prepare_submission.py   # Docker inference script
├── Dockerfile              # Container definition
├── model.pth               # Trained weights
├── training.ipynb          # Training notebook
├── FREUID_Report.pdf       # Technical report
└── README.md
Prize Eligibility & Reproducibility
This repository complies with the FREUID 2026 reproducibility requirements:
Public git repository with OSI-approved license (MIT)
Runnable Docker container with --network none
Model weights frozen at commit
Technical report included
Test-Time Augmentation: 5 variants averaged
Random seed: 42
Repository Structure
