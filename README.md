# Fine-Tuning Cellpose-SAM for 3D Cell Segmentation

Fine-tune **Cellpose-SAM** on custom 3D microscopy datasets using manually annotated instance segmentation masks.

This repository provides an end-to-end pipeline for:

- Dataset validation
- 3D volume preprocessing
- Conversion of 3D volumes into 2D training slices
- Fine-tuning a pretrained Cellpose-SAM model
- Running inference using the fine-tuned model
- Quantitative evaluation
- Saving predictions and training statistics

The project was developed for fluorescence microscopy datasets but can easily be adapted for other biomedical imaging modalities.

---

# Features

- Fine-tune the latest Cellpose-SAM model
- Supports 3D TIFF image volumes
- Automatic slice generation
- Dataset validation before training
- GPU acceleration (CUDA)
- Training loss visualization
- Model checkpoint saving
- Automatic inference after training
- Average Precision (AP) evaluation
- Clean modular codebase

---

# Repository Structure

```
fine-tuning-cellpose-sam/

│
├── config.py                 # Configuration parameters
├── train.py                  # Main training script
├── inference.py              # Inference utilities
├── dataset.py                # Dataset loading
├── preprocess.py             # Slice generation
├── evaluation.py             # AP computation
├── utils.py                  # Helper functions
│
├── data/
│   ├── train/
│   ├── test/
│
├── sliced_data/
│
├── models/
│
├── outputs/
│   ├── predictions/
│   ├── figures/
│   └── logs/
│
└── README.md
```

---

# Dataset Format

The repository expects TIFF images and corresponding instance masks.

```
train/

image_001.tif
image_001_masks.tif

image_002.tif
image_002_masks.tif
...
```

Similarly,

```
test/

image_101.tif
image_101_masks.tif
```

---

# Image Requirements

Images should satisfy:

- TIFF format
- Shape

```
(Z, Y, X)
```

Example

```
(78, 2048, 2048)
```

Recommended datatype

```
uint16
```

---

# Mask Requirements

Masks should be instance labels.

Example

```
0 = background

1 = cell 1

2 = cell 2

3 = cell 3
...
```

Every cell must have a unique integer label.

Recommended datatype

```
uint16
or
int32
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/SrujanGowda1998/fine-tuning-cellpose-sam.git

cd fine-tuning-cellpose-sam
```

Create environment

```bash
conda create -n cellpose-sam python=3.11
conda activate cellpose-sam
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Required Packages

Main dependencies include

- Python 3.11
- Cellpose
- PyTorch
- NumPy
- tifffile
- scikit-image
- matplotlib
- tqdm

---

# Configuration

Edit

```
config.py
```

Typical settings

```python
PRETRAINED_MODEL = "cpsam_v2"

BATCH_SIZE = 8

N_EPOCHS = 500

LEARNING_RATE = 1e-5

WEIGHT_DECAY = 0.1

MIN_TRAIN_MASKS = 1

USE_GPU = True
```

---

# Preparing the Dataset

The pipeline automatically

- validates image/mask pairs
- checks image dimensions
- verifies label consistency
- slices 3D volumes into 2D images
- generates training and testing datasets

No manual slicing is required.

---

# Training

Run

```bash
python train.py
```

Training performs

- Dataset validation
- Slice generation
- Cellpose-SAM fine-tuning
- Checkpoint saving
- Loss plotting
- Automatic inference on the test set
- Evaluation

---

# Output

Example directory

```
outputs/

predictions/

loss_curve.png

training.log

metrics.txt

models/

cellpose_sam_finetuned
```

---

# Inference

To run inference only

```bash
python inference.py
```

The script loads the trained model and predicts masks for unseen images.

---

# Evaluation

The repository computes

- Average Precision (AP)
- Precision
- Recall

Predictions are compared against ground truth instance masks.

---

# Training Pipeline

```
Raw 3D Images
        │
        ▼
Dataset Validation
        │
        ▼
Slice Generation
        │
        ▼
Load Training Data
        │
        ▼
Fine-tune Cellpose-SAM
        │
        ▼
Save Best Model
        │
        ▼
Inference
        │
        ▼
Evaluation
```

---

# GPU Support

Training automatically uses CUDA if available.

Example

```python
device = "cuda"
```

Otherwise,

```python
device = "cpu"
```

---

# Tested On

- Cellpose 4.x
- PyTorch 2.x
- CUDA GPUs
- Linux HPC clusters
- Windows

---

# Example Dataset

```
train/

241114_ced3gfp_worm_5.tif

241114_ced3gfp_worm_5_masks.tif

241114_ced3gfp_worm_6.tif

241114_ced3gfp_worm_6_masks.tif
```

---

# Citation

If this repository contributes to your research, please cite:

Cellpose-SAM

and

this repository.

---

# Acknowledgements

This work builds upon:

- Cellpose
- Cellpose-SAM
- PyTorch
- NumPy
- tifffile

Special thanks to the Cellpose developers for making the framework publicly available.

---

# License

MIT License

---

# Author

**Srujan Prakash Gowda**

EMBL Heidelberg

GitHub:

https://github.com/SrujanGowda1998

Research interests

- Biomedical Image Analysis
- Cell Segmentation
- Computer Vision
- Deep Learning
- Scientific AI
- Bioimage Informatics
