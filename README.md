# Skin Lesion Classification using ResNet50 (HAM10000)

This project implements a PyTorch-based pipeline for multi-class skin lesion classification using the HAM10000 dataset. It includes data preprocessing, augmentation, a custom Dataset class, ResNet50 fine-tuning, and full evaluation metrics.

---

## Dataset

The project uses the HAM10000 skin cancer dataset.

Expected structure:

Each image is referenced using `image_id + ".jpg"` from the metadata file.

---

## Features

- Metadata loading and label encoding  
- Custom PyTorch Dataset for dual image folders  
- Data augmentation: resize, flip, rotation, normalization  
- Fine-tuned **ResNet50** with pretrained ImageNet weights  
- Evaluation: accuracy, precision, recall, F1-score, confusion matrix  
- GPU acceleration support  

---

## Installation

Install dependencies:

```bash
pip install torch torchvision pandas numpy pillow scikit-learn tqdm

