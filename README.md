# Overview

This project was developed as part of the Neuroengineering course, MSc in Biomedical Engineering, Politecnico di Milano (A.Y. 2024/2025)

The aim is to implement a **U-Net-based deep learning model** for **automatic segmentation of the vocal tract** in **dynamic speech MRI (dsMRI)** data. Accurate segmentation of vocal tract structures plays a critical role in studying **speech disorders** associated with neurodegenerative diseases, such as **apraxia of speech** and **dysarthria**.

Traditional diagnostic methods are either invasive or limited in resolution, while dsMRI provides a safe and high-quality alternative. Manual segmentation, however, is time-consuming and subjective, motivating the use of AI-based approaches.

# Project Description

- **Objective**: Develop and optimize an advanced U-Net architecture for automatic segmentation of seven vocal tract structures in dsMRI (background, upper lip, lower lip, soft palate, hard palate, tongue, and head).
- **Dataset**: 820 frames from four healthy subjects performing speech protocols, acquired in collaboration between UCSF and Politecnico di Milano.
- **Approach**:
  - Data preprocessing and augmentation to overcome dataset limitations.
  - Modified U-Net architecture inspired by [IMU-Net](https://pubmed.ncbi.nlm.nih.gov/33524814/) (with residual blocks, dilated convolutions, and dropout).
  - Hyperparameter tuning (dropout rate, number of filters, loss functions including Dice loss, weighted cross-entropy, and focal loss).
  - Evaluation through leave-one-subject-out cross-validation.
- **Results**: Achieved Dice coefficient above 76% across all classes, with best performance using dropout 0.2 and a combined Dice + weighted cross-entropy loss. Promising results also on a pathological patient with non-fluent variant primary progressive aphasia.

# Graphical User Interface (GUI)
To make the system accessible to clinicians, we developed a simple GUI that:

- Visualizes dsMRI frames as a video along with their segmentations.
- Displays plots of the segmented areas for each frame.
- Shows temporal variations of these areas over the course of the speech protocol.
  
This allows medical doctors to detect speech patterns and irregularities more intuitively.

# Repository structure
```
├── VocalTractSegmentation_presentation.pdf    # Final project presentation
├── VocalTractSegmentation_notebook.ipynb      # Jupyter Notebook with U-Net implementation
├── cross_validation.ipynb                     # Notebook for leave-one-subject-out cross-validation
├── grid_search.ipynb                          # Kaggle Notebook for hyperparameter tuning
├── GUI.py                                     # GUI script for dsMRI visualization and metrics
└── README.md                                  # Project documentation
```
# Credits
Developed by: Guia Baggini, Federica Burinato, Federico De Carlo, Sara Martuscelli, Matteo Missana

Course: Neuroengineering, Politecnico di Milano (A.Y. 2024/2025)

Professor: Pietro Cerveri – Tutor: Matteo Cavicchioli
