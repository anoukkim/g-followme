# **BLIP Fine-Tuning Project**

This project fine-tunes the BLIP (Bootstrapped Language-Image Pretraining) model on a custom dataset for image captioning. The dataset used is the **Chest X-rays (Indiana University)** dataset from Kaggle.

---

## **Dataset**
### **Dataset Source**
- **Dataset Name**: [Chest X-rays (Indiana University)](https://www.kaggle.com/datasets/openi/chest-xrays)
- **Original Source**: [Open-i dataset](https://openi.nlm.nih.gov/)
- **Description**:
  - This open-access chest X-ray collection contains images and metadata from Indiana University.
  - The original images were in DICOM format and were processed into PNG format:
    - Clipped 0.5% of DICOM pixel values at top/bottom to eliminate extreme outliers.
    - Linearly scaled pixel values to fit the range 0-255.
    - Resized images to 2048px on the shorter side.
  - Metadata, including image labels and reports, was downloaded using the Open-i API.

---
