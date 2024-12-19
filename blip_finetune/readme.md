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

## **Preprocessing Steps**

The dataset labels were derived from the `reports.csv` file. Below is the step-by-step process used to preprocess the data:

1. **Merge Datasets**:
   - Merged `projections.csv` and `reports.csv` using the `uid` column to ensure each image has a corresponding report.
2. **Column Removal**:
   - Deleted the following columns: `uid`, `indication`, `projection`, `MeSH`, `Problems`, `comparison`, and `image`.
3. **Label Generation**:
   - Created a new column, `label`, by concatenating the `findings` and `impression` columns.
4. **Data Cleaning**:
   - Removed rows where the `label` column was empty (40 rows).
   - Applied text cleaning as follows:
     1. Replaced certain patterns (e.g., `XXXX`) with specific terms:
        - `comparison XXXX` → `''`
        - `x-XXXX` → `x-ray`
        - `XXXX examination` → `examination`
        - `pulmonary XXXX` → `pulmonary vascularity`
        - `costophrenic XXXX` → `costophrenic angles`
        - `XXXX are normal` → `''`
     2. Removed date-related information.
     3. Removed placeholders like `XXXX` and additional unnecessary text patterns.
5. **Text Normalization**:
   - Converted all text to lowercase.
   - Expanded contractions (e.g., `isn't` → `is not`).
   - Removed special characters and numbers.
   - Removed words with two or fewer characters (except `no` and `ct`).
6. **Final Cleanup**:
   - Removed remaining special characters for consistent formatting.

---

## **Features**
- **Dataset Management**: Process and load the chest X-ray dataset.
- **Custom Labels**: Preprocessed and cleaned labels derived from radiology reports.
- **Model Loading**: Use a pretrained BLIP model for fine-tuning on custom image-caption pairs.
- **Training Pipeline**: Configure training using `TrainingArguments` from Hugging Face.
- **Evaluation**: Compute BLEU scores on test captions for quantitative evaluation.

---

## **Project Structure**
```plaintext
blip_finetune/
│
├── main.py                  # Entry point of the project
├── config.json              # Configuration file for paths and hyperparameters
├── data.py                  # Dataset preparation and splitting logic
├── model.py                 # Model and processor initialization
├── train.py                 # Training logic
├── evaluate.py              # Evaluation and BLEU scoring
├── utils.py                 # Utility functions (e.g., collate function)
├── environment.yml          # Conda environment setup
├── README.md                # Project documentation
```

## Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/anoukkim/g-followme/blip-finetune.git
cd blip-finetune
```

### Step 2: Set Up the Environment

Install the necessary dependencies using Conda:

```bash
conda env create -f environment.yml
conda activate blip-finetune-env
```

### Step 3: Dataset

Download the dataset from Kaggle and preprocess it using the described steps. Save it to a directory and specify the path in config.json under dataset_path.

## Usage

### Training

To start fine-tuning, run:
```bash
python main.py
```
### Evaluation

Generate captions and compute BLEU scores for the test set:
```bash
python evaluate.py
```
### Results

After training, the fine-tuned model generates captions for chest X-rays. BLEU scores are computed to quantify the performance.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
