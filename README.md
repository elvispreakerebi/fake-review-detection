# Fake Review Detection: Traditional ML vs Deep Learning

Binary classification of product reviews as **fake (computer-generated, CG)** or **real (original, OR)** using review text and optional metadata. This repository accompanies a summative machine learning project that compares **logistic regression (LR)** with **TF-IDF** features against a **Bidirectional Long Short-Term Memory (BiLSTM)** deep learning model.

---

## Problem Statement

Online reviews influence purchase decisions, but fake or computer-generated reviews can distort product perception and undermine trust. This project trains and evaluates classifiers to predict whether a review is **CG** (computer-generated, treated as fake) or **OR** (original, treated as genuine). The goal is high accuracy while balancing **false positives** (genuine reviews wrongly flagged as fake) and **false negatives** (fake reviews missed), so that the system is suitable for content moderation or triage in real-world settings.

---

## Dataset

- **Name:** Fake Reviews Dataset  
- **File:** `fake_reviews_dataset.csv`  
- **Columns:** `category`, `rating`, `label` (CG or OR), `text_` (review content)  
- **Usage:** The notebook supports loading from a **local path** or downloading from **Google Drive** (share link or file ID).

### Obtaining the data

1. **Local file:** Place `fake_reviews_dataset.csv` in the same directory as the notebook (or pass the path to `load_data()`).
2. **Google Drive:** Use the default share URL in the notebook, or set your own:
   ```python
   df = load_data(google_drive_url='https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing')
   ```
   If the file is already present locally, it will not be re-downloaded.

---

## Approach

### Traditional ML: Logistic Regression + TF-IDF

- **Features:** Term frequency–inverse document frequency (TF-IDF) on review text (unigrams and bigrams; optional trigrams in one experiment), with optional one-hot category and numeric rating.
- **Model:** Logistic regression with L2 regularisation (`C` and solver varied across experiments), balanced class weights.
- **Splits:** Stratified train / validation / test (e.g. 70% / 15% / 15%) with a fixed random seed for reproducibility.
- **Metrics:** Accuracy, precision, recall, F1, ROC-AUC, log loss, confusion matrix.

### Deep Learning: BiLSTM

- **Architecture:** Embedding → Bidirectional LSTM → Dense (ReLU) → Dropout → Sigmoid output.
- **Training:** Binary cross-entropy loss, Adam or AdamW optimizer, class weights from the training set.
- **Regularisation:** Dropout, optional L2 weight decay, early stopping on validation loss, reduce-learning-rate-on-plateau.
- **Splits:** Same stratified train/val/test as the ML pipeline.
- **Metrics:** Accuracy, precision, recall, F1, ROC-AUC, test loss, confusion matrix, FN/FP counts.

### Experiments

- **Eight LR experiments:** Different solvers (e.g. saga, lbfgs), regularisation strength `C` (e.g. 0.5–1.0), and n-gram settings. Ranked by test accuracy.
- **Eight BiLSTM experiments:** Varied embedding size, LSTM units, dropout, learning rate, batch size, L2, and optimizer (Adam vs AdamW). Ranked by F1, then ROC-AUC, accuracy, then total errors (FN + FP).

Best LR and best BiLSTM are compared side-by-side (summary table, bar chart, confusion matrices). Learning curves and ROC curves are produced in the notebook for diagnosis and reporting.

---

## Repository structure

- **`README.md`** (this file): Project overview, dataset, approach, setup, and usage.
- The main implementation and analysis live in the Jupyter notebook **`fake_reviews_ml_dl_suumative.ipynb`** (summative project notebook). If the notebook is part of this repo, it will appear in the root; otherwise use the path where you have stored it.
- A written report (**`Fake_Review_Detection_Report.md`**) provides the full academic write-up (introduction, literature review, methodology, results, discussion, conclusion, references) and can be placed in the repo or in a parent directory.

---

## Setup and dependencies

- **Python:** 3.11+ recommended (notebook tested with 3.11.13).  
- **Key libraries:**
  - Data & ML: `pandas`, `numpy`, `scikit-learn` (train/test split, TF-IDF, logistic regression, metrics).
  - Deep learning: `tensorflow` (≥2.16 used in development).
  - Visualisation: `matplotlib`, `seaborn`.
  - Optional (for Google Drive download): `gdown`.

### Install (example)

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn gdown
```

Or use a environment file if you maintain one:

```bash
pip install -r requirements.txt
```

### Reproducibility

The notebook sets:

- `np.random.seed(42)`  
- `tf.random.set_seed(42)`  
- `random_state=42` in `train_test_split` (and equivalent in preprocessing)

Run cells from top to bottom so that data loading, preprocessing, and model training use the same splits and seeds.

---

## How to run

1. **Clone or download** this repository (and ensure the notebook and, if used, the report are in the expected locations).
2. **Get the dataset:** Either place `fake_reviews_dataset.csv` in the notebook directory or use the Google Drive option in the notebook (see Dataset above).
3. **Install dependencies** (see Setup and dependencies).
4. **Open** `fake_reviews_ml_dl_suumative.ipynb` in Jupyter or a compatible environment.
5. **Run all cells** in order. Execution time depends on hardware; the BiLSTM experiments are the most expensive.

The notebook is organised in sections (e.g. imports, data loading, preprocessing, LR experiments, BiLSTM experiments, comparison tables, visualisations). Running it end-to-end reproduces the comparison tables, best-LR vs best-BiLSTM summary, bar chart, and confusion matrices.

---

## Notebook outline

| Section | Content |
|--------|--------|
| 1 | Import libraries (pandas, sklearn, TensorFlow, matplotlib, seaborn, etc.) |
| 2 | Data loading and preprocessing (e.g. `load_data`, `preprocess_data_for_ml`) |
| 3 | Traditional ML: logistic regression pipeline |
| 4 | Deep learning: BiLSTM pipeline (and optional functional API) |
| 5 | Visualisation helpers (learning curves, ROC, etc.) |
| 6 | Load and preprocess data (run pipeline on dataset) |
| 7 | LR: eight experiments and comparison table (ranked by accuracy) |
| 8 | BiLSTM: eight experiments and comparison table (ranked by F1, ROC-AUC, accuracy, errors) |
| 9 | (Optional) Functional API BiLSTM |
| 10 | Results summary: best LR vs best BiLSTM (summary table) |
| 11 | Visualisations: bar chart (accuracy & F1), side-by-side confusion matrices |

---

## Results summary (from the notebook/report)

- **Best LR (e.g. lr_exp8):** ~94% test accuracy, F1 ~0.94, ROC-AUC ~0.986. Good interpretability and speed; slightly more false negatives than the best BiLSTM.
- **Best BiLSTM (e.g. dl_exp4):** ~95% test accuracy, F1 ~0.95, ROC-AUC ~0.988, fewer false negatives. Prefer when maximising detection performance is the goal.
- **Trade-off:** Use the best BiLSTM when catching more fakes is priority; use the best LR when interpretability, training/inference speed, or simpler (e.g. CPU-only) deployment is more important. The confusion matrices and summary table in the notebook support choosing a model and threshold for your application.

---

## Report and references

The written report (**Fake_Review_Detection_Report.md**) includes:

- Introduction, literature review, methodology, results, discussion, conclusion.  
- Full experiment tables (all eight LR and eight BiLSTM runs), Table I (best LR vs best BiLSTM), and interpretation of figures.  
- Abbreviations (e.g. BiLSTM, TF-IDF, ROC-AUC, FN/FP) and IEEE-style references.

For full details, tables, and citations, see that document.

---

## License and attribution

This project was developed as a summative assignment for a machine learning module. The Fake Reviews Dataset is from existing work on fake review and deceptive text detection; see the report references for relevant literature and dataset sources.
