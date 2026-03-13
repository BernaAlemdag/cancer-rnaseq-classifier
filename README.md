# 🧬 Cancer Type Classification from RNA-Seq Gene Expression

> **Bioprocess & Bioengineering Portfolio — Project 01 / 2026**  
> *Berna ALEMDAG*

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=flat-square)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red?style=flat-square)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.51-purple?style=flat-square)](https://shap.readthedocs.io)
[![UMAP](https://img.shields.io/badge/UMAP-0.5-teal?style=flat-square)](https://umap-learn.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 📖 Overview

When a cancer metastasizes across the body, one of the most critical and sometimes unanswerable clinical questions is: **where did it originate?** Identifying the tissue of origin directly determines treatment strategy. Breast cancer, kidney cancer, and lung cancer each require fundamentally different therapies; prescribing the wrong one not only fails to help the patient but can accelerate disease progression.

Historically, up to **3–5% of all cancer patients** received a Cancer of Unknown Primary (CUP) diagnosis, meaning conventional histopathology and imaging could not identify the primary tumor site. Even with modern diagnostics, CUP still accounts for **1–2% of all cancers worldwide** representing tens of thousands of patients each year with limited treatment options.
*(Rassy & Pavlidis, Nature Reviews Clinical Oncology, 2020)*

RNA sequencing (RNA-Seq) offers a compelling solution. Every human cell maintains a distinct gene expression profile a molecular fingerprint determined by which of its ~20,000 genes are active and at what level. Remarkably, **even after a cancer cell leaves its organ of origin and colonizes distant tissue, it retains much of this expression fingerprint.** A metastatic breast cancer cell in the liver still behaves, transcriptionally, like a breast cell.

This project asks a direct question: **can a machine learning model learn to classify 5 cancer types from raw RNA-Seq gene expression profiles alone and can we identify which specific genes drive each classification decision?**

---

## 🎯 Cancer Types

| Code | Cancer | Tissue of Origin | Samples |
|------|--------|-----------------|---------|
| BRCA | Breast Invasive Carcinoma | Breast | 300 |
| KIRC | Kidney Renal Clear Cell Carcinoma | Kidney | 146 |
| COAD | Colon Adenocarcinoma | Colon | 78 |
| LUAD | Lung Adenocarcinoma | Lung | 141 |
| PRAD | Prostate Adenocarcinoma | Prostate | 136 |
| **Total** | | | **801** |

> ⚠️ Notable class imbalance: BRCA (n=300) vs COAD (n=78) a nearly **4-fold difference** explicitly addressed with SMOTE.

---

## 🗺️ Pipeline

```
Raw RNA-Seq (20,531 genes · 801 patients)
        │
        ├── Variance Filtering      remove near-zero-variance genes → 19,536 retained
        │
        ├── Train / Test Split      stratified 80/20 → 641 train · 160 test
        │
        ├── Standard Scaling        fit on train only → applied to both sets
        │
        ├── SMOTE                   oversample minority classes → balanced training set
        │
        ├── PCA (100 components)    ~75% variance retained → compressed feature space
        │
        ├── 4 ML Models             RF · SVM · XGBoost · MLP
        │       ├── Standard mode   (imbalanced)
        │       └── SMOTE mode      (balanced)
        │
        ├── Evaluation              Accuracy · ROC-AUC · Per-class F1 · 5-fold CV
        │
        ├── PCA vs UMAP             linear vs non-linear structure comparison
        │
        └── SHAP Analysis           full gene space (no PCA) → gene-level explainability
```

---

## 📊 Results

All four models achieved high classification accuracy on the held-out test set (160 samples).

| Model | Accuracy (Standard) | Accuracy (+SMOTE) | ROC-AUC |
|-------|--------------------|--------------------|---------|
| Random Forest | 96.89% | **98.14%**  | 0.9988 |
| SVM (RBF) | 98.14% | 98.14% | **0.9999** |
| MLP Neural Net | **98.76%** | 98.14% | 0.9993 |
| XGBoost | 98.14% | 96.27% | 0.9979 |


### Key observations:
- **Random Forest** showed the greatest benefit from SMOTE (+1.24%), confirming that ensemble tree methods respond well to balanced training distributions
- **SVM** achieved identical accuracy in both conditions, indicating robustness to class imbalance
- **XGBoost** showed a slight decline with SMOTE (-1.86%), suggesting its internal weighting mechanisms interact with oversampling in complex ways
- All models achieved ROC-AUC above **0.997** near perfect class separation at the probability level

---

## 🔬 Methodology

### 5.1 Preprocessing

Variance filtering was applied first, removing genes with near-zero variance across all samples (threshold = 0.05). This reduced the feature space from 20,531 to **19,536 informative genes**, eliminating noise without losing biologically relevant signal.

Data was split into **80% training (641 samples)** and **20% test (160 samples)** using stratified sampling to preserve class proportions. StandardScaler was applied fit exclusively on the training set and applied to both sets to prevent data leakage.

**SMOTE** (Synthetic Minority Oversampling Technique) was then applied to the training set to address class imbalance. By generating synthetic samples for minority classes, all 5 cancer types were equalized in the training distribution. Models were trained in both standard and SMOTE-balanced modes to measure the direct impact of class balancing on per-class performance.

### 5.2 Dimensionality Reduction

**PCA** compressed 19,536 genes into **100 principal components**, retaining approximately 75% of total variance. This compression reduces the computational cost of training and mitigates the curse of dimensionality a common challenge when the number of features vastly exceeds the number of samples.

**UMAP** was additionally computed for visualization. Unlike PCA (linear), UMAP is a non-linear technique that preserves local neighbourhood structure. The resulting 2D projection revealed **5 perfectly separated cancer clusters** a finding that PCA's linear projection obscured entirely confirming that non-linear gene co-expression patterns carry discriminative signal in the data.

| | PCA | UMAP |
|--|-----|------|
| Type | Linear | Non-linear |
| Speed | Fast | ~1–2 min |
| Axes | Interpretable (% variance) | Latent manifold |
| Best for | Compression, classification | Visualization, cluster analysis |
| Result | Overlapping clusters in 2D | 5 perfectly separated islands |

### 5.3 Models & Evaluation

Four models were trained and compared: **Random Forest** (n=200 trees), **SVM** (RBF kernel, C=10), **XGBoost** (n=200 rounds, lr=0.1), and **MLP Neural Network** (256-128-64 architecture). Each model was trained in two conditions standard and SMOTE-balanced and evaluated on accuracy, ROC-AUC, and per-class F1 score.

**5-fold cross-validation** was performed within the training set to assess model stability independently of the held-out test set. This means the test set was never touched during model selection only used once for final evaluation.

---

## 🧠 Explainability — SHAP Analysis

Accuracy alone does not answer the biological question. To identify which genes drive each classification, **SHAP (SHapley Additive exPlanations)** analysis was performed using a separate XGBoost model retrained on the **full 19,536-gene space-without PCA compression**.

This design choice is critical: SHAP values computed on PCA components would correspond to mathematical abstractions, not biological entities. By bypassing PCA for the explainability model, **every SHAP value maps directly to a named gene.**

### Five SHAP visualizations generated per cancer type:

| Visualization | What it shows |
|--------------|---------------|
| Global bar chart | Top 20 genes by mean \|SHAP\| across all classes |
| Beeswarm plots | Per-cancer expression direction (red=high, blue=low) |
| Heatmap | Top 25 genes × 5 cancer types-class specificity |
| Waterfall plots | Individual patient explanation-one per cancer type |
| Gene direction chart | Over-expressed vs under-expressed genes per class |

These outputs transform the classifier from a black box into a **biologically interpretable discovery tool**-each cancer type's molecular signature becomes directly readable from the model's decisions.

---

## 📁 Repository Structure

```
📦 cancer-rnaseq-classifier
 ┣ 📓 cancer_rna_classifier.ipynb    ← Main notebook (all code + plots)
 ┣ 📄 01_cancer_rnaseq_project_pdf   ← Project description
 
```

---

## 🚀 Quick Start

**1. Clone the repository**
```bash
git clone https://github.com/bernaalemdag/cancer-rnaseq-classifier.git
cd cancer-rnaseq-classifier
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up Kaggle API**

Place your `kaggle.json` token in `~/.config/kaggle/`  
Get it from: [kaggle.com/account](https://www.kaggle.com/account)

**4. Run the notebook**
```bash
jupyter notebook cancer_rna_classifier.ipynb
```
> The dataset downloads automatically via `kagglehub` on first run — no manual download needed.

---

## 📦 Requirements

```
numpy
pandas
scikit-learn
xgboost
shap
umap-learn
imbalanced-learn
matplotlib
seaborn
kagglehub
jupyter
```

```bash
pip install numpy pandas scikit-learn xgboost shap umap-learn imbalanced-learn matplotlib seaborn kagglehub jupyter
```

---

## 🗃️ Dataset

| Property | Value |
|----------|-------|
| Source | [TCGA via Kaggle](https://www.kaggle.com/datasets/waalbannyantudre/gene-expression-cancer-rna-seq-donated-on-682016) |
| Patients | 801 |
| Genes | 20,531 |
| Cancer types | 5 |
| Missing values | None |
| Access | Free — requires Kaggle account |

---

## 🚀 Future Work

| Extension | Description |
|-----------|-------------|
| HGNC symbol mapping | Map anonymous gene IDs → named genes via Ensembl Biomart |
| Pathway enrichment | KEGG/GO analysis on top SHAP genes via gseapy |
| DESeq2 validation | Confirm SHAP genes are statistically differentially expressed |
| Multi-omics | Add DNA methylation + miRNA layers from TCGA |
| Deep learning | 1D-CNN or Transformer on raw expression vectors |
| Survival analysis | Clinical outcome prediction with TCGA clinical data |
| Interactive UMAP | Plotly 3D visualization for portfolio |

---

## 📚 References

-Dataset: TCGA RNA-Seq via Kaggle ·Tools: Python, scikit-learn, XGBoost, SHAP, UMAP, imbalanced-learn  ·  Fully open source

-Rassy, E., & Pavlidis, N. (2020). Progress in refining the clinical management of cancer of unknown primary in the molecular era. Nature reviews Clinical oncology, 17(9), 541-554.

-Weigelt, B., Glas, A. M., Wessels, L. F., Witteveen, A. T., Peterse, J. L., & van't Veer, L. J. (2003). Gene expression profiles of primary breast tumors maintained in distant metastases. Proceedings of the National Academy of Sciences, 100(26), 15901-15905.


---

## 👤 Author

**Berna ALEMDAG**  
Bioprocess & Bioengineering | Machine Learning


## 📋 Portfolio Series — 2026

This is **Project 01** of a 5-project Bioprocess & Bioengineering × Machine Learning portfolio.



---

*Dataset: TCGA RNA-Seq via Kaggle · Tools: Python, scikit-learn, XGBoost, SHAP, UMAP, SMOTE · License: MIT*
