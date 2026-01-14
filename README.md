# Interpretable Multi-Agent Deep Learning for Real-Time Critical Findings in Non-Contrast Head CT

This repository documents the experimental setup for **non-contrast head CT (NCCT)** critical finding analysis, including **critical-value screening (Task 1: binary classification)** and **critical subtype recognition (Task 2: five-class classification)**, together with a clinically motivated, interpretable **multi-agent (MAS)** structured reporting workflow. 

---

## 1. Dataset

### 1.1 Overview

* **Source**: Retrospective multicenter NCCT cohort collected from **8 medical centers in China**.
* **Scale**: **7,877** NCCT cases acquired between **2021–2025**.
* **After preprocessing/QC**: **5,490** standardized high-quality cases were retained (approximately **69.7%**). 

### 1.2 Tasks and Class Distribution

**Task 1: Critical-value screening (binary)**

* Non-critical: **2,027**
* Critical: **3,463** 

**Task 2: Critical subtype classification (five-class; performed on critical-positive cases)**

* Mixed critical findings: **1,063**
* Acute intracerebral hemorrhage: **820**
* Acute subdural/epidural hemorrhage: **339**
* Acute diffuse subarachnoid hemorrhage: **667**
* Acute large-territory cerebral infarction: **574** 

### 1.3 Inclusion / Exclusion Criteria (condensed)

**Inclusion**: Major acute critical entities (multiple hemorrhage types, acute infarction, mixed findings), non-critical findings, follow-up scans from the same patient, and both traumatic/spontaneous hemorrhage cases. 
**Exclusion**: Poor image quality; prior surgical intervention before follow-up; chronic hemorrhage/infarction; intracerebral hemorrhage due to vascular malformation or tumor. 

### 1.4 Standardization

* **Spatial resizing**: 3D NIfTI volumes were resampled using **trilinear interpolation** to a fixed size of **(96, 256, 256)**. 
* **Intensity normalization**: Linear normalization to **[-1, 1]** (mapped to [0,1] then transformed to [-1,1]). 

### 1.5 Dataset Access and Links

* **Official code repository (as stated in the paper)**: `Interpretable-MA-HeadCT`. 
* **Data availability (as stated in the paper)**: The dataset is **not publicly downloadable**; it can be requested from the **corresponding author** after publication, subject to reasonable request and compliance. 

### 1.6 Train/Validation Split Principles (recommended for leakage-free replication)

The paper emphasizes real-world multi-center data and includes follow-up scans; to ensure a correct experimental split:

1. **Patient-level splitting**: All scans (including follow-ups) from the same patient must remain in the **same split** to avoid information leakage. 
2. **Stratification**: Stratify by **critical/non-critical** for Task 1 and by **five subtypes** for Task 2 to preserve class proportions. 
3. **Optional external generalization**: For stronger multi-center validation, consider holding out one center as an external test set (center-based split). 

---

## 2. Method Summary (Brief)
![Pipeline](main/workflow.png)


* **Two-stage cascade**:

  * **Task 1**: A lightweight 3D CNN for rapid critical-value screening (high sensitivity prioritized).
  * **Task 2**: A deeper 3D CNN for five-class subtype classification, using multi-level feature fusion and attention-based recalibration. 
* **Interpretability and evidence chain**: key-slice localization, HU-driven lesion saliency mapping, and brain-herniation quantification via midline shift estimation. 
* **Multi-agent structured reporting (MAS)**: observer/scholar/auditor/reporter agents with iterative cross-checking (up to n=5 iterations), with heavier reasoning triggered mainly for ambiguous/complex cases. 

---

## 3. Experimental Results (Tables from the Paper)

### 3.1 Task 1: Critical-value Screening (Feature Branches and Loss Design)

| Configuration                   |   Accuracy |     Recall |  Precision |   F1-score |    AUC |
| ------------------------------- | ---------: | ---------: | ---------: | ---------: | -----: |
| local feature                   |     0.9803 |     0.9636 |     0.9916 |     0.9774 | 0.9964 |
| global feature                  |     0.9887 |     0.9800 |     0.9945 |     0.9872 | 0.9975 |
| fusion feature                  |     0.9851 |     0.9791 |     0.9872 |     0.9831 | 0.9953 |
| fusion feature + composite loss | **0.9909** | **0.9942** | **0.9942** | **0.9928** | 0.9907 |



### 3.2 Task 2: Five-class Subtype Classification (Feature Scale Settings)

| Feature setting     |   Accuracy |     Recall |  Precision |   F1-score |        AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| high scale feature  |     0.8315 |     0.8294 |     0.8357 |     0.8317 |     0.9470 |
| low scale feature   |     0.8324 |     0.8291 |     0.8368 |     0.8318 |     0.9472 |
| mid scale feature   |     0.8069 |     0.8078 |     0.8366 |     0.8161 |     0.9331 |
| multi scale feature | **0.8517** | **0.8618** | **0.8605** | **0.8602** | **0.9598** |



### 3.3 Inference Efficiency

**Time per image (ms/img)**

| Model                              | Time (ms/img) |
| ---------------------------------- | ------------: |
| Binary classification (Task 1)     |        16.729 |
| Five-class classification (Task 2) |        16.281 |



**Time per case (s/case)**

| Module                  | Time (s/case) |
| ----------------------- | ------------: |
| CritiScan Agent         |         0.379 |
| Classify Agent          |        1.2947 |
| Multi-Agent Report Team |       302.992 |
