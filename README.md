# 🎗️ Breast Cancer Diagnosis & Predictive Analysis using Machine Learning

Ek end-to-end Machine Learning pipeline jo clinical measurements aur digital microscopic features ka upyog karke breast cancer tumors ko **Benign (Normal / Non-Cancerous)** aur **Malignant (Cancerous)** me classify karti hai. Is project me data preprocessing, exploratory data analysis (EDA), multi-model comparison, threshold tuning (ROC-AUC), feature selection, aur modern **WDBC (30 features)** dataset par evaluation shamil hai.

---

## 📌 Project Overview & Highlights
- **Binary Classification:** Clinical attributes ke aadhar par tumor malignancy detect karna.
- **Handling Data Imperfections:** Missing values imputation (Median strategy on `Bare_Nuclei`), dropping non-predictive identifiers (`ID`), aur target mapping (`0 = Benign`, `1 = Malignant`).
- **Statistical Correlation & EDA:** Pearson Correlation Matrix aur Heatmap analysis ke jariye multicollinearity (`Cell_Size` & `Cell_Shape` correlation = 0.91) ko identify kiya gaya.
- **Model Comparison:** Logistic Regression, Support Vector Machine (SVM - RBF Kernel), aur Random Forest Classifier ka pradarshan compare kiya gaya.
- **High Sensitivity (Recall):** Medical priority ke anusaar False Negatives (FN) ko minimize kiya gaya taaki cancer cases miss na hon.
- **Feature Selection & Ablation Study:** ANOVA F-test (`SelectKBest`) aur Random Forest Feature Importance se 9 features ko ghata kar Top 5 features par laaya gaya, jisme model ne apni **95.71% accuracy** ko barkarar rakha.
- **Dataset Scaling (WDBC 30 Features):** Digital cell measurements par model ko train karke **97.37% test accuracy** aur **0 False Positives (100% Precision)** achieve kiya gaya.

---

## 📊 Datasets Used
1. **Breast Cancer Wisconsin (Original):**
   - **Source:** UCI Machine Learning Repository
   - **Instances:** 699 rows
   - **Attributes:** 9 histological ratings (Scale: 1–10) + Class (2: Benign, 4: Malignant)
2. **Wisconsin Diagnostic Breast Cancer (WDBC):**
   - **Source:** Scikit-Learn / UCI Machine Learning Repository
   - **Instances:** 569 rows
   - **Attributes:** 30 continuous real-valued physical measurements (mean, standard error, and worst cell characteristics).

---

## 📈 Model Performance & Evaluation

### 1. Wisconsin Original Dataset (9 Features) — Model Comparison
| Model | Accuracy (%) | Cancer Recall (%) | Cancer Precision (%) | F1-Score (%) | Missed Cancer (FN) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 95.00% | 91.67% | 93.62% | 92.63% | 4 | **0.9950** |
| **SVM (RBF Kernel)** | **95.71%** | **93.75%** | **93.75%** | **93.75%** | **3** | 0.9896 |
| **Random Forest** | 95.00% | 93.75% | 91.84% | 92.78% | 3 | 0.9882 |

> **Key Finding:** Default 0.50 threshold par **SVM (RBF)** ne sabse kam False Negatives (3 missed cases) aur highest accuracy (95.71%) di. Wahin ranking capability me **Logistic Regression** ne highest ROC-AUC score (0.9950) show kiya.

---

### 2. Feature Selection & Ablation Study
SelectKBest (ANOVA F-test) aur Random Forest feature importance ke aadhar par Top 5 features chune gaye:
1. `Cell_Size`
2. `Cell_Shape`
3. `Bare_Nuclei`
4. `Chromatin`
5. `Nucleoli`

| Experiment | Features Used | Accuracy (%) | Cancer Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **All Features** | 9 Features | 95.71% | 93.75% | 93.75% |
| **Optimal Subset** | **Top 5 Features** | **95.71%** | **93.75%** | **93.75%** |
| **Reduced Subset (Shape Removed)** | 4 Features | 94.29% | 91.67% | 92.63% |

> **Takeaway:** 9 features se ghata kar sirf **Top 5 features** lene par accuracy aur recall me koi drop nahi aaya. Par `Cell_Shape` hatane par 1 extra cancer patient miss hua, jo prove karta hai ki remaining variance clinical decision ke liye crucial hai.

---

### 3. WDBC Dataset (30 Features) Evaluation
WDBC dataset par SVM (RBF Kernel) ka result:
- **Test Accuracy:** **97.37%**
- **Confusion Matrix:**
  $$\begin{bmatrix} 72 & 0 \\ 3 & 39 \end{bmatrix}$$
- **Zero False Positives:** Model ne ek bhi swasth vyakti ko galat cancer flag nahi kiya (100% Precision).
- **Top 5 Most Important Measurements (RF Importance):**
  1. `worst area` (15.14%)
  2. `worst concave points` (12.65%)
  3. `worst radius` (9.35%)
  4. `worst perimeter` (8.36%)
  5. `mean concave points` (8.11%)

---

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (`LogisticRegression`, `SVC`, `RandomForestClassifier`, `SelectKBest`, `StandardScaler`, `train_test_split`)
- **Metrics & Evaluation:** `confusion_matrix`, `classification_report`, `roc_curve`, `roc_auc_score`
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Google Colab / Jupyter Notebook

---

## 🚀 How to Run Locally

1. **Repository Clone Karein:**
   ```bash
   git clone [https://github.com/ayushranjan1808/breast-cancer-diagonisis-ml-model.git](https://github.com/ayushranjan1808/breast-cancer-diagonisis-ml-model.git)
   cd breast-cancer-diagonisis-ml-model
