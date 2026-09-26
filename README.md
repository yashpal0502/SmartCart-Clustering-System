# SmartCart Clustering System

## 📌 Project Overview

**SmartCart Clustering System** is a machine learning project that uses
**customer segmentation** to identify groups of customers with similar
demographic, spending, purchasing, and behavioral characteristics.

The project applies unsupervised learning (K-Means and Agglomerative
Clustering) to discover customer segments without a predefined target
variable, then profiles each segment so the results can be turned into
actionable marketing and retention strategies.

The current work covers the **end-to-end machine learning pipeline**:
data cleaning, feature engineering, outlier handling, encoding, scaling,
dimensionality reduction, cluster-count selection, clustering, and
cluster profiling. A web application can be added later to make the
system interactive.

> 📄 For the full analysis — every chart, every number, and the
> reasoning behind each decision — see **[REPORT.md](REPORT.md)**.
> This README stays focused on what the project *is* and how to use it.

---

## 🎯 Objectives

- Understand the structure of the customer dataset.
- Clean and preprocess customer data.
- Engineer meaningful customer-level features.
- Handle missing values and outliers.
- Encode categorical variables.
- Scale numerical features for distance-based clustering.
- Use PCA to reduce dimensionality and visualize customer data.
- Analyze the appropriate number of clusters (Elbow Method + Silhouette Score).
- Apply K-Means and Agglomerative Clustering.
- Profile and characterize the resulting customer segments.

---

## 📊 Dataset

The dataset contains customer information covering:

### Demographics
- `ID`, `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`,
  `Teenhome`, `Dt_Customer`

### Spending
- `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`,
  `MntSweetProducts`, `MntGoldProds`

### Purchase Frequency / Channel
- `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`,
  `NumStorePurchases`, `NumWebVisitsMonth`

### Other Customer Information
- `Recency` — days since the customer's last purchase
- `Complain` — whether the customer complained in the last 2 years (0/1)
- `Response` — whether the customer accepted the offer in the last
  marketing campaign (0/1) — used later for cluster profiling

The raw dataset contains **2,240 customer records** across **22
columns**, with **24 missing values in `Income`**. After feature
engineering, outlier removal, and encoding, the final analysis-ready
matrix contains **2,236 records** and **18 features**.

---

## 🛠️ Technologies and Libraries

- Python, Jupyter Notebook
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn (`OneHotEncoder`, `StandardScaler`, `PCA`, `KMeans`,
  `AgglomerativeClustering`, `silhouette_score`)
- Kneed (`KneeLocator`, for automated elbow detection)

---

## 📁 Project Structure

```text
smartcart-clustering-system/
│
├── data/
│   ├── raw/                 # smartcart_customers.csv
│   
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   
│
├── visualizations/
│   ├── outliers/
│   ├── correlations/
│   └── clustering/
│
├── README.md
├── REPORT.md                # ← full analysis write-up
└── .gitignore
```

---

## 🔄 Machine Learning Workflow

```text
Raw Customer Data
       ↓
Data Understanding
       ↓
Missing Value Handling  (Income → median imputation)
       ↓
Feature Engineering     (Age, Tenure, Total_Spendings, Total_Children,
                          Education groups, Living_With)
       ↓
Feature Removal         (drop IDs, redundant/raw columns)
       ↓
Outlier Handling        (Age ≥ 90, Income ≥ 600,000)
       ↓
Correlation Analysis
       ↓
Categorical Encoding    (One-Hot Encoding)
       ↓
Feature Scaling         (StandardScaler)
       ↓
PCA (3 components)
       ↓
K Analysis              (Elbow Method + Silhouette Score)
       ↓
K-Means Clustering  (K = 4)
       ↓
Agglomerative Clustering (K = 4, Ward linkage)
       ↓
Cluster Characterization
       ↓
Customer Profiling
```

⚠️ **Important implementation detail:** in the current notebook, both
K-Means and Agglomerative Clustering (and the Elbow/Silhouette
analysis) are fit on the **3-component PCA output (`X_pca`)**, not on
the full 18-feature scaled matrix (`X_scaled`). PCA is therefore doing
double duty here — dimensionality reduction *and* the clustering
feature space — rather than being used purely for visualization. See
**REPORT.md → Limitations** for why this matters and how to change it.

---

## 🧹 Data Preprocessing (Summary)

**Missing values** in `Income` (24 rows) are filled with the median.

**Engineered features:** `Age` (current year − `Year_Birth`),
`Customer_Tenure_Days` (days since the most recent enrollment date in
the dataset), `Total_Spendings` (sum of the six `Mnt*` columns), and
`Total_Children` (`Kidhome` + `Teenhome`).

**Simplified categories:** `Education` → `Undergraduate` / `Graduate` /
`Postgraduate`; `Marital_Status` → `Living_With` (`Partner` / `Alone`).

**Columns dropped** after their engineered replacements are created:
`ID`, `Year_Birth`, `Marital_Status`, `Kidhome`, `Teenhome`,
`Dt_Customer`, and the six individual `Mnt*` spending columns.

**Outliers removed:** 3 customers with `Age ≥ 90` and 1 customer with
`Income ≥ 600,000` (2,240 → 2,236 records).

**Encoding & scaling:** `Education` and `Living_With` are one-hot
encoded; all features are then standardized with `StandardScaler`.

Full details, code, and the reasoning behind each step are in
**REPORT.md**.

---

## 📈 Results at a Glance

Using `K = 4` (selected via the Kneedle elbow method on WCSS), four
customer segments emerge:

| Cluster | Color (from cluster-profile sketches) | Income | Spending | Household | Snapshot |
|---|---|---|---|---|---|
| 0 | Red | Low–moderate | Low–moderate | Partnered, more children | Frequent site visits, low purchase volume, weakest campaign response |
| 1 | Blue | High | High | Partnered, fewer children, older | Heavy store/catalog/web buyers, most postgraduate-educated |
| 2 | Yellow | Low | Low | Living alone, most children | Highest web visits but lowest purchases, highest complaint rate |
| 3 | Green | Moderate–high | High | Living alone, fewer children | Best campaign response by far, longest tenure, heavy multi-channel buyers |

See **REPORT.md → Cluster Profiling** for the full numeric breakdown,
persona narratives, and business implications.

---

## 📌 Current Project Status

### Completed
- Data understanding, missing-value handling, feature engineering,
  feature selection, outlier handling, correlation analysis, one-hot
  encoding, feature scaling, PCA, Elbow Method, Silhouette Score
  analysis, K-Means clustering, Agglomerative Clustering, initial
  cluster characterization, cluster profiling.

### Next
- Re-check `K` against the Silhouette Score results, not only the
  elbow/Kneedle result.
- Finalize model and cluster-count selection using multiple evaluation
  metrics.
- Save the final preprocessing pipeline and clustering model
  (`models/`) instead of re-fitting in-notebook.
- Build an inference pipeline for assigning new customers to clusters.
- Optional web application for interactive customer segmentation.

---

## 👨‍💻 Project

**SmartCart Clustering System** — Machine Learning Customer
Segmentation Project