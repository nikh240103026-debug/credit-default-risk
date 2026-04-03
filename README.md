# Credit Default Risk Predictor

A Machine Learning web application that predicts whether a bank customer is likely to default on a loan within the next 2 years. Built as part of a FinTech AI portfolio targeting real-world banking risk analytics.

---

## Problem Statement

Banks face massive financial losses when customers default on loans. Manually assessing each customer's risk is slow and inconsistent. This system uses Machine Learning to automatically predict credit default risk based on customer financial behavior — helping banks make faster, data-driven lending decisions.

---

## Dataset

- **Source:** Give Me Some Credit — Kaggle
- **Size:** 150,000 customer records
- **Target:** SeriousDlqin2yrs (1 = Defaulted, 0 = Did not default)
- **Class Distribution:** 93.3% non-default, 6.7% default (imbalanced)

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Analyzed 150,000 records with 10 features
- Identified and handled missing values in MonthlyIncome (19.8%) and NumberOfDependents (2.6%) using median imputation
- Detected severe class imbalance (93.3% vs 6.7%)

### 2. Data Preprocessing
- Removed irrelevant index column
- Filled missing values with median (chosen over mean due to skewed distributions)
- Train-test split (80/20)

### 3. Handling Class Imbalance
- Applied SMOTE (Synthetic Minority Oversampling Technique) on training data only
- Balanced dataset from 111,930 vs 8,070 to 111,930 vs 111,930
- SMOTE not applied on test data to preserve real-world distribution

### 4. Model Training
- Algorithm: Random Forest Classifier
- 100 estimators
- Trained on SMOTE-balanced training data

### 5. Model Evaluation
- **ROC-AUC Score: 0.8228** (Very Good)
- **Overall Accuracy: 89%**
- ROC curve plotted to visualize model separation ability

#### Classification Report:
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0 (No Default) | 0.96 | 0.92 | 0.94 |
| 1 (Default) | 0.29 | 0.46 | 0.36 |

#### Known Limitation:
Defaulter recall of 0.46 means the model misses 54% of actual defaulters. This is a known challenge with heavily imbalanced financial datasets. Future improvements include XGBoost, threshold tuning, and cost-sensitive learning.

---

## Web Application

Built with Flask. Users input customer financial details and receive instant risk prediction.

**Input Features:**
- Age
- Monthly Income
- Debt Ratio
- Revolving Utilization of Unsecured Lines
- Number of Dependents
- Number of Open Credit Lines
- Number of Real Estate Loans
- Times 30-59 Days Late
- Times 60-89 Days Late
- Times 90+ Days Late

**Output:**
- ✅ Low Risk — Likely to repay loan
- ⚠️ High Risk — Likely to default

---

## Technologies Used

- **Language:** Python 3.11
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (Random Forest)
- **Imbalanced Data:** imbalanced-learn (SMOTE)
- **Web Framework:** Flask

---
## Note that the model file is not included due to GitHub's 100MB file size limit. To use the app, run the Jupyter notebook first to train and save the model, then run app.py. This is standard practice for large ML models in real projects.

## How to Run

### 1. Clone the repository
git clone https://github.com/nikh240103026-debug/credit-default-risk.git
cd credit-default-risk

### 2. Install dependencies
pip install flask scikit-learn pandas numpy imbalanced-learn matplotlib seaborn

### 3. Train the model
Open and run all cells in credit_default.ipynb
This will generate credit_default_model.pkl

### 4. Run the web app
python app.py

### 5. Open in browser
http://127.0.0.1:5000

---

## 📁 Project Structure
Credit Default Risk/
├── app.py                    # Flask web application
├── credit_default.ipynb      # Jupyter notebook (EDA + Model Training)
├── .gitignore                # Excludes large model file
└── README.md                 # Project documentation

---

## 🔮 Future Improvements

- Implement XGBoost for better defaulter recall
- Tune classification threshold for higher sensitivity
- Add cost-sensitive learning to penalize missed defaulters more
- Deploy on cloud (Render/Railway) for public access
- Add more features like credit history length

---

## 👨‍💻 Author

**Nikhil Raj**
B.Tech CSE (AI & Data Science) — IIIT Manipur
Building real-world FinTech AI solutions

GitHub: https://github.com/nikh240103026-debug
