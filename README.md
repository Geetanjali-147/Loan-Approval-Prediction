# Loan Approval Prediction

A machine learning classification project that predicts whether a loan application will be **approved** or **rejected** based on the applicant's personal, financial and property details.

The whole project lives in one Jupyter notebook and uses a small built-in dataset, so **no external download is needed**.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Predicting for a New Applicant](#predicting-for-a-new-applicant)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)
- [Troubleshooting](#troubleshooting)
- [Author](#author)

---

## Features

- Built-in dataset of 20 loan applications (11 features + target)
- Exploratory data analysis with seaborn / Matplotlib charts
- Label encoding and standard scaling of features
- Stratified 75% / 25% train-test split
- Four models compared: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- Accuracy, confusion matrix and classification report for every model
- Automatic selection of the best model, with ROC curve and AUC
- Custom prediction for a new applicant

## Project Structure

```
.
├── GeetanjaliAwasthi_LoanApprovalPrediction.ipynb   # Main notebook (all code)
├── requirements.txt                                 # Python dependencies
├── LoanApprovalPrediction_ProjectDocumentation.docx # Full project documentation
└── README.md                                        # This file
```

## Dataset

The dataset is defined inside the notebook as a Python dictionary and follows the structure of the public *Loan Prediction* dataset.

| Column | Type | Description |
|---|---|---|
| Gender | Categorical | Male / Female |
| Married | Categorical | Yes / No |
| Dependents | Categorical | 0, 1, 2, 3+ |
| Education | Categorical | Graduate / Not Graduate |
| Self_Employed | Categorical | Yes / No |
| ApplicantIncome | Numeric | Applicant's income |
| CoapplicantIncome | Numeric | Co-applicant's income |
| LoanAmount | Numeric | Requested loan amount |
| Loan_Amount_Term | Numeric | Loan term in months (360 for all rows) |
| Credit_History | Binary | 1 = meets guidelines, 0 = does not |
| Property_Area | Categorical | Urban / Semiurban / Rural |
| **Loan_Status** | **Target** | **Y = approved, N = rejected** |

Class balance: 12 approved (60%) and 8 rejected (40%).

## Installation

**Requirements:** Python 3.11 or later.

```bash
# 1. (Optional) create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

## Usage

1. Open `GeetanjaliAwasthi_LoanApprovalPrediction.ipynb` in **VS Code**, **Jupyter Notebook** or **Google Colab**.
2. Select the Python environment where you installed the requirements.
3. Run all cells.

The notebook prints the dataset summary, model results and the final prediction, and displays four charts (class distribution, income distribution, model comparison, ROC curve).

## Methodology

1. **Create data**: build the DataFrame from the predefined dictionary.
2. **EDA**: class balance count plot and applicant income histogram.
3. **Preprocessing**: `LabelEncoder` for categorical columns, `StandardScaler` for all features.
4. **Split**: `train_test_split(test_size=0.25, random_state=42, stratify=y)` gives 15 training and 5 test rows.
5. **Modelling**: train and evaluate four classifiers.
6. **Selection**: pick the model with the highest test accuracy.
7. **Evaluation**: ROC curve and AUC for the selected model.
8. **Prediction**: predict the outcome for a custom applicant.

### Label encoding reference

| Column | Mapping |
|---|---|
| Gender | Female = 0, Male = 1 |
| Married | No = 0, Yes = 1 |
| Dependents | 0 = 0, 1 = 1, 2 = 2, 3+ = 3 |
| Education | Graduate = 0, Not Graduate = 1 |
| Self_Employed | No = 0, Yes = 1 |
| Property_Area | Rural = 0, Semiurban = 1, Urban = 2 |
| Loan_Status | N = 0, Y = 1 |

## Results

Test-set accuracy (5 test records):

| Model | Accuracy |
|---|---|
| Logistic Regression | 0.80 |
| Decision Tree | 0.80 |
| Random Forest | 0.60 |
| Gradient Boosting | 0.80 |

Logistic Regression is selected as the best model because it is the first of the three models tied at 80%. Its confusion matrix on the test set:

|  | Predicted Rejected | Predicted Approved |
|---|---|---|
| **Actual Rejected** | 2 | 0 |
| **Actual Approved** | 1 | 2 |

> The dataset is very small, so these numbers illustrate the workflow and are not a reliable measure of real-world performance. The AUC of 1.00 on 5 test records is not meaningful.

## Predicting for a New Applicant

Edit the `sample` DataFrame in the last cell using the encoding reference above:

```python
sample = pd.DataFrame([{
    "Gender": 1,              # Male
    "Married": 1,             # Yes
    "Dependents": 0,
    "Education": 0,           # Graduate
    "Self_Employed": 0,       # No
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 1500,
    "LoanAmount": 120,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": 2        # Urban
}])
```

The default applicant above is predicted as **Approved**.

## Known Limitations

- Only 20 records; a single test prediction changes accuracy by 20 percentage points.
- `StandardScaler` is fitted before the train/test split, which causes data leakage.
- One `LabelEncoder` is reused for all columns, so mappings are not stored for new data.
- `Loan_Amount_Term` is constant (360) and carries no information.
- `cross_val_score` is imported but not used (no cross-validation).
- Sensitive attributes (Gender, Married) would need fairness and legal review in a real lending system.

## Future Improvements

- Use a larger real dataset (several hundred rows or more).
- Fit the scaler and encoders inside a scikit-learn `Pipeline` / `ColumnTransformer`.
- Add cross-validation and hyperparameter tuning (`GridSearchCV`).
- Save the trained model with `joblib` and build a Streamlit or Flask app.
- Add explainability with SHAP.

## Troubleshooting

| Problem | Solution |
|---|---|
| "Running cells requires the ipykernel package" | Run `pip install ipykernel -U`, then reselect the Python environment. |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in the active environment. |
| seaborn `FutureWarning` about `palette` | Harmless. Remove it by passing `hue` and `legend=False` to the plot call. |

## Author

**Geetanjali Awasthi**
