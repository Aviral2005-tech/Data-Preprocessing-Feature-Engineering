# Data Preprocessing & Feature Engineering

Customer Churn Prediction 

A complete end-to-end **Data Preprocessing, Feature Engineering, Feature Selection, and Machine Learning Pipeline** for customer churn prediction using Python and Scikit-learn.


The project transforms raw customer data into a clean, machine-learning-ready dataset through systematic exploratory analysis, categorical encoding, feature scaling, outlier detection and handling, feature engineering, feature selection, and model development.


---


## 📌 Project Overview


Customer churn prediction is a classification problem in which the objective is to identify customers who are likely to discontinue a service.


This project demonstrates how raw customer data can be systematically prepared for machine learning.


The complete workflow covers:


- Exploratory Data Analysis (EDA)
- Data quality analysis
- Missing-value analysis
- Duplicate detection
- Categorical encoding
- Numerical feature scaling
- Outlier detection
- Outlier handling
- Feature engineering
- Feature selection
- Train-test splitting
- Machine learning preprocessing pipeline
- Logistic Regression classification
- Model evaluation
- Output and artifact generation


The project is implemented primarily in a **Jupyter Notebook** so that every preprocessing and machine-learning step can be inspected and reproduced interactively.


---


# 🎯 Objective


The primary objective is to build a robust preprocessing and machine-learning workflow capable of preparing customer data for **churn prediction**.


The project aims to:


1. Understand the structure and quality of customer data.
2. Identify missing and duplicate records.
3. Analyze numerical and categorical variables.
4. Transform categorical variables into machine-learning-compatible representations.
5. Scale numerical features.
6. Detect and handle numerical outliers.
7. Create meaningful customer-level features.
8. Identify potentially important predictive features.
9. Build a reproducible preprocessing pipeline.
10. Train a Logistic Regression classification model.
11. Evaluate the model using multiple classification metrics.
12. Save the resulting model pipeline and analytical outputs.


---


# 💼 Business Problem


Customer churn can have a significant impact on businesses that operate using subscription or recurring-revenue models.


When customers leave, organizations may experience:


- Loss of recurring revenue
- Increased customer acquisition costs
- Reduced customer lifetime value
- Lower customer retention
- Increased marketing expenditure


A churn prediction system can help organizations identify customers who may be at higher risk of leaving.


The insights generated from such a system can support:


- Customer retention strategies
- Targeted promotional campaigns
- Personalized customer engagement
- Customer experience improvement
- Revenue protection
- Data-driven business decisions


> **Note:** This project demonstrates the technical machine-learning workflow. The generated predictions should be treated as analytical predictions rather than guaranteed customer behavior.


---


# 📊 Dataset


The project uses a structured customer churn dataset:


```text
customer_churn.csv

The dataset used during development contains 500 customer records and 9 original columns.

| Column             | Description                                    | Type             |
| ------------------ | ---------------------------------------------- | ---------------- |
| `CustomerID`       | Unique customer identifier                     | Categorical      |
| `Tenure`           | Customer tenure                                | Numerical        |
| `MonthlyCharges`   | Monthly customer charges                       | Numerical        |
| `TotalCharges`     | Total customer charges                         | Numerical        |
| `Contract`         | Customer contract type                         | Categorical      |
| `PaymentMethod`    | Customer payment method                        | Categorical      |
| `PaperlessBilling` | Indicates whether paperless billing is enabled | Categorical      |
| `SeniorCitizen`    | Indicates senior-customer status               | Numerical/Binary |
| `Churn`            | Target variable indicating churn status        | Binary           |

Target Variable
Churn

The target is a binary classification variable:

0 → Customer did not churn
1 → Customer churned

The dataset used during the current development run contains an observed churn rate of approximately:

10.60%
🏗️ Project Architecture

The project follows a seven-stage preprocessing and machine-learning workflow.

                    ┌──────────────────────┐
                    │   Raw Customer Data  │
                    │ customer_churn.csv   │
                    └──────────┬───────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │ Part 1: Data Exploration   │
                 │ EDA & Data Quality         │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │ Part 2: Categorical       │
                 │ Encoding                  │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │ Part 3: Feature Scaling   │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │ Part 4: Outlier Detection │
                 │ & Handling                │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │ Part 5: Feature            │
                 │ Engineering               │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │ Part 6: Feature Selection │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │ Part 7: Complete ML       │
                 │ Preprocessing Pipeline     │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │ Logistic Regression Model │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │ Model Evaluation          │
                 │ Accuracy / Precision      │
                 │ Recall / F1 / ROC-AUC     │
                 └───────────────────────────┘
🔄 Complete Project Workflow
Part 1 — Explore & Understand

The first stage focuses on understanding the raw dataset and identifying potential data-quality problems.

Operations
Load CSV dataset
Inspect first records
Determine dataset dimensions
Inspect column names
Analyze data types
Generate dataset information
Calculate missing values
Detect duplicate records
Calculate unique values
Generate numerical statistics
Analyze categorical variables
Analyze churn distribution
Generate initial visualizations
Main Outputs
outputs/
├── missing_values.csv
├── duplicate_analysis.csv
├── unique_values.csv
├── numerical_statistics.csv
├── categorical_summary.csv
└── churn_distribution.csv
Part 2 — Categorical Encoding

Machine-learning algorithms generally require categorical variables to be represented numerically.

This project demonstrates three encoding techniques.

2.1 Label Encoding

Label Encoding assigns an integer representation to categorical values.

Example:

Category A → 0
Category B → 1
Category C → 2

Implementation:

LabelEncoder()

Output:

outputs/label_encoded_data.csv
2.2 One-Hot Encoding

One-Hot Encoding creates separate binary features for each category.

For example:

Contract = One year
Contract = Two year
Contract = Month-to-month

can be represented using separate binary columns.

Implementation:

OneHotEncoder()

Output:

outputs/one_hot_encoded_data.csv
2.3 Ordinal Encoding

Ordinal Encoding converts categories into numerical representations.

Implementation:

OrdinalEncoder()

Output:

outputs/ordinal_encoded_data.csv

The three techniques are demonstrated as part of the preprocessing study. The final machine-learning pipeline uses One-Hot Encoding for categorical features.

Part 3 — Feature Scaling

Numerical variables may have significantly different ranges.

For example:

Tenure          → relatively small values
MonthlyCharges  → larger values
TotalCharges    → potentially much larger values

Scaling helps place numerical variables on comparable scales.

3.1 Standard Scaling

Standardization transforms numerical values based on their mean and standard deviation.

Implementation:

StandardScaler()

Output:

outputs/standard_scaled_data.csv
3.2 Min-Max Scaling

Min-Max scaling transforms values into a normalized range.

Implementation:

MinMaxScaler()

Output:

outputs/minmax_scaled_data.csv

Both techniques are demonstrated independently. The final Logistic Regression pipeline uses StandardScaler.

Part 4 — Outlier Detection & Handling

Outliers are observations that differ substantially from the general distribution of a numerical variable.

This project uses two statistical approaches.

4.1 IQR Method

The Interquartile Range method is based on:

IQR = Q3 - Q1

Lower boundary:

Q1 - 1.5 × IQR

Upper boundary:

Q3 + 1.5 × IQR

Observations outside these boundaries are considered potential outliers.

Output:

outputs/iqr_outlier_analysis.csv
4.2 Z-Score Method

The Z-score measures how far an observation is from the mean in terms of standard deviations.

The project uses:

|Z| > 3

as the threshold for potential outliers.

Output:

outputs/zscore_outlier_analysis.csv
4.3 Outlier Method Comparison

The results from the IQR and Z-score approaches are compared.

Output:

outputs/outlier_method_comparison.csv
4.4 Outlier Handling

Detected numerical outliers can be handled through IQR-based boundary capping rather than automatically deleting observations.

This approach limits extreme values while preserving the original records.

Output:

outputs/outlier_handled_data.csv

The final machine-learning pipeline applies outlier handling in a pipeline-safe manner so that the transformation is learned from the training data and then consistently applied to unseen data.

Part 5 — Feature Engineering

Feature engineering creates additional variables from existing information to provide alternative representations of customer behavior.

The current dataset supports the following engineered features.

5.1 Customer Lifetime Value

A simple lifetime-value representation is calculated using:

CustomerLifetimeValue =
MonthlyCharges × Tenure

This provides an approximate measure of the customer's accumulated monthly-charge value over their tenure.

5.2 Average Monthly Spend

Average monthly spending is derived from total charges and tenure.

Conceptually:

AverageMonthlySpend =
TotalCharges / Tenure

Appropriate handling is applied for zero-tenure cases.

5.3 Tenure Group

Customers are grouped according to their tenure.

Possible groups include:

New Customer
Short Term
Medium Term
Long Term
Loyal Customer

The exact boundaries are defined by the preprocessing implementation.

5.4 Monthly Charge Category

Customers are categorized according to their monthly charges.

Possible categories include:

Low Charge
Medium Charge
High Charge
5.5 Senior Customer

A derived categorical representation of senior-customer status is created from:

SeniorCitizen

The resulting feature is:

SeniorCustomer
Engineered Feature Summary

The current implementation creates:

CustomerLifetimeValue
AverageMonthlySpend
TenureGroup
MonthlyChargeCategory
SeniorCustomer

After removing CustomerID, the feature-engineered dataset contains the original predictive variables plus these derived features.

Output:

outputs/feature_engineered_data.csv
outputs/feature_engineering_summary.csv
Part 6 — Feature Selection

Feature selection identifies variables that may contribute to churn prediction.

The project uses two complementary approaches.

6.1 Correlation Analysis

A correlation matrix is calculated for numerical features.

The relationship between numerical features and the churn target is also analyzed.

Generated outputs:

outputs/feature_correlation_matrix.csv
outputs/feature_correlation_with_churn.csv
outputs/high_correlation_features.csv
6.2 Random Forest Feature Importance

A Random Forest classifier is used to estimate the relative importance of available features.

The importance scores provide an indication of which variables contribute more strongly to the model's predictive process.

Generated outputs:

outputs/feature_importance.csv
outputs/top_10_features.csv
outputs/selected_features.csv

A dataset containing the selected features is also generated:

outputs/selected_feature_dataset.csv

Feature importance is used as an analytical feature-selection technique. It does not by itself establish causation between a feature and customer churn.

Part 7 — Complete Machine Learning Pipeline

The final stage combines preprocessing and model training into a reproducible Scikit-learn pipeline.

The pipeline uses:

Pipeline()

and:

ColumnTransformer()
Numerical Processing

The numerical branch contains:

Numerical Features
        │
        ▼
Missing Value Imputation
        │
        ▼
IQR-Based Outlier Handling
        │
        ▼
Standard Scaling

Current numerical features include:

Tenure
MonthlyCharges
TotalCharges
SeniorCitizen
CustomerLifetimeValue
AverageMonthlySpend
Categorical Processing

The categorical branch contains:

Categorical Features
        │
        ▼
Missing Value Imputation
        │
        ▼
One-Hot Encoding

Current categorical features include:

Contract
PaymentMethod
PaperlessBilling
TenureGroup
MonthlyChargeCategory
SeniorCustomer
🤖 Machine Learning Model

The final classification model is:

LogisticRegression()

The implementation uses class balancing to address the relatively low proportion of churn observations:

LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    random_state=42
)

The complete architecture is therefore:

Raw Features
     │
     ▼
Train-Test Split
     │
     ▼
ColumnTransformer
     │
     ├────────────── Numerical ──────────────┐
     │                                       │
     │  Imputation                           │
     │  Outlier Handling                     │
     │  Standard Scaling                     │
     │                                       │
     └────────────── Categorical ────────────┤
                                             │
        Imputation                           │
        One-Hot Encoding                     │
                                             ▼
                                  Logistic Regression
                                             │
                                             ▼
                                      Predictions
                                             │
                                             ▼
                                      Evaluation
✂️ Train-Test Split

The dataset is divided into:

80% → Training Set
20% → Testing Set

For the current 500-row dataset:

Training samples → 400
Testing samples  → 100

A fixed random state is used to ensure reproducibility.

The test set is kept separate from model training and preprocessing fitting.

📈 Model Evaluation

The final model is evaluated using multiple classification metrics.

Accuracy

Measures the proportion of correctly classified observations.

Accuracy =
Correct Predictions / Total Predictions
Precision

Measures how many customers predicted as churners actually belong to the churn class.

High precision means fewer false-positive churn predictions.

Recall

Measures how many actual churn customers are correctly identified.

Recall is particularly important in churn prediction because failing to identify a customer at risk of leaving can result in a missed retention opportunity.

F1 Score

The F1 score combines precision and recall into a single metric.

It is useful when the classes are imbalanced.

ROC-AUC

ROC-AUC evaluates the model's ability to distinguish between churn and non-churn customers across classification thresholds.

📁 Generated Outputs

The project generates structured CSV outputs inside:

outputs/
Data Exploration
missing_values.csv
duplicate_analysis.csv
unique_values.csv
numerical_statistics.csv
categorical_summary.csv
churn_distribution.csv
Categorical Encoding
label_encoded_data.csv
one_hot_encoded_data.csv
ordinal_encoded_data.csv
Feature Scaling
standard_scaled_data.csv
minmax_scaled_data.csv
Outlier Analysis
iqr_outlier_analysis.csv
zscore_outlier_analysis.csv
outlier_method_comparison.csv
outlier_handled_data.csv
Feature Engineering
feature_engineered_data.csv
feature_engineering_summary.csv
Feature Selection
feature_correlation_matrix.csv
feature_correlation_with_churn.csv
high_correlation_features.csv
feature_importance.csv
top_10_features.csv
selected_features.csv
selected_feature_dataset.csv
Machine Learning
pipeline_model_performance.csv
pipeline_confusion_matrix.csv
pipeline_prediction_results.csv
customer_churn_preprocessing_pipeline.pkl
📊 Visualizations

Visualizations are stored in:

visualizations/

The current analysis generates visualizations for:

tenure_distribution.png
monthly_charges_distribution.png
total_charges_distribution.png
contract_distribution.png
churn_distribution.png

These visualizations provide a graphical view of customer distributions and churn behavior.

🛠️ Technology Stack
Programming Language
Python 3.x
Data Processing
Pandas
NumPy
Visualization
Matplotlib
Statistical Analysis
SciPy
Machine Learning
Scikit-learn
Model Persistence
Joblib
Development Environment
Jupyter Notebook
VS Code
📦 Main Python Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from scipy.stats import zscore


from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler
)


from sklearn.impute import SimpleImputer


from sklearn.compose import ColumnTransformer


from sklearn.pipeline import Pipeline


from sklearn.ensemble import RandomForestClassifier


from sklearn.linear_model import LogisticRegression


from sklearn.model_selection import train_test_split


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


import joblib
📂 Project Structure
Data Preprocessing & Feature Engineering/
│
├── churn_prediction_pipeline.ipynb
├── customer_churn.csv
│
├── outputs/
│   ├── missing_values.csv
│   ├── duplicate_analysis.csv
│   ├── unique_values.csv
│   ├── numerical_statistics.csv
│   ├── categorical_summary.csv
│   ├── churn_distribution.csv
│   │
│   ├── label_encoded_data.csv
│   ├── one_hot_encoded_data.csv
│   ├── ordinal_encoded_data.csv
│   │
│   ├── standard_scaled_data.csv
│   ├── minmax_scaled_data.csv
│   │
│   ├── iqr_outlier_analysis.csv
│   ├── zscore_outlier_analysis.csv
│   ├── outlier_method_comparison.csv
│   ├── outlier_handled_data.csv
│   │
│   ├── feature_engineered_data.csv
│   ├── feature_engineering_summary.csv
│   │
│   ├── feature_correlation_matrix.csv
│   ├── feature_correlation_with_churn.csv
│   ├── high_correlation_features.csv
│   ├── feature_importance.csv
│   ├── top_10_features.csv
│   ├── selected_features.csv
│   ├── selected_feature_dataset.csv
│   │
│   ├── pipeline_model_performance.csv
│   ├── pipeline_confusion_matrix.csv
│   ├── pipeline_prediction_results.csv
│   └── customer_churn_preprocessing_pipeline.pkl
│
├── visualizations/
│   ├── tenure_distribution.png
│   ├── monthly_charges_distribution.png
│   ├── total_charges_distribution.png
│   ├── contract_distribution.png
│   └── churn_distribution.png
│
├── preprocessing_report.md
├── feature_engineering_documentation.md
├── requirements.txt
└── README.md
⚙️ Installation
1. Install Python

Make sure Python 3.x is installed.

Verify the installation:

python --version
2. Create a Virtual Environment

Windows:

python -m venv venv

Activate it:

venv\Scripts\activate
3. Install Dependencies

Install the required libraries:

pip install pandas numpy matplotlib scipy scikit-learn joblib jupyter

Alternatively:

pip install -r requirements.txt
▶️ Running the Project
Option 1 — Jupyter Notebook

Start Jupyter:

jupyter notebook

Open:

churn_prediction_pipeline.ipynb

Run the notebook sequentially:

Cell 1
  ↓
Cell 2
  ↓
Cell 3
  ↓
...
  ↓
Final Cell

The notebook is designed to work without relying on Python's __file__ variable, making it suitable for Jupyter execution.

Option 2 — VS Code

Open the project folder in VS Code.

Select the project's virtual environment as the Python interpreter.

Open:

churn_prediction_pipeline.ipynb

Select the appropriate Jupyter kernel and execute all cells.

📌 Dataset Location

The dataset should be located in the same project directory as the notebook:

Data Preprocessing & Feature Engineering/
│
├── churn_prediction_pipeline.ipynb
└── customer_churn.csv

The notebook uses the current working environment to locate the dataset rather than relying on __file__.

🔁 Reproducibility

The project uses fixed random states where appropriate.

For example:

random_state=42

This helps produce reproducible train-test splits and machine-learning results.

The preprocessing pipeline is also saved using Joblib:

customer_churn_preprocessing_pipeline.pkl

This allows the trained preprocessing and model workflow to be reused later.

🧪 Testing & Validation

The project validates the major stages of the workflow.

| Component              | Expected Result                          |
| ---------------------- | ---------------------------------------- |
| Dataset Loading        | Dataset loads successfully               |
| Dataset Exploration    | Shape, columns and data types identified |
| Missing Value Analysis | Missing values analyzed                  |
| Duplicate Analysis     | Duplicate records analyzed               |
| Categorical Analysis   | Categorical variables summarized         |
| Label Encoding         | Encoded dataset generated                |
| One-Hot Encoding       | Encoded dataset generated                |
| Ordinal Encoding       | Encoded dataset generated                |
| Standard Scaling       | Standardized dataset generated           |
| Min-Max Scaling        | Normalized dataset generated             |
| IQR Detection          | Potential outliers identified            |
| Z-Score Detection      | Potential outliers identified            |
| Outlier Handling       | Capped dataset generated                 |
| Feature Engineering    | Derived features generated               |
| Correlation Analysis   | Correlation outputs generated            |
| Feature Importance     | Feature importance calculated            |
| Feature Selection      | Selected-feature dataset generated       |
| Train-Test Split       | Training and testing datasets created    |
| Pipeline Construction  | Complete preprocessing pipeline created  |
| Model Training         | Logistic Regression trained              |
| Model Evaluation       | Classification metrics calculated        |
| Model Persistence      | Pipeline saved using Joblib              |
⚠️ Important Implementation Notes
Customer ID

CustomerID is an identifier rather than a meaningful predictive feature.

Therefore, it is removed before the final model-training process.

Data Leakage Prevention

The final machine-learning workflow uses a train-test split before fitting preprocessing transformations.

Preprocessing operations inside the final pipeline are learned from the training data and then applied to the testing data.

This helps prevent information from the test set from influencing model training.

Categorical Variables

The final pipeline uses:

OneHotEncoder(handle_unknown="ignore")

This allows the pipeline to handle previously unseen categorical values during prediction.

Missing Values

Numerical variables use:

SimpleImputer(strategy="median")

Categorical variables use:

SimpleImputer(strategy="most_frequent")
Class Imbalance

The current dataset contains substantially fewer churn observations than non-churn observations.

The Logistic Regression classifier therefore uses:

class_weight="balanced"

to give greater consideration to the minority class during training.

📚 Key Learning Outcomes

This project provides practical experience with:

Data Preprocessing
Dataset loading
Data inspection
Data-quality analysis
Missing-value handling
Duplicate detection
Categorical Data
Label Encoding
One-Hot Encoding
Ordinal Encoding
Numerical Data
Standardization
Min-Max normalization
Statistical analysis
Outlier Analysis
IQR method
Z-score method
IQR-based outlier capping
Feature Engineering
Derived numerical features
Customer segmentation
Tenure grouping
Charge categorization
Feature Selection
Correlation analysis
Random Forest feature importance
Top-feature selection
Machine Learning
Train-test splitting
ColumnTransformer
Pipeline construction
Logistic Regression
Classification metrics
Model Evaluation
Accuracy
Precision
Recall
F1 Score
ROC-AUC
Confusion Matrix
MLOps-Oriented Practices
Reproducible preprocessing
Pipeline persistence
Structured output generation
Separation of training and testing data
🚀 Future Enhancements

The project can be extended in several directions.

Advanced Machine Learning
Random Forest
XGBoost
Gradient Boosting
Support Vector Machine
Decision Tree
Neural Networks
Model Optimization
Hyperparameter tuning
GridSearchCV
RandomizedSearchCV
Cross-validation
Threshold optimization
Imbalanced Dataset Techniques
SMOTE
Random oversampling
Random undersampling
Precision-recall analysis
Advanced Evaluation
ROC curve
Precision-Recall curve
Calibration curve
Learning curves
Cross-validation performance
Explainable AI

Future versions could integrate:

SHAP
LIME
Feature contribution analysis
Individual prediction explanations

This would make it easier to understand why a particular customer was classified as high-risk.

Interactive Application

The trained pipeline can later be integrated into:

Streamlit

or another web framework to create an interactive churn prediction application.

Potential functionality:

Customer Information
        ↓
Prediction
        ↓
Churn Probability
        ↓
Risk Category
        ↓
Retention Recommendation
Business Intelligence

The project can also be extended with:

Power BI dashboard
Tableau dashboard
SQL database integration
Customer segmentation
Retention analytics
Revenue-at-risk analysis
Deployment

Possible deployment technologies include:

FastAPI
Flask
Streamlit
Docker
Cloud deployment

A production version could expose the trained model through an API for real-time customer churn prediction.

📌 Limitations

The current project has several limitations.

The dataset is relatively small.
The available features represent only a limited set of customer characteristics.
Model performance depends heavily on the quality and representativeness of the dataset.
Logistic Regression assumes a relatively simple relationship between features and the target.
Feature importance does not imply causation.
A predicted churn probability should not be interpreted as certainty.
Additional real-world customer behavior variables could improve predictive performance.
🔐 Responsible Use

Customer churn predictions should be used as decision-support information rather than as an automatic basis for customer treatment.

When deploying such a system in a real business environment, organizations should consider:

Data privacy
Fairness
Bias
Explainability
Data security
Human oversight
Appropriate use of customer information
📄 Project Deliverables

The main submission files are:

churn_prediction_pipeline.ipynb
customer_churn.csv
preprocessing_report.md
feature_engineering_documentation.md
requirements.txt
README.md

Generated outputs and visualizations are stored separately in:

outputs/
visualizations/
🎓 Academic Context

This project demonstrates the practical application of concepts from:

Data Preprocessing
Exploratory Data Analysis
Feature Engineering
Feature Selection
Statistical Analysis
Machine Learning
Classification
Model Evaluation

It is designed as an academic and portfolio project demonstrating an end-to-end machine-learning preprocessing workflow.

🏁 Conclusion

The Customer Churn Prediction — Data Preprocessing & Feature Engineering project demonstrates a complete workflow for transforming raw customer data into a machine-learning-ready representation.

The project begins with exploratory data analysis and progresses through categorical encoding, feature scaling, outlier analysis, feature engineering, feature selection, and construction of a complete Scikit-learn preprocessing pipeline.

The final pipeline combines numerical and categorical preprocessing with Logistic Regression and evaluates the resulting model using multiple classification metrics.

The project also emphasizes reproducibility by using a fixed random state, separating training and testing data, and saving the complete preprocessing and model pipeline.

Overall, the project provides a strong foundation for developing more advanced customer churn prediction systems and can be extended with ensemble models, explainable AI, interactive dashboards, APIs, and cloud deployment.
