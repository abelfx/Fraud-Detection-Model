# Fraud Detection for E-Commerce and Banking

## Project Overview
We are developing a robust fraud detection system for two distinct domains: **E-commerce transactions** and **Bank credit transactions**. 

Fraud detection is a high-stakes challenge involving a delicate trade-off between **Security** (detecting fraud) and **User Experience** (minimizing false positives). This project leverages geolocation analysis, transaction patterns, and advanced ensemble models to protect financial assets and build institutional trust.

---

## Project Structure
The repository is organized following industry best practices for data science workflows:

```
fraud-detection/
├── api.py                    # FastAPI application for fraud detection service
├── dashboard.py              # Streamlit dashboard for visualization and interaction
├── main.py                   # CLI tool for training, preprocessing, evaluation, and prediction
├── requirements.txt          # Project dependencies
├── notebooks/                # Jupyter notebooks for fraud detection pipelines
│   ├── creditcard_fraud_pipeline.ipynb
│   └── ecommerce_fraud_pipeline.ipynb
├── src/                      # Modular source code
│   ├── __init__.py
│   ├── config.py             # Configuration settings
│   ├── data_loader.py        # Data loading utilities
│   ├── feature_engineer.py   # Feature engineering for fraud detection
│   ├── logger.py             # Logging setup
│   ├── model_evaluator.py    # Model evaluation metrics and functions
│   ├── model_trainer.py      # Model training logic
│   ├── predictor.py          # Prediction interface
│   └── preprocessor.py       # Data preprocessing pipeline
└── tests/                    # Unit tests
    ├── __init__.py
    ├── conftest.py
    ├── test_config.py
    ├── test_data_loader.py
    ├── test_feature_engineer.py
    ├── test_model_trainer.py
    ├── test_predictor.py
    └── test_preprocessor.py
```

## Key Technical Challenges
- Class Imbalance: Fraud cases make up < 1% of the data. We utilize SMOTE (Synthetic Minority Over-sampling Technique) to ensure the model learns fraud patterns effectively.
- Geolocation Mapping: Merging billion-row IP ranges with transaction logs using range-based lookups (merge_asof) for country-level insights.
- Explainability: Using SHAP (SHapley Additive exPlanations) to move beyond "black-box" models and provide actionable business recommendations.

## Installation & Setup
1. Clone the Repository:
```
git clone https://github.com/abelfx/Fraud-Detection-Model
cd fraud-detection
```

2. Environment Setup:
```
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Data Preparation:
Place raw data files (e.g., Fraud_Data.csv, IpAddress_to_Country.csv, creditcard.csv) in an appropriate location as configured in `src/config.py`.

## Usage

### CLI (Command Line Interface)
Use `main.py` for training, preprocessing, evaluation, and prediction:

```bash
# Preprocess data
python main.py preprocess --dataset both

# Train models
python main.py train --dataset both

# Evaluate models
python main.py evaluate --dataset fraud

# Make predictions
python main.py predict --dataset creditcard --input data.json
```

### API
Run the FastAPI server for real-time predictions:

```bash
uvicorn api:app --reload
```

Access the API documentation at `http://localhost:8000/docs`.

### Dashboard -- RECOMMENDED or you can use the pipeline notebooks
Launch the Streamlit dashboard for interactive analysis:

```bash
streamlit run dashboard.py
```

## Pipeline Stages
## Task 1: Data Analysis & Preprocessing
- IP Mapping: Converted IP addresses to integers to perform range-based country lookups.

## Feature Engineering:

- time_diff: Duration between signup and purchase (critical for spotting "instant" bot-driven fraud).
- hour_of_day & day_of_week: Temporal patterns in fraudulent behavior.
- Imbalance Handling: Applied SMOTE only to the training set to prevent data leakage.

## Task 2: Model Building & Training
- Baseline Model: Logistic Regression for clear interpretability.
- Ensemble Models: Random Forest and XGBoost for capturing complex, non-linear fraud signatures.
- Evaluation Metrics: Priority given to AUC-PR and F1-Score over simple accuracy.

## Task 3: Model Explainability (XAI)
- Extracting Global Importance to identify overall fraud drivers.
- Generating SHAP Force Plots for individual transaction verification (True Positives vs. False Positives).

## Data sets: Add these inside your Data/raw folder before starting your training
- [Fraud_Data.csv](https://drive.google.com/file/d/115VJ-WTPYeP9Wi_llBxUcygZhbPUE01F/view)
- [IpAddress_to_Country.csv](https://drive.google.com/file/d/1mLyLNs6VTGOltT5zUfFInXw6VDLR-MmQ/view)
- [creditcard.csv](https://drive.google.com/file/d/1UvXXxXtmFFRDU4WI6VjnDoALO1bFfC0P/view)



