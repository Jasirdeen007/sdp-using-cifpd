# Software Defect Prediction Using CIFPD

## Abstract

Software defect prediction helps identify issue reports that are likely to correspond to defective software behavior. This project builds a defect prediction system using issue-report data and a hybrid modeling pipeline that combines transformer-based language representations with machine learning classification.

The approach uses `RoBERTa` to convert issue text and selected report attributes into dense semantic embeddings, then applies `XGBoost` to classify each issue as defective or non-defective. The project also includes evaluation, saved training artifacts, and a Streamlit-based user interface for interactive prediction.

## Objective

The main objective of this project is to predict whether a software issue is defective based on the information available in issue reports. The system is designed to support:

- automated defect prediction from issue data
- repeatable model training and evaluation
- interactive prediction through a user interface
- export of prediction history for a user session

## Project Workflow

The end-to-end workflow of the project is:

1. Load the issue dataset from CSV.
2. Clean required fields and prepare the dataset.
3. Create the target label `defective` from the resolution status.
4. Rank candidate feature groups and select the most useful report attributes.
5. Combine selected fields into a single text representation called `intent_text`.
6. Generate text embeddings using `roberta-base`.
7. Train an `XGBoost` classifier on the generated embeddings.
8. Evaluate the trained model using holdout metrics and cross-validation.
9. Save trained artifacts, metadata, and evaluation reports.
10. Serve predictions through a Streamlit web interface.

## Methodology

### 1. Data Preparation

The dataset is read from a CSV file. Required columns are validated, missing values are handled, and duplicate records are removed where necessary.

### 2. Target Construction

The project creates the target variable `defective` internally from the issue resolution status:

- `FIXED` -> `1`
- all other values -> `0`

### 3. Feature Selection

Feature ranking is performed using multiple statistical and model-based techniques, including:

- Mutual Information
- Chi-Square
- Cramer's V
- Random Forest feature importance

The selected top columns are used to construct the final textual representation for modeling.

### 4. Embedding Generation

The selected fields are combined into `intent_text`, which is then passed to `RoBERTa`. The transformer model generates vector embeddings that capture the semantic meaning of the issue report.

### 5. Classification

The generated embeddings are used as input to `XGBoost`, which performs the final classification task.

### 6. Evaluation

The model is evaluated using standard classification metrics, including:

- Accuracy
- Precision
- Recall
- F1-score
- Balanced Accuracy
- ROC-AUC
- Log Loss
- Brier Score
- Confusion Matrix

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Transformers
- PyTorch
- XGBoost
- Matplotlib
- Seaborn
- Streamlit

## Project Structure

```text
.
|-- app.py
|-- check_gpu.py
|-- evaluate_model.py
|-- requirements.txt
|-- train_model.py
|-- README.md
|-- SDP_using_CIFPD.ipynb
|-- SDP_using_CIFPD_modified.ipynb
|-- src/
|   |-- __init__.py
|   `-- sdp_pipeline.py
`-- tests/
    `-- test_sdp_pipeline.py
```

## Input Requirements

The dataset should be available as a CSV file and must include at least the following columns:

- `rs`
- `sd`

Additional issue attributes may also be used by the feature selection and training pipeline.

## Installation

### Create and activate a virtual environment

On Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

### Install CUDA-enabled PyTorch

```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Install project dependencies

```powershell
pip install -r requirements.txt
```

### Verify GPU setup

```powershell
python check_gpu.py
```

## Training

Run model training from the project root:

```powershell
python train_model.py --csv Eclipse.csv --artifacts models --reports reports
```

## Evaluation

To print the saved evaluation summary:

```powershell
python evaluate_model.py --artifacts models
```

## Running the Application

Start the Streamlit user interface:

```powershell
streamlit run app.py --server.fileWatcherType none
```

The application allows the user to:

- enter issue report details
- generate a defect prediction
- view the prediction probability
- review session prediction history
- export session predictions as CSV

## Output Artifacts

After training, the following outputs are generated:

### Model artifacts

- `models/classifier.joblib`
- `models/metadata.json`

### Evaluation reports

- `reports/confusion_matrix.png`
- `reports/roc_curve.png`
- feature-ranking CSV files

## Testing

Run unit tests using:

```powershell
python -m pytest -q
```

## Applications

This project can be useful for:

- defect triage support
- software quality analysis
- issue prioritization workflows
- research in software analytics and defect prediction

## Future Scope

- improve repeated UI interaction stability
- expand field labeling with domain-specific names
- add threshold calibration and explainability support
- validate on additional datasets and cross-project scenarios
- deploy as an API-backed application
