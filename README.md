<h1 align="center">
  The Impact of Feature Scaling In Machine Learning
</h1>

<h3 align="center">
  The Impact of Feature Scaling In Machine Learning: Effects on Regression and Classification Tasks
</h3>

<p align="center">
  <strong>João Manoel Herrera Pinheiro</strong><sup>1</sup> &middot;
  <strong>Suzana Vilas Boas de Oliveira</strong><sup>2</sup> &middot;
  <strong>Thiago Henrique Segreto Silva</strong><sup>1</sup> &middot;
  <strong>Pedro Antonio Rabelo Saraiva</strong><sup>1</sup> &middot;
  <strong>Enzo Ferreira de Souza</strong><sup>1</sup><br>
  <strong>Ricardo V. Godoy</strong><sup>1</sup> &middot;
  <strong>Leonardo André Ambrosio</strong><sup>2</sup> &middot;
  <strong>Marcelo Becker</strong><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup> Department of Mechanical Engineering, University of São Paulo, São Paulo 13566-590, Brazil<br>
  <sup>2</sup> Department of Electrical and Computer Engineering, University of São Paulo, São Paulo 13566-590, Brazil<br>
</p>

<p align="center">
  <em>IEEE Access, 2025</em>
</p>

<p align="center">
  <a href="https://doi.org/10.1109/ACCESS.2025.3635541">
    <img src="https://img.shields.io/badge/Paper-PDF-b31b1b?style=flat-square&logo=arxiv&logoColor=white" alt="Paper">
  </a>&nbsp;
  <a href="https://arxiv.org/abs/2506.08274">
    <img src="https://img.shields.io/badge/arXiv-2506.08274-b31b1b?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv">
  </a>&nbsp;
</p>

This research addresses the critical lack of comprehensive studies on feature scaling by systematically evaluating 12 scaling techniques - including several less common transformations - across 14 different Machine Learning algorithms and 16 datasets for classification and regression tasks. We meticulously analyzed impacts on predictive performance (using metrics such as accuracy, MAE, MSE, and $R^2$) and computational costs (training time, inference time, and memory usage). Key findings reveal that while ensemble methods (such as Random Forest and gradient boosting models like XGBoost, CatBoost and LightGBM) demonstrate robust performance largely independent of scaling, other widely used models such as Logistic Regression, SVMs, TabNet, and MLPs show significant performance variations highly dependent on the chosen scaler. This extensive empirical analysis, with all source code, experimental results, and model parameters made publicly available to ensure complete transparency and reproducibility, offers model-specific crucial guidance to practitioners on the need for an optimal selection of feature scaling techniques.

# Machine Learning Pipeline

This repository provides a complete and reproducible machine learning pipeline for training, evaluating, and comparing regression and classification models. The project automates the entire workflow, from dataset import and preprocessing to model training, evaluation, and result generation.

## Repository Structure

```
.
├── model_configs/                  # Model hyperparameter configuration files
├── results/                        # Training outputs, figures and metrics
├── CITATION.cff
├── LICENSE
├── README.md
├── analysis-results.ipynb          # Result analysis notebook
├── dataset_split_classification.csv
├── dataset_split_classification.json
├── dataset_split_regression.csv
├── dataset_split_regression.json
├── etl_cleaning.py                 # Data cleaning and preprocessing
├── import_datasets.py              # Dataset loading utilities
├── main.py                         # Pipeline launcher
├── requirements.txt
├── results_final.csv               # Final consolidated results
├── scalers.py                      # Feature scaling methods
├── tables.tex                      # LaTeX tables generated from results
├── train_results.py                # Model training and evaluation
└── train_test_split.py             # Dataset splitting
```

---

# Pipeline Overview

The pipeline follows the workflow below:

```
                Raw Dataset
                     │
                     ▼
         import_datasets.py
                     │
                     ▼
            etl_cleaning.py
                     │
                     ▼
              scalers.py
                     │
                     ▼
         train_test_split.py
                     │
                     ▼
          train_results.py
                     │
                     ▼
         Results / Tables / Metrics
```

---

# Pipeline Components

## import_datasets.py

Responsible for:

- Loading datasets (from UCI repo)
- Reading metadata
- Preparing the raw data for preprocessing

---

## etl_cleaning.py

Performs the ETL stage:

- Missing value treatment
- Data cleaning
- Data formatting

---

## scalers.py

Applies feature normalization or scaling.

- StandardScaler
- MinMaxScaler
- MaxAbsScaler
- RobustScaler
- QuantileTransformer
- MeanCentered
- VariableStabilityScaling
- ParetoScaling
- DecimalScaling
- TanhTransformer
- LogisticSigmoidTransformer
- HyperbolicTangentTransformer

---

## train_test_split.py

Creates reproducible dataset splits.

Outputs include:

- Training set
- Validation set
- Test set

Split information is exported as:

- JSON
- CSV

for reproducibility.

---

## train_results.py

Core training module.

Responsibilities include:

- Loading processed data
- Training all configured models
- Hyperparameter optimization (if enabled)
- Model evaluation
- Metric calculation
- Saving predictions
- Exporting performance tables

Generated outputs are stored inside:

```
results/
```

---

## analysis-results.ipynb

Jupyter notebook used to:

- Analyze model performance
- Generate plots
- Compare algorithms
- Produce publication-ready figures

---

# Running the Project

The entire pipeline can be executed with a single command:

```bash
python main.py
```

The launcher performs two tasks:

1. Checks whether all required modules exist.
2. Executes the training pipeline.

Internally it verifies the presence of:

- import_datasets.py
- etl_cleaning.py
- scalers.py
- train_test_split.py
- train_results.py

If any required file is missing, execution is interrupted.

---

# Execution Flow

The launcher (`main.py`) performs:

```
Check Required Files
        │
        ▼
All files available?
        │
   Yes ─────► Run train_results.py
        │
        ▼
Save Results
```

---

# Output Files

After execution, the repository generates:

- trained models
- evaluation metrics
- prediction files
- LaTeX tables
- CSV summaries
- experiment logs

inside the `results/` directory.

---

# Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

# Reproducibility

The project was designed to be fully reproducible.

Dataset partitions are stored as:

- `dataset_split_classification.json`
- `dataset_split_classification.csv`
- `dataset_split_regression.json`
- `dataset_split_regression.csv`

ensuring identical train/test splits across experiments.

---

# Citation

# Citation
```latex
@ARTICLE{11261543,
  author={Pinheiro, João Manoel Herrera and Oliveira, Suzana Vilas Boas de and Silva, Thiago Henrique Segreto and Saraiva, Pedro Antonio Rabelo and Souza, Enzo Ferreira de and Godoy, Ricardo V. and Ambrosio, Leonardo André and Becker, Marcelo},
  journal={IEEE Access}, 
  title={The Impact of Feature Scaling in Machine Learning: Effects on Regression and Classification Tasks}, 
  year={2025},
  volume={13},
  number={},
  pages={199903-199931},
  keywords={Machine learning;Machine learning algorithms;Classification algorithms;Standards;Nearest neighbor methods;Data models;Logistics;Benchmark testing;Support vector machines;Reproducibility of results;Data preprocessing;feature scaling;machine learning algorithms;normalization;standardization},
  doi={10.1109/ACCESS.2025.3635541}}
```
