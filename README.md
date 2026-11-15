# Telecom Customer Churn: Analysis and Baseline Prediction

CSC-487 Introduction to Data Science, Assignment 2 (Complex Computing Problem), Fall 2026.
Author: **Ayman Khan, <ROLL NO. 02-134241-024>**
        **Inshrah, <ROLL NO. 02-134241-030>**
        **Hafsa Khattak, <ROLL NO. 02-134222-108>**
      
Transparent, reproducible churn analysis with two baseline classifiers (Logistic Regression, K-Nearest Neighbors) evaluated on a held-out test set with confusion matrix, precision, recall, F1, ROC-AUC and PR-AUC.

## Dataset
IBM *Telco Customer Churn* sample dataset (7,043 customers, 21 columns, target `Churn`).
* Source used by the notebook: https://github.com/IBM/telco-customer-churn-on-icp4d (`data/Telco-Customer-Churn.csv`)
* Also listed on Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
* Public sample data released by IBM for teaching; no personal data. Please check the source page for the current usage terms.

**Acquisition:** the notebook downloads the CSV automatically to `data/telco.csv` if it is missing. To do it manually:
```bash
mkdir -p data
curl -L -o data/telco.csv https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```
(A copy is included in `data/` for convenience.)

## How to run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook churn_analysis.ipynb                   # Run All
```
Google Colab also works: upload `churn_analysis.ipynb` and run all cells (internet access needed for the download).
Everything is seeded (`random_state=42`), so results are reproducible.

## Repository layout
| Path | Content |
|---|---|
| `churn_analysis.ipynb` | Full pipeline: audit, cleaning, EDA, preprocessing, models, evaluation, recommendation |
| `figures/` | All generated plots (fig0 workflow, fig1-fig11) |
| `outputs/` | CSV/JSON tables (model comparison, treatments, feature selection, K search, bootstrap CIs, cost analysis) |
| `requirements.txt` | Python dependencies |
| `DEMO_SCRIPT.md` | Narration script for the 3-5 minute demonstration video |

## Key results (test set, n = 1,409, 374 churners)
| Model | Threshold | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Majority baseline | 0.5 | 0.00 | 0.00 | 0.00 | 0.500 |
| Logistic Regression | 0.5 | 0.639 | 0.545 | 0.589 | 0.839 |
| Logistic Regression (balanced) | 0.5 | 0.509 | 0.778 | 0.615 | 0.839 |
| KNN (K=81) | 0.5 | 0.588 | 0.583 | 0.585 | 0.833 |
| KNN (K=81) | 0.31 | 0.499 | 0.805 | 0.616 | 0.833 |

Design choices: split before any learned transformation; imputation/scaling/encoding inside sklearn `Pipeline`s; K and thresholds chosen with cross-validation on the training set only; bootstrap CIs for uncertainty.

## Limitations
Single snapshot, no time dimension, no churn reason, illustrative cost figures. See the notebook conclusions and the PDF report.
