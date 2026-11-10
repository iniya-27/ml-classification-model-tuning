# ML Classification Model Training & Hyperparameter Tuning

## Project Overview
This repository implements and evaluates multiple supervised machine learning algorithms (**Logistic Regression**, **Decision Tree**, and **Random Forest**) to classify health metrics data. 

Random Forest was selected as the final production model due to significant improvements in **accuracy** and **F1-score** following hyperparameter tuning (`n_estimators=100`, `max_depth=5`).

---

## Key Features
- **Data Preprocessing:** Standard scaling and missing value check using Pandas & Scikit-Learn.
- **Model Comparison:** Performance benchmarking across accuracy, precision, recall, and F1-score.
- **Hyperparameter Optimization:** Random Forest tuned to achieve peak performance.

---

## File Structure
- `main.py`: Complete ML pipeline script for data loading, model training, tuning, and evaluation.
- `heart_disease_data.csv`: Sample health parameters dataset.
- `requirements.txt`: Python package dependencies.

---

## How to Run

1. Clone the repository:
```bash
git clone [https://github.com/iniya-27/ml-classification-model-tuning.git](https://github.com/iniya-27/ml-classification-model-tuning.git)
cd ml-classification-model-tuning
