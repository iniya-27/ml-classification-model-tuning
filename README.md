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

# ML Classification Model Training & Hyperparameter Tuning

## Project Overview
This repository implements and evaluates multiple supervised machine learning algorithms (**Logistic Regression**, **Decision Tree**, and **Random Forest**) to classify health metrics data. 

Random Forest was selected as the final production model due to significant improvements in **accuracy** and **F1-score** following hyperparameter tuning (`n_estimators=100`, `max_depth=5`).

---

## Model Evaluation Summary

| Model | Accuracy | F1-Score | Status |
|---|---|---|---|
| Logistic Regression | 0.8500 | 0.8400 | Baseline |
| Decision Tree | 0.8800 | 0.8750 | Evaluated |
| **Random Forest (Tuned)** | **0.9400** | **0.9380** | **Selected Final Model** |
