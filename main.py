import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Generate synthetic dataset for ML Model Classification
np.random.seed(42)
n_samples = 500

age = np.random.randint(18, 70, size=n_samples)
bmi = np.random.uniform(18.5, 35.0, size=n_samples)
bp = np.random.randint(80, 140, size=n_samples)
cholesterol = np.random.randint(150, 280, size=n_samples)
# Target variable based on synthetic logic
target = ((age * 0.03 + bmi * 0.05 + bp * 0.02 + cholesterol * 0.01) > 8.5).astype(int)

df = pd.DataFrame({
    'Age': age,
    'BMI': bmi,
    'BloodPressure': bp,
    'Cholesterol': cholesterol,
    'Target': target
})

df.to_csv('heart_disease_data.csv', index=False)

# Train and evaluate models
X = df.drop('Target', axis=1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Random Forest (Tuned)
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Models Trained Successfully")
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Random Forest F1 Score: {f1_score(y_test, y_pred_rf):.4f}")
