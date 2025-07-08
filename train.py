# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# === Step 1: Load the dataset ===
df = pd.read_csv(r"dataset/train1.csv")  # Replace with your actual dataset file
df.columns = df.columns.str.strip()  # Clean column names

print("📋 Columns in dataset:", df.columns.tolist())

# === Step 2: Check for 'Label' column ===
if 'Label' not in df.columns:
    raise Exception("❌ Dataset must contain a 'Label' column.")

# === Step 3: Handle missing values and infs ===
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(axis=1, thresh=len(df) * 0.9, inplace=True)  # Drop mostly empty columns
df.dropna(inplace=True)

# === Step 4: Separate features and label ===
y = df['Label']
X = df.drop(columns=['Label'])

# Select only numeric features
X = X.select_dtypes(include=[np.number])
y = y[X.index]  # Align indices

# === Step 5: Encode labels ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === Step 6: Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# === Step 7: Feature scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Step 8: Hyperparameter Tuning with GridSearchCV ===
print("🔍 Starting hyperparameter tuning...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

base_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

model = grid_search.best_estimator_

print(f"✅ Best Parameters: {grid_search.best_params_}")

# === Step 9: Evaluate the tuned model ===
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy: {accuracy:.4f}")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# === Step 10: Save the model, scaler, and label encoder ===
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/app_id_classifier.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(le, "model/label_encoder.pkl")

print("📦 Tuned model, scaler, and label encoder saved to /model/")

# === Step 11: Feature Importance Plot ===
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=feature_names[indices])
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("model/feature_importances.png")
plt.close()

# === Step 12: Confusion Matrix Plot ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("model/confusion_matrix.png")
plt.close()

print("📊 Feature importance and confusion matrix plots saved to /model/")
