import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Loading Dataset...")

# Load Dataset
df = pd.read_csv("dataset/churn.csv")

print(df.head())

# -----------------------------
# Data Cleaning
# -----------------------------

print("\nCleaning Dataset...")

# Remove customerID
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Convert required columns

df["gender"] = df["gender"].map({
    "Female": 0,
    "Male": 1
})

df["Partner"] = df["Partner"].map({
    "No": 0,
    "Yes": 1
})

df["Dependents"] = df["Dependents"].map({
    "No": 0,
    "Yes": 1
})

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

print("Dataset Cleaned Successfully!")

# -----------------------------
# Features
# -----------------------------

X = df[[
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]]

y = df["Churn"]

print("\nFeatures Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nDataset Split Successfully!")

# -----------------------------
# Train Model
# -----------------------------

print("\nTraining Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Trained Successfully!")

# -----------------------------
# Evaluation
# -----------------------------

y_pred = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------

joblib.dump(model, "models/churn_model.pkl")

print("\nModel Saved Successfully!")

print("\nBackend Completed Successfully!")