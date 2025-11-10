# ============================================
# 🩺 DIABETES PREDICTION USING MACHINE LEARNING
# ============================================

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------------
# Step 1: Load the Dataset
# ------------------------------
# Pima Indians Diabetes Dataset (available publicly)
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

# Load dataset into pandas DataFrame
data = pd.read_csv(url)

print("✅ Dataset Loaded Successfully!\n")
print("First 5 Rows of the Dataset:")
print(data.head())

# ------------------------------
# Step 2: Data Preprocessing
# ------------------------------
print("\n🔍 Checking for Missing Values:")
print(data.isnull().sum())

# Define features (X) and target (y)
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Feature scaling using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# Step 3: Split Data into Train & Test
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\n📊 Data Split Completed:")
print(f"Training Samples: {X_train.shape[0]}")
print(f"Testing Samples: {X_test.shape[0]}")

# ------------------------------
# Step 4: Train the Model
# ------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

print("\n🤖 Model Training Completed Successfully!")

# ------------------------------
# Step 5: Make Predictions
# ------------------------------
y_pred = model.predict(X_test)

# ------------------------------
# Step 6: Evaluate the Model
# ------------------------------
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("\n📈 Model Evaluation Results:")
print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

# ------------------------------
# Step 7: Test with a Sample Input
# ------------------------------
# Sample input format:
# [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
sample_data = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])

# Scale the sample data before prediction
sample_scaled = scaler.transform(sample_data)
sample_prediction = model.predict(sample_scaled)

print("🔮 Sample Prediction Result:")
if sample_prediction[0] == 1:
    print("➡️ The person is likely to have Diabetes.")
else:
    print("➡️ The person is NOT likely to have Diabetes.")

# ------------------------------
# Step 8: End of Program
# ------------------------------
print("\n✅ Program Completed Successfully!")
