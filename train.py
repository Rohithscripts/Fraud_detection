import pandas as pd
import numpy as np
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix

# Load preprocessed dataset (Balanced)
df = pd.read_csv("data/processed_data.csv")

# Define features and target
X = df.drop(columns=['Class'])
y = df['Class']

# 🚀 Train Isolation Forest Model on Balanced Data
print("🚀 Training Isolation Forest model...")
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(X)

# 🔍 Predict anomalies
predictions = model.predict(X)
df['Anomaly'] = np.where(predictions == -1, 1, 0)  # Convert -1 to 1 (fraud), 1 to 0 (normal)

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Save trained model
joblib.dump(model, "model/isolation_forest.pkl")
print("✅ Model saved to `model/isolation_forest.pkl`.")

# Save fraud detection results
df.to_csv("data/anomaly_detected.csv", index=False)
print("✅ Fraud detection results saved to `data/anomaly_detected.csv`.")

# 📊 Compute Evaluation Metrics
precision = precision_score(y, df['Anomaly'])
recall = recall_score(y, df['Anomaly'])
f1 = f1_score(y, df['Anomaly'])
conf_matrix = confusion_matrix(y, df['Anomaly'])

print("\n📊 Model Evaluation Metrics:")
print(f"🔹 Precision: {precision:.4f}")
print(f"🔹 Recall: {recall:.4f}")
print(f"🔹 F1 Score: {f1:.4f}")

# ✅ Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Balanced Data)")
plt.show()
