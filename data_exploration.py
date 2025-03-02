import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ✅ Load preprocessed dataset
df = pd.read_csv("data/processed_data.csv")

# ✅ Ensure reports directory exists
os.makedirs("reports", exist_ok=True)

# 📊 **Class Distribution After SMOTE**
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)
plt.title("Balanced Transaction Class Distribution (After SMOTE)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.savefig("reports/class_distribution.png")  # ✅ Save Graph
plt.show()

# 📊 **Correlation Heatmap**
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")
plt.savefig("reports/correlation_heatmap.png")  # ✅ Save Graph
plt.show()

# 📊 **Transaction Amount Distribution (Fraud vs Normal)**
plt.figure(figsize=(10, 5))
sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, color='blue', label="Normal", kde=True)
sns.histplot(df[df['Class'] == 1]['Amount'], bins=50, color='red', label="Fraud", kde=True)
plt.legend()
plt.title("Transaction Amount Distribution (Balanced Data)")
plt.savefig("reports/amount_distribution.png")  # ✅ Save Graph
plt.show()

print("✅ All visualizations saved in `reports/` folder.")
