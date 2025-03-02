import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Drop unnecessary 'Time' column
df.drop(columns=['Time'], inplace=True, errors='ignore')

# Standardize 'Amount' column
scaler = StandardScaler()
df[['Amount']] = scaler.fit_transform(df[['Amount']])

# ✅ **Apply SMOTE (Oversampling Fraud Cases)**
X = df.drop(columns=['Class'])
y = df['Class']

smote = SMOTE(sampling_strategy=1.0, random_state=42)  # 1:1 Balance
X_resampled, y_resampled = smote.fit_resample(X, y)

# Convert back to DataFrame
df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
df_balanced['Class'] = y_resampled  # Add back the target column

# ✅ **Save Balanced Data**
df_balanced.to_csv("data/processed_data.csv", index=False)
print(f"✅ Preprocessing complete. Balanced data saved to `data/processed_data.csv`.")
