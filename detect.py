import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load("model/isolation_forest.pkl")

# ✅ Define a sample transaction (Modify values as needed)
new_transaction = pd.DataFrame([{
    "V1": 1.2, "V2": -2.3, "V3": 0.4, "V4": 0.8, "V5": -0.5, "V6": 1.1, "V7": -0.3, "V8": 0.7, "V9": -1.2, 
    "V10": 0.6, "V11": -0.9, "V12": 1.5, "V13": 0.4, "V14": -0.8, "V15": 0.9, "V16": -0.2, "V17": 0.1, 
    "V18": -1.0, "V19": 0.3, "V20": -0.4, "V21": 0.2, "V22": -0.7, "V23": 0.6, "V24": -0.5, "V25": 0.8, 
    "V26": -0.9, "V27": 1.3, "V28": -1.4, "Amount": 200.0
}])

# ✅ Predict fraud status
prediction = model.predict(new_transaction)
fraud_status = "Fraudulent" if prediction[0] == -1 else "Normal"

# ✅ Output result
print(f"🛑 Transaction Status: {fraud_status}")
