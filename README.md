Anomaly Detection in Transaction Data

Overview

This project focuses on detecting fraudulent transactions using Machine Learning techniques. Implemented using Python, Scikit-Learn, and TensorFlow, the system leverages Isolation Forest and Autoencoders for anomaly detection.

Features

Anomaly Detection Model: Used Isolation Forest to identify fraudulent transactions.

Feature Engineering & Data Analysis: Performed using Pandas & NumPy to enhance insights and model accuracy.

Model Optimization: Applied cross-validation and evaluation metrics to improve performance.

Class Imbalance Handling: Used SMOTE to balance transaction classes.

Visualization & Insights: Correlation heatmap and distribution plots were generated to understand data patterns.

Data Insights

1. Feature Correlation Heatmap

The heatmap shows strong and weak correlations between transaction features.

Identifies redundant and important features for model training.
![correlation_heatmap](https://github.com/user-attachments/assets/bcfd41b7-0292-44bb-bf08-fbfc1081d861)


2. Transaction Amount Distribution

The transaction amounts are highly skewed.

Fraudulent transactions have a similar distribution to normal transactions, emphasizing the need for anomaly detection techniques.
![amount_distribution](https://github.com/user-attachments/assets/ebcd1d70-16de-4151-9016-0fdd18e91a77)

3. Balanced Class Distribution

SMOTE was applied to balance normal and fraudulent transaction counts.

Prevents the model from being biased toward the majority class.

![class_distribution](https://github.com/user-attachments/assets/7c72ee86-bb8c-4cc1-971e-b677843560a2)


Model Performance (Isolation Forest)

Precision: 0.9905

Recall: 0.0191

F1-Score: 0.0375

Observations:

High Precision indicates that most of the flagged fraudulent transactions were correct.

Low Recall suggests that many fraudulent transactions were missed.

F1-Score is low due to the imbalance in precision and recall, highlighting a trade-off.

Tools & Technologies Used

Programming Language: Python

Libraries: Pandas, NumPy, Scikit-Learn, TensorFlow, Matplotlib

Data Preprocessing: Feature Scaling, SMOTE for class balancing

Modeling: Isolation Forest, Autoencoders

Future Improvements

Improve recall by tuning hyperparameters and exploring hybrid models.

Experiment with additional deep learning approaches for anomaly detection.

Deploy the model as an API for real-time fraud detection.

Conclusion

This project successfully demonstrates anomaly detection techniques in financial transactions. While the model achieves high precision, improving recall remains a challenge. Future work will focus on optimizing trade-offs between these metrics for better fraud detection.

Author: Gowrishetty Rohith Kumar
Date: November 2023 - December 2023

