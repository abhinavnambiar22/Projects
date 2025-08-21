# Network Intrusion Detection using Machine Learning

This project uses machine learning to classify network traffic as either benign or a DoS (Denial of Service) attack. The goal is to build and evaluate various classification models for a network intrusion detection system (IDS).

### **Dataset**

The model is trained on the "IDS Intrusion CSV" dataset, specifically the `02-15-2018.csv` file, which contains network flow features and corresponding labels.

### **Models Implemented**

Several classification algorithms were trained and evaluated:
* Support Vector Machine (SVM) with RBF and Polynomial kernels
* Random Forest
* Gradient Boosting
* XGBoost
* Logistic Regression

### **Data Processing**

The workflow includes the following preprocessing steps:
1.  Loading the dataset and dropping the timestamp.
2.  Encoding the categorical labels ("Benign", "DoS Attack") into numerical values (0, 1).
3.  Handling infinite and NaN values through mean imputation.
4.  Scaling all features using `StandardScaler`.
5.  Splitting the data into training and testing sets.

### **Results**

All models achieved exceptionally high accuracy, ranging from **99.80% to 100.00%**, demonstrating their effectiveness in detecting DoS attacks in this dataset.

Performance was evaluated using:
* Accuracy Score
* Classification Reports (Precision, Recall, F1-Score)
* Confusion Matrix
* ROC Curve and Precision-Recall Curve visualizations

### **Dependencies**

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* xgboost
