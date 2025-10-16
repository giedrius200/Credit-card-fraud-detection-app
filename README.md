# 💳 Credit Card Fraud Detection System

A machine learning–based application developed as part of my **Information Systems Engineering Bachelor’s Thesis at Vilnius University**, focused on detecting fraudulent credit card transactions using real-world datasets and modern classification algorithms.

---

## 🚀 Project Overview
This project addresses the challenge of **identifying fraudulent transactions in financial systems** through data-driven anomaly detection.  
The goal was to compare and evaluate the performance of multiple supervised learning models on highly imbalanced datasets.

---

## 🧠 Key Features
- Implemented **Logistic Regression**, **Decision Trees**, **Random Forest**, **Gradient Boosting**, and **Neural Networks** for fraud classification  
- Performed **data preprocessing**, feature scaling, and **class imbalance correction** using SMOTE  
- Built a **modular Python pipeline** for model training, evaluation, and visualization  
- Evaluated models with metrics such as **Precision**, **Recall**, **F1-Score**, **ROC-AUC**, and **Confusion Matrix**  
- Designed for **reproducibility and interpretability**

---

## 🧩 Tech Stack
- **Programming Language:** Python  
- **Libraries:** pandas, NumPy, scikit-learn, imbalanced-learn, matplotlib, seaborn  
- **Environment:** Jupyter Notebook / VS Code

---

## 📊 Results & Insights
The experiments showed that **ensemble methods** (Random Forest, Gradient Boosting) achieved the best performance on detecting rare fraudulent activities.  
The project demonstrates the **practical application of AI and data science** to strengthen financial fraud prevention systems.

---

## ⚙️ How It Works
1. Load the dataset and preprocess the data  
2. Split into training and test sets  
3. Apply scaling and handle class imbalance  
4. Train models and evaluate performance  
5. Compare metrics and visualize results  

Example:
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
