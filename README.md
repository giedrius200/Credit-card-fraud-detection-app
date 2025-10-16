# 💳 Credit Card Fraud Detection App (Tkinter + Machine Learning)

An interactive desktop application for **credit card fraud detection** built in Python using **Tkinter** for GUI and **machine learning models** for classification.  
Developed as part of my **Information Systems Engineering Bachelor's Thesis at Vilnius University**, this project combines data analysis, model training, visualization, and explainability (via SHAP) — all inside a single user-friendly interface.

---

## 🧠 Overview

The app allows users to:
- Upload transaction datasets (`.csv`)
- Analyze data (summary stats, correlations, class distributions)
- Choose between multiple ML algorithms  
- Adjust model parameters or use defaults  
- Train models using either **simple train/test split** or **5-fold cross-validation**
- View detailed metrics, confusion matrix, and SHAP explanations
- Predict on new datasets and compare metrics side-by-side

---

## 🧩 Features

✅ Graphical Interface built with **Tkinter** and **ttkthemes**  
✅ Built-in algorithms:
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- Neural Network (MLP)  
✅ Optional 5-fold cross-validation  
✅ SHAP value visualization for feature importance  
✅ Confusion Matrix and evaluation metrics display  
✅ MLP parameter experiments and exportable results  
✅ Threaded execution for smooth UI performance  

---

## 🧰 Tech Stack

| Category | Technologies |
|-----------|--------------|
| **Language** | Python 3.10+ |
| **GUI** | Tkinter, ttkthemes, PIL |
| **Data Science** | pandas, NumPy, scikit-learn, imbalanced-learn |
| **Visualization** | matplotlib, seaborn, SHAP |
| **Metrics** | accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC |

---

## 🗂️ Project Structure

```
fraud_detection_app/
├── kodas_naujas5_copy.py   # Main application file
├── data/                   # (Optional) place CSV files here
├── outputs/                # Confusion matrices, SHAP plots, reports
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/fraud-detection-app.git
cd fraud-detection-app
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Example `requirements.txt`:
```txt
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
ttkthemes
Pillow
shap
```

---

## ▶️ Usage

Run the application:
```bash
python kodas_naujas5_copy.py
```

1. Click **“Pasirinkti CSV failą”** to load your dataset  
2. Click **“Atlikti duomenų analizę”** to view summary and correlations  
3. Select an algorithm and adjust parameters if needed  
4. Click **“Apmokyti modelį”** to train and evaluate  
5. View:
   - Metrics (Accuracy, Precision, Recall, F1, ROC/PR AUC)
   - Confusion Matrix  
   - SHAP summary plot  
6. Use **“Nuspėti naujas klases”** to test predictions on new data  

---

## 📊 Example Output

- Confusion Matrix  
- SHAP Summary Plot  
- Metrics Table (training vs prediction)  
- Exported MLP experiment results (`mlp_experiment_results.csv`)

---

## 🧾 Author

**Giedrius Lukoševičius**  
🎓 Vilnius University – Information Systems Engineering  
📧 [https://www.linkedin.com/in/giedrius-it-dev/]

---

## 🪪 License
This project is licensed under the [MIT License](LICENSE).

---

## 💡 Future Improvements

- Add real-time detection API (FastAPI backend)  
- Add Streamlit web version  
- Include model saving/loading and dataset preprocessing tools  
- Enhance visual theme and metric dashboards  
