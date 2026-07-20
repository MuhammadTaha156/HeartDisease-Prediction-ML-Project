# ❤️ End‑to‑End Heart Disease Prediction (ML): From Data Cleaning to Web Deployment

An end-to-end machine learning project for heart disease risk assessment, featuring advanced EDA, data cleaning, feature engineering, and model deployment.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Project Overview

This project delivers a complete pipeline for heart disease risk prediction, from raw clinical data to an interactive web application. The system processes patient health metrics, handles data anomalies, performs feature engineering, and provides real-time risk assessments using a trained K-Nearest Neighbors (KNN) classifier.

### 🎯 Key Achievements

- **Physiological Data Imputation**: Corrected critical data anomalies (zero values in `Cholesterol` and `RestingBP`) using conditional mean-imputation
- **90%+ Data Quality**: Cleaned and standardized 918 patient records across 12 clinical features
- **Multi-Model Evaluation**: Compared 5 machine learning models (Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM)
- **Interactive Web App**: Deployed a Streamlit-based interface for real-time predictions
- **Production-Ready**: Serialized model, scaler, and feature columns for seamless deployment

## 🏥 Dataset Features

The dataset contains clinical measurements from patients with heart disease risk indicators:

| Feature | Description | Type |
|---------|-------------|------|
| **Age** | Patient age in years | Numeric |
| **Sex** | Male/Female | Categorical |
| **ChestPainType** | TA, ATA, NAP, ASY | Categorical |
| **RestingBP** | Resting blood pressure (mm Hg) | Numeric |
| **Cholesterol** | Serum cholesterol (mg/dL) | Numeric |
| **FastingBS** | Fasting blood sugar > 120 mg/dL (0/1) | Binary |
| **RestingECG** | Normal, ST, LVH | Categorical |
| **MaxHR** | Maximum heart rate achieved | Numeric |
| **ExerciseAngina** | Exercise-induced angina (Y/N) | Binary |
| **Oldpeak** | ST depression induced by exercise | Numeric |
| **ST_Slope** | Up, Flat, Down | Categorical |
| **HeartDisease** | Target (0: Normal, 1: Disease) | Binary |

## 🚀 Project Pipeline

### 1. Exploratory Data Analysis (EDA)

```python
# Automated dataset profiling with custom analytics
import sheryanalysis as sh
sh.analyze(df)
```

**Visual Insights:**
- Feature distributions using histograms with KDE
- Categorical feature relationships with heart disease
- Correlation heatmaps for numerical features
- Box plots and violin plots for clinical metrics

### 2. Data Cleaning & Imputation

**Critical Data Fixes:**
- `Cholesterol` zero-values → Replaced with mean of valid entries (244.64 mg/dL)
- `RestingBP` zero-values → Replaced with mean of valid entries (132.54 mm Hg)
- Verified data integrity post-imputation with distribution checks

```python
# Example: Cholesterol imputation
ch_mean = df.loc[df["Cholesterol"] != 0, "Cholesterol"].mean()
df["Cholesterol"] = df["Cholesterol"].replace(0, ch_mean).round(2)
```

### 3. Feature Engineering

**Categorical Encoding:**
- Applied one-hot encoding to all categorical features
- Converted boolean results to integers for model compatibility

**Feature Standardization:**
- Scaled numerical features using `StandardScaler`
- Centered around mean (μ=0) with standard deviation (σ=1)

### 4. Model Training & Evaluation

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| **Logistic Regression** | 86.96% | 0.8846 |
| **KNN** (Selected) | 86.41% | 0.8815 |
| **Naive Bayes** | 84.78% | 0.8614 |
| **Decision Tree** | 80.98% | 0.8293 |
| **SVM** | 84.78% | 0.8667 |

**Model Selection:** KNN chosen for deployment due to strong performance and interpretability in clinical settings.

### 5. Web Application

The Streamlit application provides:
- ✨ Clean, responsive UI with custom CSS
- 📊 Two-column layout for intuitive data entry
- 🎨 Color-coded risk assessment results
- 📈 Interactive progress bars for risk visualization
- 📋 Input summary expander for transparency

## 💻 Technical Stack

### Core Libraries
```python
pandas==2.2.3          # Data manipulation
numpy==2.1.3          # Numerical operations
scikit-learn==1.6.1   # ML models & preprocessing
matplotlib==3.9.2     # Visualization
seaborn==0.13.2       # Statistical visualizations
streamlit==1.28.0     # Web application
joblib==1.4.2         # Model serialization
```

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/heart-disease-predictor.git
cd heart-disease-predictor
```

### 2. Create Virtual Environment
```bash
# Create environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

## 📁 Project Structure

```
heart-disease-predictor/
├── app.py                  # Streamlit web application
├── main.ipynb              # Jupyter notebook with full pipeline
├── KNN_heart.pkl           # Trained KNN model
├── scaler.pkl              # StandardScaler for feature normalization
├── columns.pkl             # Feature column names for inference
├── heart.csv               # Dataset
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## 🎯 How to Use the Web App

1. **Launch the Application**
   ```bash
   streamlit run app.py
   ```

2. **Enter Patient Information**
   - Fill in all clinical parameters
   - Use sliders for continuous variables
   - Select from dropdown options for categorical variables

3. **Get Prediction**
   - Click "🔍 Predict Your Risk" button
   - View immediate risk assessment
   - Review input summary if needed

4. **Interpret Results**
   - 🟢 **Low Risk**: Green indicator with confidence bar
   - 🔴 **High Risk**: Red indicator with recommendation to consult cardiologist

## 🔬 Model Details

### KNN Classifier Configuration
```python
KNeighborsClassifier(
    n_neighbors=5,
    algorithm='auto',
    leaf_size=30,
    metric='minkowski',
    p=2  # Euclidean distance
)
```

### Preprocessing Pipeline
1. **Imputation**: Mean replacement for zero physiological values
2. **Encoding**: One-hot encoding for categorical variables
3. **Scaling**: StandardScaler for numerical features

## 📊 Results & Insights

### Clinical Correlations
- **Chest Pain Type**: Strongest predictor (ASY patients have higher risk)
- **Age**: Higher risk in older patients
- **Cholesterol**: Mixed correlation with heart disease
- **Exercise Angina**: Significant positive correlation with disease presence

### Model Performance Comparison
The KNN classifier achieved 86.41% accuracy on the test set, making it suitable for deployment in clinical decision support systems.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👤 Author

**Muhammad Taha**
- GitHub: [@Muhammad Taha](https://github.com/MuhammadTaha156)
- LinkedIn: [Muhammad Taha](https://www.linkedin.com/in/muhammad-taha101/)

