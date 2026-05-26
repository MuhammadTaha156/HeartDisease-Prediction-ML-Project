# Heart Disease Risk Analysis & Predictive Feature Engineering Pipeline

This project delivers an End-to-End Exploratory Data Analysis (EDA), Data Cleaning/Imputation, and Advanced Preprocessing pipeline using a Heart Disease dataset. The core objective of this project is to analyze clinical metrics (like cholesterol, blood pressure, age, and chest pain types), resolve critical data anomalies (zero-value physiological errors), and transform raw health profiles into a clean, fully standardized numerical feature space ready for Machine Learning predictive modeling.

## 🚀 Key Project Highlights
* **Physiological Data Imputation:** Identified and corrected highly unrealistic baseline errors (e.g., `0` values in `Cholesterol` and `RestingBP`) using advanced conditional mean-imputation techniques to preserve data distribution stability.
* **Biomedical Exploratory Data Analysis (EDA):** Leveraged multiple data visualization techniques (**Violin plots, Boxplots, and Hue-based Countplots**) to study interactions between clinical indicators and heart disease outcomes.
* **Dynamic Dataset Profiling:** Applied custom analytical package routines (`sheryanalysis`) to generate rapid feature health insights.
* **Categorical Multi-Encoding:** Converted nominal medical markers (such as chest pain categories, resting ECG states, and ST slope behaviors) into clean binary subspaces via dummy variable scaling.
* **Feature Standardization:** Utilized `StandardScaler` to remove physical scale discrepancies across multi-unit features (like Age vs. Blood Pressure).

---

## 📊 Clinical Dataset Attributes
The dataset profiles clinical features from patients to track heart disease determinants:
* `Age`: Age of the patient [years]
* `Sex`: Patient gender [M: Male, F: Female]
* `ChestPainType`: Type of chest pain [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
* `RestingBP`: Resting blood pressure [mm Hg]
* `Cholesterol`: Serum cholesterol [mm/dl]
* `FastingBS`: Fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
* `RestingECG`: Resting electrocardiogram results [Normal, ST: having ST-T wave abnormality, LVH: showing left ventricular hypertrophy]
* `MaxHR`: Maximum heart rate achieved [Numeric value between 60 and 202]
* `ExerciseAngina`: Exercise-induced angina [Y: Yes, N: No]
* `Oldpeak`: ST depression induced by exercise relative to rest [Numeric value]
* `ST_Slope`: The slope of the peak exercise ST segment [Up: Upsloping, Flat: Flat, Down: Downsloping]
* `HeartDisease`: Target risk classification [1: Heart Disease, 0: Normal]

---

## 🛠️ Project Pipeline Structure

### 1. Initial Inspection & Quality Check
* Analyzed dimensions, structural types, missing intervals, and target balance using `info()`, `describe()`, and `value_counts()`.
* Mapped the explicit count breakdown of the target variable (`HeartDisease`) using horizontal bar charts (`kind='barh'`).

### 2. Targeted Physiology Cleaning & Mean Imputation
* **Cholesterol Imputation:** Caught critical dataset anomalies where `Cholesterol == 0` (physiologically impossible for a living patient). Extracted the localized mean of valid, non-zero rows and smoothly replaced the faulty zeroes.
* **RestingBP Imputation:** Caught structural data entry bugs where `RestingBP == 0`. Cleaned these rows by calculating and substituting the non-zero feature average, rounding values to stabilize the variances.
* *Re-evaluated feature distributions pre- and post-imputation using automated custom subplot layout loops to check for distribution shifts.*

### 3. Advanced Visualization & Clinical Correlations
* Mapped out the relationship between Categorical Risks (`Sex`, `ChestPainType`, `FastingBS`) vs. `HeartDisease` using conditional countplots.
* Utilized **Boxplots** to study distributions of `Cholesterol` levels across healthy vs. heart disease cohorts.
* Utilized **Violin Plots** to visualize the density distributions of patient `Age` against risk levels.
* Reviewed multiple correlations across continuous features using automated linear feature heatmaps.

### 4. Categorical Pipeline Conversion
* Transformed multi-class string categories (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`) into binary numeric columns using dummy encoding configurations.
* Coerced boolean variables into integer standards (`.astype(int)`) to maximize computing speed during ML training.

### 5. Multi-Unit Feature Scaling
* Applied **`StandardScaler`** (Z-score normalization) across numeric indicators (`Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`).
* This scales all variables to center around a mean ($\mu$) of 0 with a standard deviation ($\sigma$) of 1, eliminating multi-unit variance issues during modeling.

---

## 💻 Technical Stack Used
* **Language:** Python
* **Data Core:** Pandas, NumPy
* **Visualization Engine:** Matplotlib, Seaborn
* **Custom Profiling:** `sheryanalysis==0.1.0`
* **Data Normalization:** Scikit-Learn (`sklearn.preprocessing.StandardScaler`)

---

## ⚙️ How to Setup & Run
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name