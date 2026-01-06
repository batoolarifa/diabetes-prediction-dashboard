# 💉 Diabetes Prediction End-to-End Machine Learning Pipeline & Dashboard

## **Project Overview**

Diabetes is a growing global health concern affecting millions worldwide. Early prediction can guide preventive interventions, reduce healthcare costs, and improve patient outcomes.

This project delivers a **comprehensive end-to-end solution**:

1. **Machine Learning Pipeline**: Trains multiple gradient boosting models on tabular health data, using advanced **feature engineering** for improved predictive performance.
   
   <br>
2. **Interactive Dashboard**: Deploys the trained model via **Streamlit**, allowing users (patients or clinicians) to estimate diabetes risk using minimal input features, visualize top contributing factors, and receive actionable recommendations.

## **Business & Industry Value**

* **Clinicians**: Quickly assess risk without extensive lab tests.
* **Healthcare Systems**: Population-level risk screening for early intervention.
* **Patients**: Understand personal risk and preventive actions.
* **Data Science / Research**: Provides a reusable pipeline for tabular medical prediction tasks.

## **About Me**

* Software Engineering Student @ Karachi University
* AI & Machine Learning Practitioner | Applying technology to create real-world value 📈

**Connect with Me**:

* [LinkedIn](https://www.linkedin.com/in/arifa-batool/)
* [Kaggle](https://www.kaggle.com/arifa-batool)
* Email: [thearifabatool@gmail.com](mailto:thearifabatool@gmail.com)

## **Table of Contents**


- [💉 Diabetes Prediction End-to-End Machine Learning Pipeline \& Dashboard](#-diabetes-prediction-end-to-end-machine-learning-pipeline--dashboard)
  - [**Project Overview**](#project-overview)
  - [**Business \& Industry Value**](#business--industry-value)
  - [**About Me**](#about-me)
  - [**Table of Contents**](#table-of-contents)
  - [**1. Dataset Overview**](#1-dataset-overview)
  - [**2. Project Workflow**](#2-project-workflow)
    - [**2.1 Loading Data**](#21-loading-data)
    - [**2.2 Quick Data Checks**](#22-quick-data-checks)
    - [**2.3 Exploratory Data Analysis**](#23-exploratory-data-analysis)
    - [**2.4 Feature Engineering**](#24-feature-engineering)
    - [**2.5 Data Preparation for Models**](#25-data-preparation-for-models)
    - [**2.6 Model Training**](#26-model-training)
    - [**2.7 Model Evaluation**](#27-model-evaluation)
    - [**2.8 Final Model Selection \& Prediction**](#28-final-model-selection--prediction)
  - [**3. Dashboard UI**](#3-dashboard-ui)
    - [**3.1 Input Form \& Interactivity**](#31-input-form--interactivity)
    - [**3.2 Prediction \& Risk Visualization**](#32-prediction--risk-visualization)
    - [**3.3 Feature Contributions**](#33-feature-contributions)
  - [**4. Screenshots**](#5-screenshots)
  - [**5. Project Structure**](#4-project-structure)
  - [**6. Getting Started**](#7-getting-started)
    - [**Clone Repository**](#clone-repository)
    - [**Install**](#install)
    - [**Run Streamlit Dashboard**](#run-streamlit-dashboard)
  - [**Conclusion**](#conclusion)



## **1. Dataset Overview**

* Dataset: [Kaggle Playground Series S5E12 – Diabetes Prediction Dataset](https://www.kaggle.com/competitions/playground-series-s5e12/data)
* Type: Tabular health data with **53 features** including demographics, lifestyle, and clinical indicators.
* Target Variable: `diagnosed_diabetes` (Binary: Yes/No)
* Feature Types: Numerical and categorical requiring preprocessing and encoding.


## **2. Project Workflow**

### **2.1 Loading Data**

* Loaded training and test datasets.
* Verified shape, column alignment, and identified the target variable.

### **2.2 Quick Data Checks**

* Checked for missing values, duplicates, and correct data types.
* Conducted basic summary statistics to detect outliers and anomalies.

### **2.3 Exploratory Data Analysis**

* Checked class balance for the target.
* Visualized distributions for numerical features (age, BMI, cholesterol, blood pressure).
* Identified correlations and potential patterns linked to diabetes.

### **2.4 Feature Engineering**

* Generated **new features** using age, BMI, activity, sleep, blood pressure, and cholesterol:

  * Interaction features (e.g., `age_bmi_risk`, `activity_x_age`)
  * Ratios (e.g., `ldl_hdl_ratio`, `tg_hdl_ratio`)
  * Lifestyle-adjusted metrics (e.g., `daily_physical_hours`, `sleep_efficiency_pct`)
* Ensured feature engineering is **applied consistently** to train, test, and live user input in the dashboard.

### **2.5 Data Preparation for Models**

* Encoded categorical features (`Yes` → 1, `No` → 0).
* Ensured all features were numeric and model-ready.
* Split features and target while keeping **cross-validation consistency**.

### **2.6 Model Training**

* Used **Stratified K-Fold Cross-Validation** for robust performance evaluation.
* Models trained:

  * **XGBoost**
  * **LightGBM**
  * **CatBoost**
* Evaluated models using **ROC-AUC** and **Confusion Matrices**.

### **2.7 Model Evaluation**

* Compared cross-validated metrics.
* Visualized ROC curves and confusion matrices for each model.
* Selected the **best-performing model** (LightGBM in final deployment).

### **2.8 Final Model Selection & Prediction**

* Trained final model on full dataset.
* Saved model with Joblib (`best_model.pkl`) for **runtime predictions**.
* Integrated with **Streamlit dashboard** for real-time user predictions.


## **3. Dashboard UI**

The Streamlit dashboard allows users to **enter minimal inputs** and get **real-time diabetes risk predictions**.

### **3.1 Input Form & Interactivity**

* Inputs: Age, BMI, Waist-to-Hip Ratio, Blood Pressure, Heart Rate, Cholesterol, HDL, LDL, Triglycerides, Sleep, Physical Activity, Family History, Cardiovascular History, Hypertension History.
* Custom styling with rounded input boxes.
* Columns to organize inputs neatly.
* Input validation for extreme values with **warnings**.

### **3.2 Prediction & Risk Visualization**

* Predicts **probability of diabetes**.
* Categorizes **risk**:

  * Low Risk → White
  * Moderate Risk → Yellow
  * High Risk → Red
* Dynamic **progress bar** to indicate risk percentage.
* Recommendations provided based on risk category.

### **3.3 Feature Contributions**

* Top **5 contributing features** displayed as interactive **Plotly horizontal bar charts**.
* Helps understand **why the model predicted a certain risk**.


## **4. Screenshots**


**Dashboard Input Form**
![Dashboard Input_1](https://github.com/user-attachments/assets/a50ea59f-ea5e-4539-b481-95f799bd58df)

![Dashboard Input_2](https://github.com/user-attachments/assets/f616e0ea-8f00-4788-b354-51a0cdeaafae)

**Prediction & Risk**
![Prediction Output](https://github.com/user-attachments/assets/e33bce7d-b670-4511-8454-c0e43e76b20a)

**Top 5 Feature Contributions**
![Feature Contribution Chart](https://github.com/user-attachments/assets/613940d8-8880-4acb-8528-f86d6036bc60)

## **5. Project Structure**

Here’s a clean **project structure** in Markdown format for your diabetes prediction project:

```markdown
# Project Structure

```

│
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   └── README.md
│
├── models/
│   ├── best_model.pkl
│   └── predict_model.py
│
├── notebook/
│   ├── diabetic_notebook.ipynb.ipynb
│   
├── utils/
│   ├── feature_engineering.py
│
├── app.py
│── requirements.txt
├── README.md
├── requirements.txt
```
```

## **6. Getting Started**

### **Clone Repository**

```bash
https://github.com/batoolarifa/diabetes-prediction-dashboard
```

### **Install**

```bash
pip install -r requirements.txt
```

### **Run Streamlit Dashboard**

```bash
streamlit run app.py
```

## **Conclusion**


This project covers **end-to-end ML pipeline workflow**, **feature engineering logic**, **model training**, and **interactive Streamlit dashboard**, with screenshots and industry-relevant structure.

