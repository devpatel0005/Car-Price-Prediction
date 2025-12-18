# 🚗 Car Price Prediction using Machine Learning

A complete **Machine Learning regression project** that predicts the **price of a car** based on its specifications using **Linear Regression, Decision Tree, and Random Forest** models. The project includes data cleaning, exploratory data analysis (EDA), feature engineering, encoding, scaling, model training, and evaluation.

---

## 📌 Project Overview

Buying or selling a car requires a fair understanding of its market price. This project aims to build a predictive model that estimates car prices using multiple features such as engine size, mileage, fuel type, car company, and more.

The model is trained on a real-world car price dataset and evaluated using **R² score** to compare different regression algorithms.

---

## 🧠 ML Models Used

* **Linear Regression**
* **Decision Tree Regressor**
* **Random Forest Regressor**

Among these, **Random Forest** achieves the best performance with higher accuracy and better generalization.

---

## 🛠️ Tech Stack

* **Python**
* **Pandas, NumPy** – Data handling
* **Matplotlib, Seaborn** – Data visualization
* **Scikit-learn** – ML models & preprocessing
* **Google Colab / Jupyter Notebook**

---

## 📂 Dataset

* Source: `CarPrice_Assignment.csv`
* Loaded directly from GitHub raw URL
* Contains both **numerical** and **categorical** features

### Target Variable

* `price`

---

## 🔍 Exploratory Data Analysis (EDA)

* Checked missing values & duplicates
* Univariate analysis using **boxplots**
* Handled **skewness** using `log1p` transformation
* Correlation analysis using **heatmap**
* Brand name extracted from `CarName`

---

## ⚙️ Data Preprocessing

* Dropped irrelevant columns (`car_ID`, `symboling`)
* One-Hot Encoding for categorical features
* Median imputation for missing values
* Feature scaling using **StandardScaler**
* Train-test split (80% / 20%)

---

## 📈 Model Evaluation

Evaluation Metric Used:

* **R² Score**

| Model             | Train R²  | Test R²             |
| ----------------- | --------- | ------------------- |
| Linear Regression | High      | Moderate            |
| Decision Tree     | Very High | Lower (Overfitting) |
| Random Forest     | Very High | **Best**            |

---

## 📊 Results

* Random Forest performs best with strong generalization
* Linear Regression works well but limited by linear assumptions
* Decision Tree shows overfitting

---

## 🚀 How to Run the Project Locally

```bash
# Clone the repository
git clone https://github.com/devpatel0005/Car-Price-Prediction.git

# Move into project directory
cd Car-Price-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the script
python car_price_prediction.py
```

---

## 🌐 Deployment (Optional)

This ML model can be deployed using **Streamlit**, **Flask**, or **FastAPI**.


---

## 📌 Future Improvements

* Hyperparameter tuning
* Model persistence using `joblib`
* Web interface for user input
* CI/CD pipeline for deployment

---

## 👤 Author

**Patel Dev Dharmesh**

* GitHub: [https://github.com/devpatel0005](https://github.com/devpatel0005)
* LinkedIn: [https://www.linkedin.com/in/devdpatel190905](https://www.linkedin.com/in/devdpatel190905)

---

## ⭐ If you like this project, give it a star!
