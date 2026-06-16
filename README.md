# Week5
# FinTech Fraud Detection Architecture - Adey Innovations Inc.

Production-grade pipeline evaluating fraud classification frameworks across E-commerce and Bank Card transactions.

## 🛠️ Environment Initialization & Setup

This infrastructure is engineered and tested using **Python 3.10+**. 

```bash
# 1. Clone the active repository structure
git clone [https://github.com/yourusername/fraud-detection.git](https://github.com/yourusername/fraud-detection.git)
cd fraud-detection

# 2. Configure a clean virtual environment space
python3 -m venv venv
source venv/bin/activate  # On Windows run: venv\Scripts\activate

# 3. Apply the minimal dependency packages
pip install --upgrade pip
pip install -r requirements.txt

## 📊 Datasets Used
This project utilizes three distinct datasets to analyze, detect, and model fraudulent behavior across e-commerce and banking environments:

1. **Fraud_Data.csv (E-commerce Fraud Dataset)**
   * **Description:** Contains e-commerce transaction data, including user sign-up times, purchase times, device IDs, browser types, and a binary target variable (`class`) indicating fraud.
   * **Source:** Provided via challenge repository / Kaggle E-commerce Fraud dataset.

2. **IpAddress_To_Country.csv (Geolocation Mapping)**
   * **Description:** Maps numerical IP address ranges (lower and upper bounds) to specific country names. Used to engineer geographical features from user transaction IPs.
   * **Source:** MaxMind GeoIP database / Challenge reference data.

3. **creditcard.csv (Bank/Credit Card Fraud Dataset)**
   * **Description:** Contains anonymized European cardholder transactions containing highly imbalanced PCA-transformed numerical features ($V1$ to $V28$), transaction amounts, and a target variable (`Class`).
   * **Source:** Kaggle Credit Card Fraud Detection dataset.

---

## 🗂️ Notebook Mapping & Pipeline Workflow
The repository is structured to follow a production-grade machine learning lifecycle. Below is how the notebooks map to each core phase:

* **`01_EDA.ipynb` (Exploratory Data Analysis)**
  * Handles initial data profiling, missing value checks, distributions of transaction frequencies, and visualization of class imbalances for both datasets.
* **`02_Feature_Engineering.ipynb` (Preprocessing & Geolocation Integration)**
  * Merges IP addresses to country codes using range boundaries, converts timestamps to cyclical/duration features, scales numerical inputs, and encodes categorical variables.
* **`03_Model_Building.ipynb` (Modeling & Parameterized Training)**
  * Implements stratified training splits, runs parameterized baselines (Logistic Regression), trains tuned XGBoost architectures, evaluates metrics under extreme class imbalance, and persists artifacts to the `models/` directory.
* **`04_Model_Explainability.ipynb` (Interpretability & Insights)**
  * Deploys SHAP and LIME frameworks to map feature importances and explain why specific transactions were flagged as fraud.