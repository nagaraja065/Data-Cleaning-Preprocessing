# 📌 Task 1: Data Cleaning & Preprocessing (Titanic Dataset)

## 📖 Overview
This project is part of **Task 1: Data Cleaning and Preprocessing**, where we prepare the Titanic dataset for further machine learning analysis.  
The notebook includes steps for:
- Handling missing values  
- Feature engineering  
- Encoding categorical variables  
- Outlier detection & removal  
- Scaling numerical features  
- Training a baseline Logistic Regression model  

The cleaned dataset is saved as `train_cleaned.csv` for reuse.  

---

## 📂 Project Structure
```
task1-data-cleaning/
 ├─ data/
 │   ├─ train.csv            # Raw dataset
 │   └─ train_cleaned.csv    # Cleaned dataset
 ├─ notebooks/
 │   └─ task1_cleaning.ipynb # Main notebook
 ├─ README.md                # Project documentation
 └─ requirements.txt         # Dependencies
```

---

## 🛠️ Steps Performed
1. **Data Loading & Exploration**  
   - Checked dataset info, summary, missing values  
   - Visualized distributions (Age, Fare, Pclass vs Survival)  

2. **Handling Missing Values**  
   - Age → median imputation  
   - Embarked → most frequent (mode)  
   - Fare → median  
   - Cabin → simplified to deck letter, filled missing as "Unknown"  

3. **Feature Engineering**  
   - Extracted **Title** from passenger names  
   - Created **FamilySize** (`SibSp + Parch + 1`)  
   - Derived **IsAlone** (binary feature)  

4. **Encoding Categorical Variables**  
   - Sex → mapped to 0 (male), 1 (female)  
   - One-hot encoding for Embarked, Title, and Cabin  

5. **Outlier Handling**  
   - Used IQR method to remove extreme values in **Fare**  

6. **Scaling Features**  
   - Standardized Age, Fare, and FamilySize  

7. **Baseline Model**  
   - Logistic Regression trained on cleaned dataset  
   - Evaluated using accuracy and classification report  

---

## 📊 Results
- Cleaned dataset saved: `data/train_cleaned.csv`  
- Logistic Regression baseline accuracy: **~0.78 – 0.82** (varies depending on outlier handling & splits)  

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/titanic-data-cleaning-task1.git
   cd titanic-data-cleaning-task1
   ```

2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/task1_cleaning.ipynb
   ```

---

## 📦 Dependencies
- Python 3.8+  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- jupyter  

(installable via `pip install -r requirements.txt`)  

---

## ✅ Deliverables
- Jupyter notebook: `notebooks/task1_cleaning.ipynb`  
- Cleaned dataset: `data/train_cleaned.csv`  
- Project report (this README)  
