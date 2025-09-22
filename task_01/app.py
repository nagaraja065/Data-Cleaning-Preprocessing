# ===============================
# Step 1: Import libraries
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# Step 2: Load dataset
# ===============================
df = pd.read_csv("data/train.csv")   # make sure train.csv is inside a 'data' folder
print("Shape of dataset:", df.shape)
df.head()
# ===============================
# Step 3: Quick data overview
# ===============================
df.info()
df.describe()
df.isnull().sum().sort_values(ascending=False)
# ===============================
# Step 4: Visualize missing values & distributions
# ===============================
plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

sns.countplot(data=df, x="Pclass", hue="Survived")
plt.title("Survival Count by Passenger Class")
plt.show()

sns.boxplot(x=df["Age"])
plt.title("Age Distribution with Outliers")
plt.show()

sns.histplot(df["Fare"], bins=30)
plt.title("Fare Distribution")
plt.show()
# ===============================
# Step 5: Handle missing values
# ===============================
df["Age"].fillna(df["Age"].median(), inplace=True)          # Age -> median
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)  # Embarked -> mode
df["Fare"].fillna(df["Fare"].median(), inplace=True)        # Fare -> median

# Cabin: too many missing -> simplify by extracting deck letter
df["Cabin"].fillna("Unknown", inplace=True)
df["Cabin"] = df["Cabin"].astype(str).str[0]

df.isnull().sum()
# ===============================
# Step 6: Feature Engineering
# ===============================
# Title from Name
df["Title"] = df["Name"].str.extract(",\s*([^\.]+)\.")

# Family size and IsAlone
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

df[["Name","Title","FamilySize","IsAlone"]].head()
# ===============================
# Step 7: Encode categorical variables
# ===============================
# Sex: binary map
df["Sex"] = df["Sex"].map({"male":0,"female":1})

# One-hot encoding for Embarked, Title, Cabin
df = pd.get_dummies(df, columns=["Embarked","Title","Cabin"], drop_first=True)

df.head()
# ===============================
# Step 8: Handle outliers (Fare example)
# ===============================
Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)
IQR = Q3 - Q1
mask = ~((df["Fare"] < (Q1 - 1.5*IQR)) | (df["Fare"] > (Q3 + 1.5*IQR)))
df = df[mask].copy()
print("Shape after removing outliers:", df.shape)
# ===============================
# Step 9: Scale numeric features
# ===============================
num_cols = ["Age","Fare","FamilySize"]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()
# ===============================
# Step 10: Baseline model check
# ===============================
# Select features (drop non-useful ones)
X = df.drop(["Survived","Name","Ticket","PassengerId"], axis=1)
y = df["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))
# ===============================
# Step 11: Save cleaned dataset
# ===============================
df.to_csv("data/train_cleaned.csv", index=False)
print("Cleaned dataset saved to data/train_cleaned.csv")