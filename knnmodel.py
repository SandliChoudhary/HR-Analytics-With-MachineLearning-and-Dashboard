# 📥 Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 📁 Load and clean data
df = pd.read_excel("HR_data cleaned.xlsx")
df = df.drop(columns=['emp no', 'Employee Number', 'CF_attrition label', 'Attrition count'], errors='ignore')

# 🔤 Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])

# 🎯 Use only 2 columns: Age and Distance From Home
X = df[['Age', 'Distance From Home']]
y = df['Attrition']

# ⚖️ Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔀 Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 🧠 Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# ✅ Predict and evaluate
y_pred = knn.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))

# 📊 Simple Scatter Plot of the features
plt.figure(figsize=(8, 6))
plt.scatter(X['Age'], X['Distance From Home'], c=y, cmap='bwr', edgecolor='k', alpha=0.7)
plt.xlabel("Age")
plt.ylabel("Distance From Home")
plt.title("HR Attrition: Age vs Distance From Home")
plt.grid(True)
plt.show()
