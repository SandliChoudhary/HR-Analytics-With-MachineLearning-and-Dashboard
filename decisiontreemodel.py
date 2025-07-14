# 1.  Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 2.  Load the dataset
df = pd.read_excel("HR_data cleaned.xlsx")

# 3.  Drop columns that could leak the target or are unnecessary
df = df.drop(columns=[
    'emp no', 'Employee Number', 'CF_attrition label', 
    'Attrition count', 'CF_current Employee'
], errors='ignore')

# 4.  Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# 5.  Define features and target
X = df.drop(columns='Attrition')  # Features
y = df['Attrition']               # Target

# 6.  Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 7.  Train Decision Tree model with depth control
model = DecisionTreeClassifier(max_depth=4, random_state=0)
model.fit(X_train, y_train)

# 8.  Predict and Evaluate
y_pred = model.predict(X_test)

print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))


# 9.  Visualize the Tree
plt.figure(figsize=(20, 10))
plot_tree(
    model, 
    feature_names=X.columns, 
    class_names=['No', 'Yes'], 
    filled=True, 
    rounded=True, 
    fontsize=5
)
plt.title("Decision Tree for Employee Attrition")
plt.show()
