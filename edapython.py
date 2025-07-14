# 🔍 Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Optional: set a clean style
sns.set(style="whitegrid")

# 📁 Load your cleaned Excel file
df = pd.read_excel("HR_data cleaned.xlsx")

# 1. Basic Info
print("Shape:", df.shape)
print("Column Names:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nFirst 5 Rows:\n", df.head())

# 2. Job Role vs Attrition
plt.figure(figsize=(10, 5))
sns.countplot(x='Job Role', data=df, hue='Attrition', palette='Set1')
plt.title("Attrition by Job Role")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#  3. Department vs Attrition
plt.figure(figsize=(6, 4))
sns.countplot(x='Department', data=df, hue='Attrition', palette='coolwarm')
plt.title("Attrition by Department")
plt.show()

#  4. Age Band Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='CF_age band', data=df, hue='Attrition', palette='Set3')
plt.title("Age Band vs Attrition")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. High Attrition Roles
high_attrition_roles = df[df['Attrition'] == 'Yes']['Job Role'].value_counts()
print("\nTop Job Roles with Highest Attrition:\n", high_attrition_roles)

# 6. heatmap
plt.figure(figsize=(18, 14))  # Bigger figure size
numeric_cols = df.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_cols.corr()
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.7},
    annot_kws={"size": 8}
)
plt.title("Correlation Heatmap of Numeric Features", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 7. by age
# Group by Age Band and calculate Attrition Rate
age_attrition = df.groupby('CF_age band')['Attrition'].value_counts(normalize=True).unstack().fillna(0) * 100

# Plot
age_attrition.plot(kind='bar', stacked=True, colormap='coolwarm', figsize=(10,6))
plt.title("Attrition Rate by Age Band (%)")
plt.ylabel("Percentage")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 8. Box Plot: Monthly Income vs Attrition
plt.figure(figsize=(6, 5))
sns.boxplot(x='Attrition', y='Monthly Income', data=df, palette='cool')
plt.title("Monthly Income Distribution by Attrition")
plt.xlabel("Attrition")
plt.ylabel("Monthly Income")
plt.tight_layout()
plt.show()

# 9 Box Plot: Distance From Home vs Attrition
plt.figure(figsize=(6, 5))
sns.boxplot(x='Attrition', y='Distance From Home', data=df, palette='viridis')

plt.title("Distance From Home vs Attrition")
plt.xlabel("Attrition")
plt.ylabel("Distance From Home")
plt.tight_layout()
plt.show()

