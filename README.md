# Titanic_EDA
Exploratory Data Analysis on the Titanic dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
df = pd.read_csv("C:/Users/anurag/Downloads/train.csv")

# Display first few rows
display(df.head())

# Check for missing values
print(df.isnull().sum())

# Fill missing Age values with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked values with the most frequent value
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column (too many missing values)
df.drop(columns=['Cabin'], inplace=True)

# Univariate Analysis
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# Survival rate by gender
plt.figure(figsize=(6,4))
sns.barplot(x=df['Sex'], y=df['Survived'])
plt.title("Survival Rate by Gender")
plt.show()

# Survival rate by class
plt.figure(figsize=(6,4))
sns.barplot(x=df['Pclass'], y=df['Survived'])
plt.title("Survival Rate by Passenger Class")
plt.show()

# Creating new feature: Family Size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Survival rate by family size
plt.figure(figsize=(8,5))
sns.barplot(x=df['FamilySize'], y=df['Survived'])
plt.title("Survival Rate by Family Size")
plt.show()

# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=['number'])

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
