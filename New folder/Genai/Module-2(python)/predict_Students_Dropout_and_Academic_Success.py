#Data Visualization using Matplotlib and Seaborn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Replace 'path_to_your_file.csv' with the actual path to your CSV file.
df = pd.read_csv(r"C:\Users\lahar\Gen-AI\Predict Student Dropout and Academic Success.csv")


# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Get basic information about the dataset
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values in the Dataset:")
print(df.isnull().sum())

# Describe the dataset (gives statistical summary of numerical columns)
print("\nStatistical Summary:")
print(df.describe())
#Convert numerical columns to float
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].astype(float)

# Correlation matrix between numerical columns
correlation_matrix = df[numeric_columns].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# -------------------------
# Visualizations
# -------------------------

# 1. Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 2. Distribution of a numerical column (e.g., GPA)
plt.figure(figsize=(8, 5))
sns.histplot(df['GPA'], kde=True, bins=30)
plt.title('Distribution of GPA')
plt.xlabel('GPA')
plt.ylabel('Frequency')
plt.show()

# 3. Scatter plot to visualize the relationship between two variables (e.g., Attendance Rate and GPA)
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Attendance Rate', y='GPA', data=df)
plt.title('Attendance Rate vs GPA')
plt.xlabel('Attendance Rate')
plt.ylabel('GPA')
plt.show()

# 4. Bar plot of student dropout rates by category (assuming there is a 'Dropout' column)
if 'Dropout' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Dropout', data=df)
    plt.title('Student Dropout Count')
    plt.xlabel('Dropout Status')
    plt.ylabel('Count')
    plt.show()
else:
    print("\n' Dropout' column not found in the dataset.")

# 5. Box plot to see distribution of GPA by category (assuming there's a 'Study Time' column)
if 'Study Time' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Study Time', y='GPA', data=df)
    plt.title('GPA Distribution by Study Time')
    plt.xlabel('Study Time')
    plt.ylabel('GPA')
    plt.show()
else:
    print("\n' Study Time' column not found in the dataset.")

