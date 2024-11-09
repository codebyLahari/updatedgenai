import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score


# Load the dataset
df = pd.read_csv('Updated_Healthcare-Diabetes.csv')

#code to show all columns clearly 
pd.set_option('display.max_columns', None)

# Display dataset info (column names, non-null counts, and data types)
print("Dataset Info:")
print(df.info())

# Display the first 5 rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Get summary statistics for numerical columns
print("\nDataset Description (Numerical Summary):")
print(df.describe())

# Update the "Pregnancies" column to limit values to a maximum of 3
df['Pregnancies'] = df['Pregnancies'].apply(lambda x: min(x, 5))

# Set the plot size
plt.figure(figsize=(12, 8))

# Create subplots for each column
columns = ['Glucose', 'Insulin', 'BloodPressure', 'BMI']

# Generate separate boxplots for each column
for i, col in enumerate(columns, 1):
    plt.subplot(2, 2, i)  # 2 rows, 2 columns
    sns.boxplot(data=df, y=col, color='lightblue')
    plt.title(f'Boxplot for {col}')
    plt.ylabel(col)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

# Define the function to replace outliers with mean, ignoring zeros
def replace_outliers_with_mean_ignore_zeros(df, col):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define the outlier range
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Calculate the mean ignoring outliers and zeros
    mean_value = df[(df[col] >= lower_bound) & (df[col] <= upper_bound) & (df[col] != 0)][col].mean()
    
    # Replace outliers with the mean
    df[col] = df[col].where((df[col] >= lower_bound) & (df[col] <= upper_bound), mean_value)

# Columns to process
columns = ['Glucose', 'Insulin', 'BloodPressure', 'BMI']

# Print original descriptive statistics for Insulin before replacing outliers
print("\nOriginal Dataset Description for Insulin:")
print(df['Insulin'].describe())

# Replace outliers for each column
for col in columns:
    replace_outliers_with_mean_ignore_zeros(df, col)

    # Print bounds and mean for each column
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mean_value = df[(df[col] >= lower_bound) & (df[col] <= upper_bound) & (df[col] != 0)][col].mean()
    
    print(f"\nColumn: {col}")
    print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}, Mean Value for Replacement: {mean_value}")

# Print descriptive statistics for Insulin after replacing outliers
print("\nDataset Description for Insulin after Replacing Outliers with Mean (Ignoring Zeros):")
print(df['Insulin'].describe())

# Create box plots after replacing outliers
plt.figure(figsize=(12, 8))
for i, col in enumerate(columns, 1):
    plt.subplot(2, 2, i)  # 2 rows, 2 columns
    sns.boxplot(data=df, y=col, color='lightblue')
    plt.title(f'Boxplot for {col} after Outlier Replacement')
    plt.ylabel(col)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# Correlation Matrix
correlation_matrix = df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Step 5: Logistic Regression Model
# Assuming 'Outcome' column is the target variable for diabetes prediction
X = df[['Glucose', 'BloodPressure', 'Insulin', 'BMI']]
y = df['Outcome']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train logistic regression model
logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nLogistic Regression Model Accuracy: {:.2f}%".format(accuracy * 100))
