import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load the dataset into a DataFrame
file_path = r"C:\Users\lahar\Gen-AI\ecommerce_dataset.csv"
df = pd.read_csv(file_path)

# Check for missing values in each column
missing_values = df.isna().sum()

# Display the number of missing values for each column
print("Missing Values in Each Column:")
print(missing_values[missing_values > 0])

# Display total number of rows and columns
print(f"\nTotal Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")

# Display DataFrame info
print("\nDataFrame Info:")
print(df.info())

# Display the first few rows of the DataFrame
print("\nDataFrame Head:")
print(df.head())

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())  # Include all columns, even non-numeric
# Generate a pair plot for selected numeric columns
# Adjust the column names according to your dataset
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

if len(numeric_columns) > 1:  # Ensure there are at least two numeric columns
    sns.pairplot(df[numeric_columns])
    plt.title('Pair Plot of Numeric Features')
    plt.show()
else:
    print("Not enough numeric columns for a pair plot.")

# Convert categorical columns to numerical using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Specify the columns of interest
columns_of_interest = ['Age', 'Time_Spent', 'Number_of_Items_Viewed', 'Purchased']

# Check if the specified columns exist in the DataFrame
if all(col in df.columns for col in columns_of_interest):
    # Calculate the correlation matrix for the specified columns
    correlation_matrix = df[columns_of_interest].corr()

    # Display the correlation matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Correlation Matrix: Selected Features')
    plt.show()
else:
    print("One or more specified columns do not exist in the dataset.")
# Specify the columns of interest
columns = ['Age', 'Time_Spent', 'Number_of_Items_Viewed']

# Specify features and target variable
X = df[columns]
y = df['Purchased']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Make predictions
y_pred = logistic_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

