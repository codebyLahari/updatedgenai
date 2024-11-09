import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load the CSV file
file_path = r"C:\Users\lahar\Gen-AI\dailyActivity_merged.csv"
df = pd.read_csv(file_path)

#code to show all columns clearly 
pd.set_option('display.max_columns', None)

# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# Get basic information about the DataFrame
print("\nDataFrame Info:")
print(df.info())

# Display the first few rows of the DataFrame
print("\nDataFrame Head:")
print(df.head())

# Get descriptive statistics
print("\nDataFrame Description:")
print(df.describe())

# Convert 'activityDate' to datetime format
df['ActivityDate'] = pd.to_datetime(df['ActivityDate'])

# Set up the plotting style
sns.set(style="whitegrid")

# Create a box plot for total steps
plt.figure(figsize=(12, 6))
sns.boxplot(x='ActivityDate', y='TotalSteps', data=df)
plt.xticks(rotation=45)
plt.title('Box Plot of Total Steps by Activity Date')
plt.xlabel('Activity Date')
plt.ylabel('Total Steps')
plt.tight_layout()
plt.show()

# Create a box plot for calories
plt.figure(figsize=(12, 6))
sns.boxplot(x='ActivityDate', y='Calories', data=df)
plt.xticks(rotation=45)
plt.title('Box Plot of Calories by Activity Date')
plt.xlabel('Activity Date')
plt.ylabel('Calories')
plt.tight_layout()
plt.show()

# Calculate the correlation matrix excluding 'Id'
correlation_matrix = df.drop(columns=['Id','ActivityDate']).corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Prepare data for linear regression
X = df.drop(columns=['Id', 'Calories','ActivityDate'])  # Independent variables
y = df['Calories']  # Dependent variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Create a scatter plot of actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Line for perfect prediction
plt.title('Actual vs Predicted Total Active Minutes')
plt.xlabel('Actual Total Active Minutes')
plt.ylabel('Predicted Total Active Minutes')
plt.show()

# Print the coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

