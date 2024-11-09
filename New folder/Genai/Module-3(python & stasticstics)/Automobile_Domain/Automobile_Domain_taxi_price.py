import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = r"C:\Users\lahar\Gen-AI\taxi_dataset.csv"
data = pd.read_csv(file_path)

# Code to show all columns clearly 
pd.set_option('display.max_columns', None)

# Display the first few rows of the dataset
print("Head of the dataset:")
print(data.head(), end="\n\n")

# Display information about the dataset
print("Info about the dataset:")
print(data.info(), end="\n\n")

# Display summary statistics of the dataset
print("Summary statistics:")
print(data.describe(), end="\n\n")

# Check for missing values
missing_values = data.isnull().sum()
print("Missing values in each column:")
print(missing_values[missing_values > 0])

# Create a pair plot for selected numeric columns
pairplot_columns = ['trip_distance', 'fare_amount', 'trip_duration']
sns.pairplot(data[pairplot_columns])
plt.title('Pair Plot for Selected Columns')
plt.show()

# Define features and target variable
X = data[['trip_distance', 'trip_duration']]  # Features
y = data['fare_amount']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions
y_pred = linear_model.predict(X_test)

# Print model coefficients and evaluation metrics
print(f"Model Coefficients: {linear_model.coef_}")
print(f"Intercept: {linear_model.intercept_:.2f}")

# Optional: Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.title('Actual vs. Predicted Fare Amounts')
plt.xlabel('Actual Fare Amount')
plt.ylabel('Predicted Fare Amount')
plt.show()