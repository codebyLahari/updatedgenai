# Import the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# 1. Load the dataset from the provided Excel file
file_path = 'Student_Performance_Data(SPD24).xlsx'
df = pd.read_excel(file_path)

# Check for missing values and handle if necessary
df = df.dropna()  # Drop any rows with missing values

# 2. Visualize the Data
# Select only numeric columns for correlation matrix
numeric_df = df.select_dtypes(include=[np.number])

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Pairplot to visualize relationships between variables
sns.pairplot(numeric_df)
plt.show()

# 3. Prepare Data for Regression
# Define independent (X) and dependent (y) variables
# Make sure to use only numeric columns and map non-numeric ones if necessary
X = df[['Study Hours', 'Class Participation', 'Extracurricular Activities', 'Motivation Level']]  # Update these column names if needed
y = df['Performance']  # Update if the target column name is different

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create and Train the Regression Model
# Initialize the linear regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# 5. Evaluate the Model
# Calculate Mean Squared Error and R-squared value
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Model Performance:')
print(f'Mean Squared Error: {mse}')
print(f'R-squared value: {r2}')

# 6. Hypothesis Testing using statsmodels
# Add a constant term to the independent variables for the intercept
X_with_constant = sm.add_constant(X)

# Fit the regression model using statsmodels
model_stats = sm.OLS(y, X_with_constant).fit()

# Print out the model summary (contains p-values, coefficients, etc.)
print("\nRegression Model Summary:")
print(model_stats.summary())

# 7. Visualize Predictions vs Actual Performance
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual Performance')
plt.ylabel('Predicted Performance')
plt.title('Actual vs Predicted Student Performance')
plt.show()
