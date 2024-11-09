import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


# Load the dataset into a DataFrame
file_path = r"C:\Users\lahar\Gen-AI\loan_default_dataset.csv"
df = pd.read_csv(file_path)

#code to show all columns clearly 
pd.set_option('display.max_columns', None)

# Check for missing values in each column
missing_values = df.isnull().sum()

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

# Create boxplots for selected features
features = ['InterestRate', 'LoanAmount', 'DTIRatio', 'LoanTerm', 'NumCreditLines']

plt.figure(figsize=(15, 10))

for i, feature in enumerate(features):
    plt.subplot(3, 2, i + 1)  # Adjust the grid as needed
    sns.boxplot(data=df, y=feature, hue='Default', palette='Set2', legend=False)  # Use hue instead of x
    plt.title(f'Boxplot of {feature} by Default Status')
    plt.xlabel('Default')
    plt.ylabel(feature)

plt.tight_layout()
plt.show()


# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=['number'])

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()


# Calculate the correlation matrix including only numeric variables
correlation_matrix = df.select_dtypes(include='number').corr()

# Create a heatmap for the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".4f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix Heatmap')
plt.show()

X = df[['InterestRate', 'LoanAmount', 'DTIRatio', 'LoanTerm', 'NumCreditLines']]
y = df['Default']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit the logistic regression model with increased max_iter
model = LogisticRegression(max_iter=200)  # Increase max_iter
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create pie charts for each Default status
plt.figure(figsize=(12, 6))

# Pie chart for Default = 0
plt.subplot(1, 2, 1)
default_0_counts = df[df['Default'] == 0]['NumCreditLines'].value_counts()  # Fix: Use df to filter
plt.pie(default_0_counts, labels=default_0_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2", n_colors=len(default_0_counts)))
plt.title('Distribution of NumCreditLines for Default = 0')

# Pie chart for Default = 1
plt.subplot(1, 2, 2)
default_1_counts = df[df['Default'] == 1]['NumCreditLines'].value_counts()  # Fix: Use df to filter
plt.pie(default_1_counts, labels=default_1_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2", n_colors=len(default_1_counts)))
plt.title('Distribution of NumCreditLines for Default = 1')

plt.tight_layout()
plt.show()

# Perform label encoding in one line
df['EmploymentTypeencoded'] = LabelEncoder().fit_transform(df['EmploymentType'])

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='EmploymentType', hue='Default', palette='Set2')
plt.title('Count of Default by Employment Type')
plt.xlabel('Employment Type')
plt.ylabel('Count')
plt.legend(title='Default', loc='upper right', labels=[0,1])
plt.xticks(rotation=45)
plt.show()

