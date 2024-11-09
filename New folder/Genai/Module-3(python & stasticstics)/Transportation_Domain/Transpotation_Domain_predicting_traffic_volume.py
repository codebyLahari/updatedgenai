import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load the dataset into a DataFrame
df = pd.read_csv(r'C:\Users\lahar\Gen-AI\Metro_Interstate_Traffic_Volume.csv')

#code to show all columns clearly 
pd.set_option('display.max_columns', None)

# Fill missing values
df['holiday'].fillna('No Holiday', inplace=True)  # Replace NaN in 'holiday' with 'No Holiday'

# Now print the DataFrame or summary to see all columns
print(df.describe())
# Check for null values in each column
null_values = df.isnull().sum()
print("Null Values in Each Column:")
print(null_values)


# Convert 'date_time' to datetime
df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce', dayfirst=True)

# Extract date and time
df['time'] = df['date_time'].dt.time
df.drop('date_time', axis=1, inplace=True)
# Display DataFrame information, head, and summary statistics
print("\nDataFrame Info:")
print(df.info())

print("\nDataFrame Head:")
print(df.head())

# Create subplots for each column
columns = ['rain_1h','temp']

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

# Remove outliers in a single line
data = df[df['rain_1h'].isin([999]) & df['temp'].isin([50])]

# Example of categorical columns
categorical_columns = ['holiday', 'weather_main']  # Replace with your actual categorical column names

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to each categorical columns
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

print("\nSummary Statistics:")
print(df.describe())  # Include all columns, even non-numeric

num_df=df[['temp','rain_1h','snow_1h','clouds_all','traffic_volume']]

# Calculate the correlation matrix
correlation_matrix = num_df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create the heatmap
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
# Set the title
plt.title('Heatmap of Feature Correlations')
# Show the plot
plt.show()

# Define the target variable and features
X = df[categorical_columns]  # Replace 'target_column' with your actual target column
y = df['traffic_volume']  # Replace with your actual target column

# Split the dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')