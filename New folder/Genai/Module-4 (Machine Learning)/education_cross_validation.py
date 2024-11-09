import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression  # Example model
from sklearn.metrics import accuracy_score


# Load the dataset
data = pd.read_excel('Student_Performance_Data(SPD24).xlsx')

# --- Step 1: Data Exploration ---
print("----- Data Exploration -----")
# Display basic statistics (mean, median, mode) for key features
print("Mean Attendance Rate: ", data['Attendance Rate'].mean())
print("Median Study Hours: ", data['Study Hours'].median())
print("Mode of Socioeconomic Status: ", data['Socioeconomic Status'].mode()[0])

# Check for missing values
print("\nMissing values in each column:\n", data.isnull().sum())



# Example using LabelEncoder for binary categorical variables like 'Yes'/'No'
label_encoder = LabelEncoder()

# Convert binary columns like 'Yes'/'No' into 1/0
data['Attendance of Tutoring Sessions'] = label_encoder.fit_transform(data['Attendance of Tutoring Sessions'])  # Yes/No -> 1/0
data['Parental Involvement'] = label_encoder.fit_transform(data['Parental Involvement'])  # High/Low -> 1/0
data['Behavioral Issues'] = label_encoder.fit_transform(data['Behavioral Issues'])  # Yes/No -> 1/0

# Convert 'Socioeconomic Status' (Low/Medium/High) to numeric
data['Socioeconomic Status'] = label_encoder.fit_transform(data['Socioeconomic Status'])

# Select relevant features for clustering (converted to numeric)
X = data[['Attendance Rate', 'Study Hours', 'Behavioral Issues']]

# Scale the data to ensure all features contribute equally
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Step 3: Clustering Students ---
# Convert 'Performance Score' to numeric (assuming 'High', 'Medium', 'Low' strings)
performance_mapping = {'High': 2, 'Medium': 1, 'Low': 0}
data['Performance Score Numeric'] = data['Performance Score'].map(performance_mapping)

# Visualize the clusters
# Create scatter plot with 'Attendance Rate' and 'Study Hours' as input, and 'Performance Score Numeric' as output
sns.scatterplot(data=data, x='Attendance Rate', y='Study Hours', hue='Performance Score Numeric', palette='coolwarm')
plt.title('Student Performance based on Attendance and Study Hours')
plt.show()


# --- Step 5: Logistic Regression ---
print("\n----- Logistic Regression -----")
# Simplify and convert 'Performance Score' to binary (High = 1, else = 0)
data['Performance Binary'] = data['Performance Score'].apply(lambda x: 1 if x == 'High' else 0)

# Define feature set and target variable
X = data[['Attendance Rate', 'Study Hours', 'Parental Involvement']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variables to dummies
y = data['Performance Binary']

# Split the data into training and temp (for validation and test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Further split temp into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize StratifiedKFold for cross-validation
kf = StratifiedKFold(n_splits=5)

# Cross-validation
cv_scores = []
model = LogisticRegression(max_iter=1000)  # Example model

for train_index, val_index in kf.split(X_train, y_train):
    X_kf_train, X_kf_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_kf_train, y_kf_val = y_train.iloc[train_index], y_train.iloc[val_index]
    
    model.fit(X_kf_train, y_kf_train)
    y_kf_pred = model.predict(X_kf_val)
    
    cv_score = accuracy_score(y_kf_val, y_kf_pred)
    cv_scores.append(cv_score)

# Output the cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean CV score:", sum(cv_scores) / len(cv_scores))


# Optionally, save the datasets to new Excel files
X_train.to_excel(r"C:\Users\lahar\Gen-AI\train_data.xlsx", index=False)
X_val.to_excel(r"C:\Users\lahar\Gen-AI\val_data.xlsx", index=False)
X_test.to_excel(r"C:\Users\lahar\Gen-AI\test_data.xlsx", index=False)

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# --- Step 6: Logistic Regression with Confusion Matrix ---
print("\n----- Logistic Regression with Confusion Matrix -----")

# Train the Logistic Regression model on the training data
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
cm_display.plot(cmap='Blues')
plt.title("Confusion Matrix for Student Performance Prediction")
plt.show()

# Print the classification report (includes precision, recall, F1-score, etc.)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
