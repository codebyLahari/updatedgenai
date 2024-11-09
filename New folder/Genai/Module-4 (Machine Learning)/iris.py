import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Correct path to the CSV file
df = pd.read_csv(r'C:\Users\lahar\Gen-AI\iris.csv')


#code to show all columns clearly 
pd.set_option('display.max_columns', None)

print(df.info())
print(df.head())
# Check for missing values
print("\nMissing values in each column:\n", df.isnull().sum())

# Encode the 'species' column
label_encoder = LabelEncoder()
df['Species'] = label_encoder.fit_transform(df['Species'])

# Display the mapping of species to numbers
species_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("\nSpecies to Number Mapping:", species_mapping)

# Separate features and target variable
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]  # Change 'species' to your target column if needed
y = df['Species']  # Keep the target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 2: Standardize the training set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Step 3: Apply PCA to the training set
pca = PCA(n_components=2)  # Reduce to 2 dimensions
X_train_pca = pca.fit_transform(X_train_scaled)

# Create a DataFrame for PCA results
pca_train_df = pd.DataFrame(data=X_train_pca, columns=['PC1', 'PC2'])
pca_train_df['Species'] = y_train  # Add the species back for visualization

# Visualize PCA results for the training set
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Species', data=pca_train_df, palette='viridis', alpha=0.7)
plt.title('PCA of Iris Dataset (Training Set)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Step 4: Standardize the test set using the same scaler fitted on the training set
X_test_scaled = scaler.transform(X_test)  # Only transform the test set

# Step 5: Transform the test set using the fitted PCA model
X_test_pca = pca.transform(X_test_scaled)

# Create a DataFrame for PCA results of the test set
pca_test_df = pd.DataFrame(data=X_test_pca, columns=['PC1', 'PC2'])
pca_test_df['Species'] = y_test  # Add the species back for visualization

# Visualize PCA results for the test set
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Species', data=pca_test_df, palette='viridis', alpha=0.7)
plt.title('PCA of Iris Dataset (Test Set)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Step 4: Display explained variance
explained_variance = pca.explained_variance_ratio_
print(f"Explained Variance by Each Principal Component: {explained_variance}")
print(f"Total Explained Variance: {explained_variance.sum():.2f}")

# Step 6: Fit the Logistic Regression model using PCA features from the training set
logistic_model = LogisticRegression()
logistic_model.fit(X_train_pca, y_train)

# Step 7: Perform cross-validation
cv_scores = cross_val_score(logistic_model, X_train_pca, y_train, cv=5)  # 5-fold cross-validation

# Print the cross-validation scores and their mean
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.2f}")

# Step 7: Make predictions on the PCA-transformed test set
y_pred = logistic_model.predict(X_test_pca)

# Step 8: Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print(f"Logistic Regression Accuracy: {accuracy:.2f}")