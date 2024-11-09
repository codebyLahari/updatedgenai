import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind

# Load the dataset
file_path = r"C:\Users\lahar\Gen-AI\xAPI-Edu-Data.csv"
data = pd.read_csv(file_path)

# Overview of the dataset
print("First few rows of the dataset:")
print(data.head())

# 1. Check for Simpson's Paradox

# Let's compare the overall grades between students who participate in extracurricular activities and those who don't
# Assuming 'ParentschoolSatisfaction' is a proxy for extracurricular activity participation and 'Class' as their grade level
grouped = data.groupby('ParentschoolSatisfaction')['Class'].value_counts(normalize=True).unstack()
print("\nClass distribution by extracurricular participation (Simpson's Paradox):")
print(grouped)

# 2. Clustering Techniques

# Selecting numerical features for clustering (assuming 'raisedhands' is the proxy for study time, and 'VisITedResources' for engagement)
features = ['raisedhands', 'VisITedResources', 'Discussion', 'AnnouncementsView']
X = data[features]

# K-Means Clustering+
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Visualizing the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='raisedhands', y='VisITedResources', hue='Cluster', palette='Set1')
plt.title('Clustering Students Based on Study Habits and Engagement')
plt.show()

# 3. Testing the Data - T-Test to check performance difference based on participation
# Assuming 'ParentschoolSatisfaction' indicates participation in extracurricular activities, and 'Class' is a categorical grade
data['Class_numeric'] = data['Class'].apply(lambda x: 1 if x == 'H' else (2 if x == 'M' else 3))

# Split data based on extracurricular activity participation
group_1 = data[data['ParentschoolSatisfaction'] == 'Good']['Class_numeric']
group_2 = data[data['ParentschoolSatisfaction'] == 'Bad']['Class_numeric']

# Perform T-Test
t_stat, p_value = ttest_ind(group_1, group_2)

print("\nT-Test Results:")
print(f"T-statistic: {t_stat}, P-value: {p_value}")

# Check if the difference is statistically significant
if p_value < 0.05:
    print("The difference in performance based on extracurricular participation is statistically significant.")
else:
    print("No significant difference in performance based on extracurricular participation.")
