import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Step 1: Simulating Data
np.random.seed(42)
n_students = 500

# Simulating scores before and after using the platform
scores_before = np.random.normal(loc=60, scale=10, size=n_students)  # avg 60, std 10
improvement = np.random.normal(loc=7, scale=3, size=n_students)  # avg improvement 7, std 3
scores_after = scores_before + improvement

# Creating DataFrame
data = pd.DataFrame({
    'Student_ID': range(1, n_students + 1),
    'Score_Before': scores_before,
    'Score_After': scores_after,
    'Improvement': improvement
})

# Displaying first few rows of the data
print(data.head())

# Step 2: Point Estimation (Mean Improvement)
mean_improvement = data['Improvement'].mean()
print(f"Point Estimate (Mean Improvement): {mean_improvement:.2f}")

# Step 3: Confidence Interval for the Improvement
confidence_level = 0.95
confidence_interval = stats.t.interval(confidence_level, len(data['Improvement']) - 1, 
                                       loc=mean_improvement, 
                                       scale=stats.sem(data['Improvement']))
print(f"95% Confidence Interval for Improvement: {confidence_interval}")

# Step 4: Hypothesis Testing
# Null Hypothesis: Mean improvement = 0
# Alternative Hypothesis: Mean improvement > 0 (one-tailed test)
t_stat, p_value = stats.ttest_1samp(data['Improvement'], popmean=0)
p_value /= 2  # One-tailed test

alpha = 0.05  # Significance level
if p_value < alpha and t_stat > 0:
    print(f"Reject the null hypothesis (p-value = {p_value:.5f}, t-statistic = {t_stat:.2f})")
    print("Conclusion: The platform significantly improves scores.")
else:
    print(f"Fail to reject the null hypothesis (p-value = {p_value:.5f}, t-statistic = {t_stat:.2f})")
    print("Conclusion: No significant improvement in scores.")

# Step 5: Plotting the Improvement Distribution
plt.hist(data['Improvement'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Score Improvements')
plt.xlabel('Improvement in Scores')
plt.ylabel('Number of Students')
plt.axvline(mean_improvement, color='red', linestyle='dashed', linewidth=1)
plt.show()
