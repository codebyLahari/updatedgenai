import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Constants
total_students = 200
pass_rate = 0.85  # Assumed 85% pass rate
fail_rate = 1 - pass_rate
mean_score = 70  # Assumed mean score
std_deviation = 10  # Assumed standard deviation
pass_threshold = 50
distinction_threshold = 85

# 1. Bernoulli Distribution - Pass/Fail Analysis
# Probability of passing or failing
bernoulli_pass_fail = stats.bernoulli(pass_rate)
pass_probability = bernoulli_pass_fail.pmf(1)  # Probability of passing
fail_probability = bernoulli_pass_fail.pmf(0)  # Probability of failing

print("Bernoulli Distribution:")
print(f"Probability of passing: {pass_probability:.2f}")
print(f"Probability of failing: {fail_probability:.2f}")

# 2. Binomial Distribution - Number of Students Expected to Pass
# Binomial distribution to find expected number of students passing
students_passing_prob = stats.binom.pmf(170, total_students, pass_rate)

print("\nBinomial Distribution:")
print(f"Probability of exactly 170 students passing: {students_passing_prob:.4f}")

# 3. Normal Distribution - Score Distribution Analysis
# Z-scores and probabilities for various thresholds
z_fail = (pass_threshold - mean_score) / std_deviation
z_distinction = (distinction_threshold - mean_score) / std_deviation

fail_prob = stats.norm.cdf(z_fail)
distinction_prob = 1 - stats.norm.cdf(z_distinction)

# Mean, Median, Mode
mean = mean_score
median = mean_score  # For normal distribution, mean = median = mode
mode = mean_score

print("\nNormal Distribution (Score Analysis):")
print(f"Probability of failing (score below 50): {fail_prob:.2f}")
print(f"Probability of scoring distinction (above 85): {distinction_prob:.2f}")
print(f"Mean: {mean}, Median: {median}, Mode: {mode}")

# Plot Normal Distribution
x = np.linspace(mean_score - 4*std_deviation, mean_score + 4*std_deviation, 1000)
y = stats.norm.pdf(x, mean_score, std_deviation)

plt.plot(x, y, label="Normal Distribution")
plt.axvline(x=pass_threshold, color='r', linestyle='--', label='Pass Threshold (50)')
plt.axvline(x=distinction_threshold, color='g', linestyle='--', label='Distinction Threshold (85)')
plt.title("Normal Distribution of Exam Scores")
plt.xlabel("Scores")
plt.ylabel("Probability Density")
plt.legend()
plt.grid()
plt.show()
