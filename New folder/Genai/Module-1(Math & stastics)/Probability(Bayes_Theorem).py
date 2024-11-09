import numpy as np
from scipy.stats import norm

# Step 1: Define the parameters
mean_score = 65  # Mean score of students
std_dev = 10  # Standard deviation of scores
total_students = 200  # Total number of students

# Step 2: Probability of failing (score < 50)
fail_score = 50
z_fail = (fail_score - mean_score) / std_dev

# Using the cumulative distribution function (CDF) to find the probability of failing
prob_fail = norm.cdf(z_fail)

# Expected number of students failing
expected_fail = prob_fail * total_students

print(f"Probability of failing (score below 50): {prob_fail:.4f}")
print(f"Expected number of students failing: {int(expected_fail)}")

# Step 3: Cumulative probability for scoring between 50 and 80 (Pass)
pass_lower_bound = 50
pass_upper_bound = 80

# Z-scores for the lower and upper bounds
z_lower = (pass_lower_bound - mean_score) / std_dev
z_upper = (pass_upper_bound - mean_score) / std_dev

# Cumulative probability between 50 and 80
prob_pass = norm.cdf(z_upper) - norm.cdf(z_lower)
expected_pass = prob_pass * total_students

print(f"Cumulative probability for scoring between 50 and 80 (Pass): {prob_pass:.4f}")
print(f"Expected number of students passing: {int(expected_pass)}")

# Step 4: Using Bayes' Theorem to calculate conditional probabilities
# Let's assume:
# P(Fail | Score < 50) = prob_fail (From previous calculation)
# P(Pass | Score between 50 and 80) = prob_pass

# Bayes Theorem: P(A|B) = (P(B|A) * P(A)) / P(B)

# Assuming prior probabilities based on historical data or assumptions
prior_fail = 0.30  # Probability that students fail overall
prior_pass = 0.70  # Probability that students pass overall

# P(Score < 50 | Fail) ~ 1 (because failing students tend to score below 50)
P_score_fail = 1.0

# P(Score between 50 and 80 | Pass) ~ 1 (because passing students tend to score in this range)
P_score_pass = 1.0

# Applying Bayes' Theorem
posterior_fail = (P_score_fail * prior_fail) / prob_fail
posterior_pass = (P_score_pass * prior_pass) / prob_pass

print(f"\nBayes Theorem Results:")
print(f"Posterior probability of failing given score < 50: {posterior_fail:.4f}")
print(f"Posterior probability of passing given score between 50 and 80: {posterior_pass:.4f}")
