import pandas as pd

# Load the dataset
data = pd.read_csv(r"C:\Users\lahar\Gen-AI\dataset.csv")

# Clean the column names (remove any leading/trailing spaces)
data.columns = data.columns.str.strip()

# Verify the available columns
print(data.head())
print(data.columns)

# Inspect the columns to find the one representing dropout status
# Assuming 'Target' represents the outcome column for dropout
outcome_column = 'Target'

# Step 1: Calculate prior probabilities
total_students = len(data)
dropout_count = data[outcome_column].value_counts().get(1, 0)  # Assuming '1' indicates dropout
graduate_count = data[outcome_column].value_counts().get(0, 0)  # Assuming '0' indicates graduate

P_dropout = dropout_count / total_students
P_graduate = graduate_count / total_students

print(f"P(Dropout): {P_dropout}, P(Graduate): {P_graduate}")

# Step 2: Calculate conditional probabilities
# Assuming relevant columns: 'Daytime/evening attendance' and 'Previous qualification'
attendance_column = 'Daytime/evening attendance'  # Update based on your dataset
grades_column = 'Curricular units 1st sem (grade)'  # Update based on your dataset

# Conditional probabilities for 'attendance' and 'grades' given 'dropout' (Target=1)
P_attendance_given_dropout = data[data[outcome_column] == 1][attendance_column].mean()
P_grades_given_dropout = data[data[outcome_column] == 1][grades_column].mean()

# Probabilities of 'attendance' and 'grades' overall
P_attendance = data[attendance_column].mean()
P_grades = data[grades_column].mean()

print(f"P(Attendance | Dropout): {P_attendance_given_dropout}")
print(f"P(Grades | Dropout): {P_grades_given_dropout}")

# Step 3: Applying Bayes' theorem
# To find P(Dropout | Attendance, Grades)
specific_attendance = 1  # Example attendance value
specific_grades = 12  # Example grades value

# Calculate conditional probabilities for specific attendance and grades
P_specific_attendance_given_dropout = (data[data[outcome_column] == 1][attendance_column] == specific_attendance).mean()
P_specific_grades_given_dropout = (data[data[outcome_column] == 1][grades_column] == specific_grades).mean()

# Calculate overall probabilities for specific attendance and grades
P_attendance = (data[attendance_column] == specific_attendance).mean()
P_grades = (data[grades_column] == specific_grades).mean()

# Apply Bayes' theorem
P_dropout_given_specific = (P_specific_attendance_given_dropout * P_specific_grades_given_dropout * P_dropout) / (P_attendance * P_grades)

print(f"P(Dropout | Attendance={specific_attendance}, Grades={specific_grades}): {P_dropout_given_specific}")
