# Global variables
total_students = 0
total_scores = 0

# Function to update global variables for total students and scores
def update_totals(score_sum, student_count):
    global total_students, total_scores
    total_students += student_count
    total_scores += score_sum

# Function to calculate average score for a student
def calculate_average(*scores):
    return sum(scores) / len(scores) if len(scores) > 0 else 0

# Function to generate a report for each student
def generate_student_report(name, *scores):
    print(f"\nReport for Student: {name}")
    print(f"Scores: {scores}")
    average_score = calculate_average(*scores)
    print(f"Average Score: {average_score}")
    highest_score = max(scores, key=lambda x: x)  # Using lambda to find the highest score
    print(f"Highest Score: {highest_score}")
    
    # Updating global totals
    update_totals(sum(scores), 1)

# Function to get overall average score for all students
def overall_average():
    if total_students == 0:
        return 0
    return total_scores / total_students

# Main function to execute the program
def main():
    print("Welcome to the School Management System")
    
    # Taking input for three students
    generate_student_report("Alice", 85, 90, 78)
    generate_student_report("Bob", 88, 72, 91, 85)
    generate_student_report("Charlie", 80, 77, 85, 89, 90)
    
    # Display overall average for all students
    print("\nOverall Average Score for all students: ", overall_average())

# Calling the main function
main()
