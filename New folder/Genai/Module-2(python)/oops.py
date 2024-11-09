from abc import ABC, abstractmethod

# 1. Encapsulation: The base class with protected attributes
class Person(ABC):
    def __init__(self, name, age):
        self._name = name  # Protected attribute
        self._age = age    # Protected attribute
    
    @abstractmethod
    def get_role(self):
        pass

# 2. Inheritance: Student and Teacher inherit from Person
class Student(Person):
    def __init__(self, name, age, grade):
        super().__init__(name, age)  # Calling the parent class constructor
        self.__grade = grade  # Encapsulated attribute
    
    # Method for getting student-specific role (Polymorphism)
    def get_role(self):
        return "Student"
    
    # Encapsulated method to access grade
    def get_grade(self):
        return self.__grade
    
    # Method to show polymorphism (Overriding)
    def get_details(self):
        return f"Name: {self._name}, Age: {self._age}, Role: {self.get_role()}, Grade: {self.__grade}"

class Teacher(Person):
    def __init__(self, name, age, subject):
        super().__init__(name, age)
        self.__subject = subject  # Encapsulated attribute
    
    # Method for getting teacher-specific role (Polymorphism)
    def get_role(self):
        return "Teacher"
    
    # Encapsulated method to access subject
    def get_subject(self):
        return self.__subject
    
    # Method to show polymorphism (Overriding)
    def get_details(self):
        return f"Name: {self._name}, Age: {self._age}, Role: {self.get_role()}, Subject: {self.__subject}"

# 3. Polymorphism in action: A common method for different classes
def display_details(person):
    print(person.get_details())

# 4. Abstraction: Forcing classes to implement get_role method
# Main program demonstrating the concepts

student1 = Student("Lahari", 13, 85)  # Creating a Student object
teacher1 = Teacher("Mr. Sharma", 35, "Math")  # Creating a Teacher object

# Using polymorphism to call the same method on different objects
display_details(student1)  # This will call the student's get_details method
display_details(teacher1)  # This will call the teacher's get_details method
