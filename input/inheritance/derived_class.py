from base_class import BaseClass

class DerivedClass(BaseClass):
    def __init__(self, name, age):
        super().__init__(name)  # Call the constructor of the base class
        self.age = age

    def display_info(self):
        self.display_name()  # Call a method from the base class
        print(f"Age: {self.age}")
