from derived_class import DerivedClass

def main():
    # Create an instance of the derived class
    person = DerivedClass("Alice", 30)
    # Call a method that uses both base and derived class attributes/methods
    person.display_info()

if __name__ == "__main__":
    main()
