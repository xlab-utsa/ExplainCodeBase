"""Python sample."""
import os


class Animal:
    """Base animal."""

    def __init__(self, name):
        self.name = name

    def speak(self):
        return describe(self.name)


class Dog(Animal):
    def __init__(self, name):
        super().__init__(name)

    def bark(self):
        def loud(text):
            return text.upper()

        return loud(self.speak())


def describe(name):
    return os.path.join("animals", name)


if __name__ == "__main__":
    dog = Dog("rex")
    dog.bark()
