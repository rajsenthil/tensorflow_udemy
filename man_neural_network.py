class SimpleClass():
    def __init__(self):
        print("hello")

    def yell(self):
        print("Yelling")

x = SimpleClass()

print(x.yell())

class ExtendedClass(SimpleClass):
    def __init__(self):
        super().__init__()
        print("Extended")

y = ExtendedClass()
y.yell()

