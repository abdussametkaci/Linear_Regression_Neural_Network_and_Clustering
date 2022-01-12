class Point:
    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def __str__(self):
        return f"(x: {self.x}, y: {self.y})"

    def to_tuple(self):
        return self.x, self.y

    def __iter__(self):
        for i in self.__dict__.values():
            yield i
