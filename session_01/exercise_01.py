import random


class DataDistribution:
    def __init__(self, W=None, b=None):
        self.W = W or random.uniform(-5, 5)
        self.b = b or random.uniform(-5, 5)

    def generate(self):
        while True:
            x = random.uniform(-200, 200)
            y = self.W * x + self.b
            yield x, y

    def __call__(self):
        return self.generate()
