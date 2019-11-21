import itertools
import random


class DataDistribution:
    def __init__(self, W=None, b=None):
        self.W = W or random.uniform(-5, 5)
        self.b = b or random.uniform(-5, 5)

    def generate(self, num_iters=None):
        for step in itertools.count(0, 1):
            if num_iters is not None and num_iters == step:
                break
            x = random.uniform(-200, 200)
            y = self.W * x + self.b
            yield x, y

    def __call__(self, num_iters=None):
        return self.generate(num_iters=num_iters)
