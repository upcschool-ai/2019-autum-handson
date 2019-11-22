import itertools
import random


class DataDistribution:
    def __init__(self, W=None, b=None):
        self.W = W or random.uniform(-5, 5)
        self.b = b or random.uniform(-5, 5)

    def generate(self, num_iters=None):
        # This uses the `generate_sample` method and wraps it with a generator for ease of use
        for step in itertools.count(0, 1):
            if num_iters is not None and num_iters == step:
                break
            yield self.generate_sample()

    def generate_sample(self):
        # This is probably what you have implemented. It returns a single data point
        x = random.uniform(-200, 200)
        y = self.W * x + self.b
        return x, y

    def __call__(self, num_iters=None):
        return self.generate(num_iters=num_iters)
