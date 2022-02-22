from typing import List
import numpy as np
from numba import int32, float32, njit  # import the types
from numba.experimental import jitclass
from numba.typed import List

spec = [
    ("x", float32),
    ("y", float32),
    ("dx", float32),
    ("dy", float32),
    ("distance", float32),
]
# super slow
@jitclass(spec)
class Boid(object):
    def __init__(self, x, y, dx, dy, distance):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.distance = distance

    def get_average(self, boids):
        len = 0
        x_avg = 0
        y_avg = 0
        for boid in boids:
            dist2 = (self.x - boid.x) ** 2 + (self.y - boid.y) ** 2
            if dist2 <= self.distance ** 2 and dist2 != 0:
                len += 1
                x_avg += boid.x
                y_avg += boid.y
        return x_avg / len, y_avg / len


class Boid_std(object):
    def __init__(self, x, y, dx, dy, distance):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.distance = distance

    def get_average(self, boids):
        len = 0
        x_avg = 0
        y_avg = 0
        for boid in boids:
            dist2 = (self.x - boid.x) ** 2 + (self.y - boid.y) ** 2
            if dist2 <= self.distance ** 2 and dist2 != 0:
                len += 1
                x_avg += boid.x
                y_avg += boid.y
        return x_avg / len, y_avg / len


def numba_class(n: int):
    boids = [
        Boid(np.random.rand() * 10, np.random.rand() * 10, 0, 0, 10) for _ in range(n)
    ]

    for boid in boids:
        x_avg, y_avg = boid.get_average(boids)


def std_class(n: int):
    boids = [
        Boid_std(np.random.rand() * 10, np.random.rand() * 10, 0, 0, 10)
        for _ in range(n)
    ]

    for boid in boids:
        x_avg, y_avg = boid.get_average(boids)
