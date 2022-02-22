from typing import List, Tuple
import random
import numpy as np
from numba import jit
from base.simulation import Simulation
from base.app import App
from base.ui import UI, UISetting, UISettings
from pyglet.window import Window
import pyglet


@jit(nopython=True)
def get_indexes_of_group(
    index_boid: int, x: np.array, y: np.array, distance: float
) -> np.array:
    list_indexes = []
    for index in range(len(x)):
        distance2 = (x[index_boid] - x[index]) ** 2 + (y[index_boid] - y[index]) ** 2
        if (index != index_boid) and (distance2 < distance ** 2):
            list_indexes.append(index)
    return np.array(list_indexes)


@jit(nopython=True)
def get_avg_of_indexes(indexes: np.array, vector: np.array) -> float:
    if len(indexes) == 0:
        return 0
    else:
        return vector[indexes].mean()


@jit(nopython=True)
def get_dist2diff(
    indexes: np.array, boid_index: np.array, x: np.array, y: np.array
) -> Tuple[float, float]:
    dist2diff_list_x = []
    dist2diff_list_y = []
    if len(indexes) == 0:
        return 0, 0
    else:
        for index in indexes:
            distance2 = np.sqrt(
                (x[boid_index] - x[index]) ** 2 + (y[boid_index] - y[index]) ** 2
            )
            diff_pos_x = x[boid_index] - x[index]
            diff_pos_y = y[boid_index] - y[index]
            dist2diff_list_x.append(diff_pos_x / distance2)
            dist2diff_list_y.append(diff_pos_y / distance2)
        return np.array(dist2diff_list_x).mean(), np.array(dist2diff_list_y).mean()


@jit(nopython=True)
def update_vectors(
    x: np.array,
    y: np.array,
    dx: np.array,
    dy: np.array,
    distance: float,
    c_a: float,
    c_c: float,
    c_s: float,
    dt: float,
    max_d: float,
    width: float,
    height: float,
) -> float:
    x_avg = np.zeros(x.size)
    y_avg = np.zeros(x.size)
    dx_avg = np.zeros(x.size)
    dy_avg = np.zeros(x.size)
    dist2diff_x = np.zeros(x.size)
    dist2diff_y = np.zeros(x.size)

    for boid_index in range(len(x)):
        # get average
        flock_indexes = get_indexes_of_group(boid_index, x, y, distance)
        x_avg[boid_index] = get_avg_of_indexes(flock_indexes, x)
        y_avg[boid_index] = get_avg_of_indexes(flock_indexes, y)
        dx_avg[boid_index] = get_avg_of_indexes(flock_indexes, dx)
        dy_avg[boid_index] = get_avg_of_indexes(flock_indexes, dy)
        dist2diff_x[boid_index], dist2diff_y[boid_index] = get_dist2diff(
            flock_indexes, boid_index, x, y
        )

    # check if it is the same dist2 = (x-x_avg)**2+(y-y_avg)**2
    # update vectors
    dx += (c_a * (dx_avg - dx) + c_c * (x_avg - x) + c_s * dist2diff_x * 100) * dt
    x += dx
    dy += (c_a * (dy_avg - dy) + c_c * (y_avg - y) + c_s * dist2diff_y * 100) * dt
    y += dy

    # move to the other side of the window
    x[x > width] -= width
    x[x < 0] += width
    y[y > height] -= height
    y[y < 0] += height

    # limiting speed
    dx[dx > max_d] = max_d
    dx[dx < -max_d] = -max_d
    dy[dy > max_d] = max_d
    dy[dy < -max_d] = -max_d

    return x, y, dx, dy


class SimulationBoids(Simulation):
    def __init__(self, settings: UISettings, height: int, width: int) -> None:
        self.settings = settings
        self.width = width
        self.height = height
        self.init_simulation()

    def init_simulation(self):
        """To be called when any change in the settings is made and we want to launch a new simulation without restarting the app."""
        self.x = np.random.random(self.settings.get_value("n_boids")) * self.width
        self.y = np.random.random(self.settings.get_value("n_boids")) * self.height
        self.dx = np.random.uniform(
            -self.width / 100, self.width / 100, size=self.x.size
        )
        self.dy = np.random.uniform(
            -self.height / 100, self.height / 100, size=self.x.size
        )
        self.boids = []
        self.batch = pyglet.graphics.Batch()
        for boid_index in range(self.settings.get_value("n_boids")):
            self.boids.append(
                pyglet.shapes.Circle(
                    x=self.x[boid_index],
                    y=self.y[boid_index],
                    radius=self.settings.get_value("scale_boids"),
                    color=(
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                    ),
                    batch=self.batch,
                )
            )

    def update(self, dt):
        self.x, self.y, self.dx, self.dy = update_vectors(
            x=self.x,
            y=self.y,
            dx=self.dx,
            dy=self.dy,
            distance=self.settings.get_value("distance"),
            c_a=self.settings.get_value("alignment_coef"),
            c_c=self.settings.get_value("cohesion_coef"),
            c_s=self.settings.get_value("separation_coef"),
            dt=dt,
            max_d=self.settings.get_value("max_d"),
            width=self.width,
            height=self.height,
        )

        for boid_index in range(len(self.boids)):
            self.boids[boid_index].x = self.x[boid_index]
            self.boids[boid_index].y = self.y[boid_index]

    def draw(self):
        if self.settings.get_changed("n_boids"):
            self.init_simulation()
        self.batch.draw()


settings = UISettings(
    [
        UISetting(
            dtype="int",
            type="input",
            value=10,
            name="n_boids",
            description="Number of boids",
        ),
        UISetting(
            dtype="float",
            type="slider",
            value=1,
            min=0,
            max=1000,
            step=1,
            format="%.0f",
            name="distance",
            description="Distance between boids",
        ),
        UISetting(
            dtype="float",
            type="slider",
            value=1,
            min=0,
            max=1,
            step=1,
            format="%.2f",
            name="alignment_coef",
            description="Alignment coef",
        ),
        UISetting(
            dtype="float",
            type="slider",
            value=1,
            min=0,
            max=1,
            step=1,
            format="%.2f",
            name="cohesion_coef",
            description="Cohesion coef",
        ),
        UISetting(
            dtype="float",
            type="slider",
            value=1,
            min=0,
            max=1,
            step=1,
            format="%.2f",
            name="separation_coef",
            description="Separation coef",
        ),
        UISetting(
            dtype="float",
            type="slider",
            value=1,
            min=0,
            max=100,
            step=1,
            format="%.0f",
            name="max_d",
            description="Max speed",
        ),
        UISetting(
            dtype="float",
            type="slider",
            value=1,
            min=0,
            max=5,
            step=1,
            format="%.0f",
            name="scale_boids",
            description="Scale",
        ),
    ]
)

WIDTH = 1000
HEIGHT = 1000
app = App(
    width=WIDTH,
    height=HEIGHT,
    settings=settings,
    simulation=SimulationBoids(width=WIDTH, height=HEIGHT, settings=settings),
    dt=1 / 60,
)
pyglet.app.run()
