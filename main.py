from typing import Tuple
import pyglet
import imgui
import imgui.core
from imgui.integrations.pyglet import PygletFixedPipelineRenderer
import random
import numpy as np
from numba import jit


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


class Simulation:
    def __init__(
        self,
        n_boids,
        width,
        height,
        distance,
        alignment_coef,
        cohesion_coef,
        separation_coef,
        scale_boids,
        max_d,
    ):
        self.n_boids = n_boids
        self.width = width
        self.height = height
        self.distance = distance
        self.scale_boids = scale_boids
        self.alignment_coef = alignment_coef
        self.cohesion_coef = cohesion_coef
        self.separation_coef = separation_coef
        self.max_d = max_d
        self.scale = 1
        self.x = np.random.random(self.n_boids) * self.width
        self.y = np.random.random(self.n_boids) * self.height
        self.dx = np.random.uniform(
            -self.width / 100, self.width / 100, size=self.x.size
        )
        self.dy = np.random.uniform(
            -self.height / 100, self.height / 100, size=self.x.size
        )
        self.boids = []
        self.batch = pyglet.graphics.Batch()
        for boid_index in range(n_boids):
            self.boids.append(
                pyglet.shapes.Circle(
                    x=self.x[boid_index],
                    y=self.y[boid_index],
                    radius=self.scale_boids,
                    color=(
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                    ),
                    batch=self.batch,
                )
            )

    def update_params(
        self,
        alignment_coef,
        cohesion_coef,
        separation_coef,
        scale_boids,
        max_d,
        distance,
    ):
        self.alignment_coef = alignment_coef
        self.cohesion_coef = cohesion_coef
        self.separation_coef = separation_coef
        self.scale_boids = scale_boids
        self.max_d = max_d
        self.distance = distance

    def update(self, dt):
        self.x, self.y, self.dx, self.dy = update_vectors(
            x=self.x,
            y=self.y,
            dx=self.dx,
            dy=self.dy,
            distance=self.distance,
            c_a=self.alignment_coef,
            c_c=self.cohesion_coef,
            c_s=self.separation_coef,
            dt=dt,
            max_d=self.max_d,
            width=self.width,
            height=self.height,
        )

        for boid_index in range(self.n_boids):
            self.boids[boid_index].x = self.x[boid_index]
            self.boids[boid_index].y = self.y[boid_index]

    def draw(self):
        self.batch.draw()


class UI:
    def __init__(self, window):
        imgui.create_context()
        self.impl = PygletFixedPipelineRenderer(window)
        imgui.new_frame()
        imgui.end_frame()

        # Window variables
        self.n_boids = 500
        self.scale_boids = 2
        self.distance = 10
        self.alignment_coef = 0.8
        self.cohesion_coef = 0.8
        self.separation_coef = 0.8
        self.max_d = 100

    def render(self):
        imgui.render()
        self.impl.render(imgui.get_draw_data())
        imgui.new_frame()

        imgui.begin("Window")
        imgui.text("This is the test window.")
        self.n_boids_changed, self.n_boids = imgui.input_int(
            "Number of boids", self.n_boids
        )
        self.scale_boids_changed, self.scale_boids = imgui.input_float(
            "Scale of boids", self.scale_boids
        )
        self.distance_changed, self.distance = imgui.input_float(
            "Distance", self.distance
        )
        self.alignment_changed, self.alignment_coef = imgui.slider_float(
            "alignment coef", self.alignment_coef, 0.0, 2.0, "%.2f", 1.0
        )
        self.cohesion_changed, self.cohesion_coef = imgui.slider_float(
            "cohesion coef", self.cohesion_coef, 0.0, 2.0, "%.2f", 1.0
        )
        self.separation_changed, self.separation_coef = imgui.slider_float(
            "separation coef", self.separation_coef, 0.0, 2.0, "%.2f", 1.0
        )
        self.max_d_changed, self.max_d = imgui.input_float("max vel", self.max_d)

        imgui.end()

        imgui.end_frame()


class App(pyglet.window.Window):
    def __init__(self, width, height):
        super().__init__(width, height, "Boids", resizable=True)
        self.set_vsync(False)
        pyglet.clock.schedule_interval(self.update, 1 / 60)
        self.fps_display = pyglet.window.FPSDisplay(window=self)
        self.UI = UI(self)
        self.simulation = Simulation(
            width=self.width,
            height=self.height,
            scale_boids=self.UI.scale_boids,
            n_boids=self.UI.n_boids,
            distance=self.UI.distance,
            alignment_coef=self.UI.alignment_coef,
            cohesion_coef=self.UI.cohesion_coef,
            separation_coef=self.UI.separation_coef,
            max_d=self.UI.max_d,
        )

    def on_draw(self):
        self.fps_display.draw()
        self.simulation.draw()

    def update(self, dt):
        self.clear()
        self.UI.render()
        if (
            self.UI.scale_boids_changed
            or self.UI.distance_changed
            or self.UI.alignment_changed
            or self.UI.cohesion_changed
            or self.UI.max_d_changed
            or self.UI.separation_changed
        ):
            self.simulation.update_params(
                alignment_coef=self.UI.alignment_coef,
                cohesion_coef=self.UI.cohesion_coef,
                separation_coef=self.UI.separation_coef,
                scale_boids=self.UI.scale_boids,
                max_d=self.UI.max_d,
                distance=self.UI.distance,
            )
        elif self.UI.n_boids_changed:
            self.simulation = Simulation(
                width=self.width,
                height=self.height,
                scale_boids=self.UI.scale_boids,
                n_boids=self.UI.n_boids,
                distance=self.UI.distance,
                alignment_coef=self.UI.alignment_coef,
                cohesion_coef=self.UI.cohesion_coef,
                separation_coef=self.UI.separation_coef,
                max_d=self.UI.max_d,
            )
        else:
            self.simulation.update(dt)
            self.simulation.draw()


app = App(1000, 800)
pyglet.app.run()
