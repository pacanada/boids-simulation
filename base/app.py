from base.ui import UISettings, UI
from base.simulation import Simulation
import pyglet


class App(pyglet.window.Window):
    def __init__(
        self,
        width: int,
        height: int,
        settings: UISettings,
        simulation: Simulation,
        name: str = "Window",
        dt: float = 1 / 60,
    ):
        super().__init__(width, height, name, resizable=True)
        self.set_vsync(False)
        pyglet.clock.schedule_interval(self.update, dt)
        self.fps_display = pyglet.window.FPSDisplay(window=self)
        self.UI = UI(self, settings)
        self.simulation = simulation

    def on_draw(self):
        self.fps_display.draw()
        self.simulation.draw()

    def update(self, dt):
        self.clear()
        self.UI.render()
        self.simulation.update(dt)
        self.simulation.draw()
