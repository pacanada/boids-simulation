from abc import ABC, abstractmethod


class Simulation(ABC):
    @abstractmethod
    def update(self, dt):
        pass

    @abstractmethod
    def draw(
        self,
    ):
        pass
