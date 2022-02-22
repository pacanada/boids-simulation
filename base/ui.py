from typing import List
import imgui
from imgui.integrations.pyglet import PygletFixedPipelineRenderer


class UISetting:
    def __init__(
        self,
        dtype: str,
        type: str,
        value: float,
        min: float = None,
        max: float = None,
        step: float = None,
        format: str = None,
        name: str = "No name",
        description: str = "No description",
    ):
        self.dtype = dtype
        self.type = type
        self.value = value
        self.min = min
        self.max = max
        self.step = step
        self.format = format
        self.name = name
        self.description = description
        self.changed = False

    def set_setting(self):
        """Missing some combinations"""
        if self.dtype == "int" and self.type == "input":
            self.changed, self.value = imgui.input_int(self.description, self.value)
        elif self.dtype == "float" and self.type == "slider":
            self.changed, self.value = imgui.slider_float(
                self.description, self.value, self.min, self.max, self.format, self.step
            )
        else:
            raise Exception("Unknown type")
        return self


class UISettings(List):
    def __init__(self, settings: List[UISetting]):
        self.settings = settings
        self._index = 0

    def __iter__(self):
        for setting in self.settings:
            yield setting

    def __setitem__(self, item, value):
        self.settings[item] = value
        return self

    def get_value(self, name: str):
        out = None
        for setting in self.settings:
            if setting.name == name:
                out = setting.value
        if out is None:
            raise KeyError("Setting not found")
        return out

    def get_changed(self, name: str):
        out = None
        for setting in self.settings:
            if setting.name == name:
                out = setting.changed
        if out is None:
            raise KeyError("Setting not found")
        return out


class UI:
    def __init__(
        self, window, settings: UISettings, name: str = "Window", text: str = "None"
    ):
        imgui.create_context()
        self.impl = PygletFixedPipelineRenderer(window)
        imgui.new_frame()
        imgui.end_frame()
        self.settings = settings
        self.name = name
        self.text = text

    def render(self):
        imgui.render()
        self.impl.render(imgui.get_draw_data())
        imgui.new_frame()

        imgui.begin(self.name)
        imgui.text(self.text)
        for index, setting in enumerate(self.settings):
            self.settings[index] = setting.set_setting()

        imgui.end()
        imgui.end_frame()
