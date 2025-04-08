import numpy as np
import functools

import napari
from PyQt5.QtCore import QTimer
from magicgui import magicgui
from magicgui.widgets import Container, CheckBox, PushButton


class ViewerMixin:
    def __init__(self, interval_range):
        self.interval_range = interval_range
        self.play_state = {"is_playing": False, "timer": QTimer()}

    def create_playback_widget(self):
        @magicgui(
            auto_call=True,
            time={
                "widget_type": "FloatSlider",
                "max": self.interval_range[1] - self.interval_range[0],
            },
            window_scale={
                "widget_type": "FloatSlider",
                "max": 3,
                "min": 0.1,
                "step": 0.1,
                "value": 1,
            },
            play={"widget_type": "PushButton"},
        )
        def playback_widget(time: float, play: bool = False, window_scale: float = 1):
            window = window_scale
            self.update(time + self.interval_range[0], window=window)

        # playback tools
        def update_slider():
            playback_widget.time.value = (playback_widget.time.value + 0.05) % (60 * 20)

        def poll_state():
            if self.play_state["is_playing"]:
                update_slider()

        self.play_state["timer"].timeout.connect(poll_state)

        def toggle_play(event):
            self.play_state["is_playing"] = not self.play_state["is_playing"]
            if self.play_state["is_playing"]:
                self.play_state["timer"].start(3)
                playback_widget.play.text = "Pause"
            else:
                self.play_state["timer"].stop()
                playback_widget.play.text = "Play"

        # finish defining the playback widget
        playback_widget.play.changed.connect(toggle_play)

        return playback_widget

    def compile(self):
        # Placeholder for the compile function. This should be implemented in the subclass.
        pass

    def update(self, time, window):
        pass  # Placeholder for the update function. This should be implemented in the subclass.

    def run(self):
        """
        Run the viewer with the playback widget.
        This method creates the playback widget and other additional widgets, of the subclass.
        It then adds the playback widget to the viewer's window and starts the napari event loop.
        """
        self.playback_widget = self.create_playback_widget()
        self.add_additional_widgets(self.playback_widget)
        self.viewer.window.add_dock_widget(self.playback_widget.native)
        napari.run()

    def add_additional_widgets(self, playback_widget):
        # Placeholder for adding additional widgets. This should be implemented in the subclass.
        pass


class MultiViewer(ViewerMixin):
    def __init__(self, interval_range, viewer_list):
        super().__init__(interval_range)
        self.viewer = napari.Viewer()
        self.viewer_list = viewer_list

    def update(self, time, window):
        # Update all viewers with the new time and window
        for viewer in self.viewer_list:
            viewer.update(time, window)

    def compile(self):
        # Compile all viewers

        for viewer in self.viewer_list:
            viewer.compile(self.viewer)

    def run(self):
        self.playback_widget = self.create_playback_widget()
        for viewer in self.viewer_list:
            viewer.add_additional_widgets(self.playback_widget)
        self.viewer.window.add_dock_widget(self.playback_widget.native)
        napari.run()
