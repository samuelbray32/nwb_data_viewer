import numpy as np
import functools

import napari
from PyQt5.QtCore import QTimer
from magicgui import magicgui
from magicgui.widgets import Container, CheckBox, PushButton

from .viewer_mixin import ViewerMixin
from .time_series import NwbTimeSeriesViewer

class NwbSpatialSeriesViewer(NwbTimeSeriesViewer):
    def __init__(self,
                 nwb_obj,
                 interval_range,
                 channel_index=[0, 1],
                 n_plot = 1000,
                 embedded_module=False,
                 ):
        super(NwbTimeSeriesViewer,self).__init__(interval_range)
        self.nwb_obj = nwb_obj
        self.data_timestamps = nwb_obj.timestamps
        self.data = nwb_obj.data

        self.ind_start = np.searchsorted(self.data_timestamps, interval_range[0])
        self.ind_end = np.searchsorted(self.data_timestamps, interval_range[1])
        self.data_timestamps = self.data_timestamps[self.ind_start : self.ind_end]

        self.sample_rate = int(
            1 / np.median(np.diff(self.data_timestamps))
        )  # TODO: estimate this better
        self.channel_index = channel_index
        self.n_plot = n_plot
        self.embedded_module = embedded_module

    def compile(self,viewer=None):
        self.viewer = viewer if viewer else napari.Viewer()
        window = int(1 * self.sample_rate)  # 1 second window
        step = int(window / self.n_plot)  # step size for subsampling
        loc = window + self.ind_start

        self.tracks_data = np.ones((self.n_plot, 4))


        # const x data
        self.tracks_data[:, 3] = np.linspace(-300, -20, self.n_plot)
        # electrode index
        self.tracks_data[:, 0] = 10
        # data
        self.tracks_data[:, 2] = (
            np.ones(self.n_plot) * 100
            + 100 * 5  # null initialization
        )

        if not self.embedded_module:
            self.tracks_layer = self.viewer.add_tracks(
                self.tracks_data,
                name=self.nwb_obj.name,
            )

            self.points_layer = self.viewer.add_points(
                data=np.zeros((1, 2)),
                edge_color='red',
                face_color='red',
                size=20,
            )

    def update(self, time, window, **kwargs):
        frame_data = self.get_data(time, window)
        self.tracks_data[:, 3] = frame_data[:, 0]
        self.tracks_data[:, 2] = frame_data[:, 1]
        self.tracks_data[:, 1] = np.linspace(
            time-window, time+window, self.n_plot
        )
        self.tracks_data[:,:2] = 1

        self.tracks_layer.data = self.tracks_data
        self.tracks_layer.refresh()
        self.points_layer.data = np.flip(frame_data[-1,])

    def add_additional_widgets(self, playback_widget):
        # Placeholder for adding additional widgets. This should be implemented in the subclass.
        pass





