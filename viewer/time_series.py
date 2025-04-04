import numpy as np
import functools

import napari
from PyQt5.QtCore import QTimer
from magicgui import magicgui
from magicgui.widgets import Container, CheckBox, PushButton
CHUNK_SIZE = 10000

class NwbTimeSeriesViewer:

    def __init__(self, nwb_obj, interval_range, n_plot=1000, max_channels=10,
                 channel_index=None, image_shape = (512, 512), channel_options=None):
        """Initialize the time series viewer.

        Parameters
        ----------
        nwb_obj : object
            The NWB object in an open nwb file containing the data.
        interval_range : tuple
            The start and end times of the interval to display.
        n_plot : int
            The number of data points to plot.
        max_channels : int
            The maximum number of channels to display.
        channel_index : list, optional
            The indices of the channels to display. If None, the first max_channels's will be displayed.

        """

        self.interval_range = interval_range
        self.max_channels = max_channels
        self.channel_index = channel_index[:max_channels] if channel_index is not None else np.arange(max_channels)
        self.n_plot = n_plot
        self.image_shape = image_shape

        self.channel_options = channel_options if channel_options is not None else np.arange(nwb_obj.data.shape[1])

        self.data_timestamps = nwb_obj.timestamps
        self.data = nwb_obj.data
        self.viewer = napari.Viewer()

        self.ind_start = np.searchsorted(
            self.data_timestamps, interval_range[0]
        )
        self.ind_end = np.searchsorted(
            self.data_timestamps, interval_range[1]
        )
        self.data_timestamps = self.data_timestamps[self.ind_start : self.ind_end]
        self.sample_rate = int(1 / np.median(
            np.diff(self.data_timestamps)
        )) # TODO: estimate this better

    def compile(self):
        # initial display settings
        window = int(1 * self.sample_rate) # 1 second window
        step = int(window / self.n_plot)  # step size for subsampling
        loc = window + self.ind_start

        # electrode_index = np.arange(self.max_electrodes)
        # display location info for the eseires data
        self.tracks_scale = self.image_shape[0] / (self.max_channels) / 1200
        self.tracks_spacing = self.image_shape[0] / (self.max_channels)
        # make the tracks data and layers
        # self.tracks_layers = []
        self.tracks_data = np.ones((self.n_plot * self.max_channels, 4))
        for i in range(self.max_channels):
            ind_track = np.arange(self.n_plot) + i * self.n_plot
            # const x data
            self.tracks_data[ind_track, 3] = np.linspace(-300, -20, self.n_plot)
            # electrode index
            self.tracks_data[ind_track, 0] = i
            # data
            self.tracks_data[ind_track, 2] = (
                np.ones(self.n_plot) * self.tracks_scale + self.tracks_spacing * i # null initialization
            )
        # add the layer
        self.tracks_layer = self.viewer.add_tracks(self.tracks_data, name="channel " + str(self.channel_index[i]))
        # self.update(0, 1)

    def run(self):
        # PLAYBACK_REFRESH = 3 # ms
        # gui tools
        playback_scale = 100
        self.play_state = {
            "is_playing": False,
            "timer": QTimer()
        }

        @magicgui(
            auto_call=True,
            time={"widget_type": "FloatSlider", "max": self.interval_range[1]-self.interval_range[0],},
            window_scale={
                "widget_type": "FloatSlider",
                "max": 3,
                "min": 0.1,
                "step": 0.1,
                "value": 1,
            },
            play={"widget_type": "PushButton"},
        )
        def my_widget(time: float, play: bool = False, window_scale: float = 1):
            window = window_scale#int(window_scale)
            self.update(time+self.data_timestamps[0], window=window)

        """
        @magicgui(
            auto_call=False,
            call_button="Apply",
            channel_picker={
                "widget_type": "Select",
                "choices": list(self.channel_options),  # e.g., np.arange(data.shape[1]) or your custom list
                "allow_multiple": True,
                "label": "Pick Channels",
            }
        )
        def channel_widget(channel_picker=()):
            # Convert channel_picker (which is a tuple of items) to a list
            selected_channels = list(channel_picker)

            # Limit to self.max_channels if user selects too many
            self.channel_index = selected_channels[: self.max_channels]

            # Clear the cached data so future calls see the new channels
            self.get_data_chunk.cache_clear()

            # # Rebuild the layer data with the updated channels
            # self.compile()
        """
        # playback tools
        def update_slider():
            my_widget.time.value = (my_widget.time.value + 0.05) % (60 * 20)

        def poll_state():
            if self.play_state["is_playing"]:
                update_slider()

        self.play_state["timer"].timeout.connect(poll_state)

        def toggle_play(event):
            self.play_state["is_playing"] = not self.play_state["is_playing"]
            if self.play_state["is_playing"]:
                self.play_state["timer"].start(3)
                my_widget.play.text = "Pause"
            else:
                self.play_state["timer"].stop()
                my_widget.play.text = "Play"

        # finish defining the playback widget
        my_widget.play.changed.connect(toggle_play)
        # create the channel selection widget, with link to the payback widget for update
        channel_widget = self.create_channel_checkbox_widget(my_widget)
        # stick the widgets in the viewer window and run
        self.viewer.window.add_dock_widget(channel_widget.native)
        self.viewer.window.add_dock_widget(my_widget.native)
        self.viewer.window.add_dock_widget(channel_widget.native)
        napari.run()


    def update(
    self,
    time,
    window,
    **kwargs,
    ):
        # update e-series
        # window = int(
        #     np.round(window * self.sample_rate / self.n_plot) * self.n_plot
        # )  # ensure clean subsampled number
        # loc = np.digitize(time, self.data_timestamps)
        # loc = np.searchsorted(self.data_timestamps, time) + self.ind_start
        # print(window)
        frame_data = self.get_data(time, window)
        # print(frame_data.shape)

        ind_track = np.arange(self.n_plot)
        for i in range(frame_data.shape[1]):
            shift = i * self.n_plot
            if frame_data.shape[0] < ind_track.shape[0]:
                self.tracks_data[ind_track[-frame_data.shape[0]]+shift :, 2] = (
                    frame_data[:, i] * self.tracks_scale + self.tracks_spacing * i
                )
            else:
                self.tracks_data[ind_track+shift, 2] = (
                    frame_data[:, i] * self.tracks_scale + self.tracks_spacing * i
                )
        self.tracks_layer.data = self.tracks_data
        self.tracks_layer.refresh()


    # data streaming tools
    def get_data(self, time, window):
        """ load data from the nwb object. Uses chunked caching to speed up return
        when calling in sequence.

        Parameters
        ----------
        loc : int
            The location of the data to load.
        window : int
            The window around loc to load. (in seconds)
        """
        target_times = np.linspace(time-window, time+window, self.n_plot)

        # 2) Find the nearest indices in nwb_timestamps for each target time.
        idx_array = np.searchsorted(self.data_timestamps, target_times, side="left")
        # Make sure indices don’t go out of bounds:
        # idx_array[idx_array >= len(self.data_timestamps)] = len(self.data_timestamps) - 1
        idx_array = np.clip(idx_array, 0, len(self.data_timestamps) - 1)
        idx_array += self.ind_start
        ind_st, ind_end = idx_array[0], idx_array[-1]
        # print(ind_st, ind_end)

        chunk_index_st = self.get_chunk_index(ind_st)
        chunk_index_end = self.get_chunk_index(ind_end)
        if chunk_index_st == chunk_index_end:
            return self.get_data_chunk(chunk_index_st)[
                self.get_chunk_frame_index(ind_st) : self.get_chunk_frame_index(ind_end)
            ][idx_array -idx_array[0]-1]

        return np.concatenate(
            [
                self.get_data_chunk(chunk_index_st)[
                    self.get_chunk_frame_index(ind_st) :
                ]
            ]
            + [
                self.get_data_chunk(ci)[
                    : (
                        CHUNK_SIZE
                        if not ci == chunk_index_end
                        else self.get_chunk_frame_index(ind_end)
                    )
                ]
                for ci in range(chunk_index_st + 1, chunk_index_end + 1)
            ],
            axis=0,
        )[idx_array -idx_array[0]-1]

    @functools.lru_cache(maxsize=int(1e6//CHUNK_SIZE))
    def get_data_chunk(self, chunk_index):
        return self.data[
            CHUNK_SIZE * chunk_index : CHUNK_SIZE * (chunk_index + 1),
            self.channel_index,
        ]

    # data streaming tools
    @staticmethod
    def get_chunk_index(ind):
        return int(ind / CHUNK_SIZE)

    @staticmethod
    def get_chunk_frame_index(ind):
        return int(ind % CHUNK_SIZE)


    #----------- Widget Creation -------------------
    def create_channel_checkbox_widget(self, play_widget):
            container = Container(layout="vertical")
            # Create a checkbox for each channel option.
            for ch in list(self.channel_options):
                # Each checkbox is pre-set True if the channel is in the current selection.
                cb = CheckBox(text=str(ch), value=(ch in self.channel_index))
                container.append(cb)
            # Create an Apply button
            apply_btn = PushButton(text="Apply")
            container.append(apply_btn)

            def apply_changes(event):
                # Gather the channels that are checked
                new_indexes = [int(w.text) for w in container
                            if isinstance(w, CheckBox) and w.value]
                # Limit to max_channels if too many are selected
                self.channel_index = new_indexes[: self.max_channels]
                # Clear the cached data so that new channels are used
                self.get_data_chunk.cache_clear()
                # Rebuild the layer data with the updated channel indices
                self.update(
                    play_widget.time.value + self.data_timestamps[0],
                    play_widget.window_scale.value,
                )
                # self.compile()

            apply_btn.changed.connect(apply_changes)
            return container
