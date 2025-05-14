import numpy as np
import functools

import napari
from PyQt5.QtCore import QTimer
from magicgui import magicgui
from magicgui.widgets import Container, CheckBox, PushButton

from .viewer_mixin import ViewerMixin

CHUNK_SIZE = 10000


class NwbTimeSeriesViewer(ViewerMixin):
    def __init__(
        self,
        nwb_obj,
        interval_range,
        n_plot=1000,
        max_channels=10,
        channel_index=None,
        image_shape=(512, 512),
        channel_options=None,
        channel_groups: dict = None,
    ):
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
        channel_options : list, optional
            The list of available channels to display. If None, all channels will be displayed.
        image_shape : tuple
            The shape of the image to display. Default is (512, 512).
        channel_groups: dict, optional
            A dictionary where keys are channel groups and values are lists of channel indices.

        """
        super().__init__(interval_range)

        self.max_channels = max_channels
        self.channel_index = (
            channel_index[:max_channels]
            if channel_index is not None
            else np.arange(max_channels)
        )
        self.n_plot = n_plot
        self.image_shape = image_shape

        self.channel_options = (
            channel_options
            if channel_options is not None
            else (np.arange(nwb_obj.data.shape[1]) if len(nwb_obj.data.shape) > 1 else [0])
        )
        self.channel_groups = channel_groups
        self.update_displayed_groups()

        self.data_timestamps = nwb_obj.timestamps
        self.data = nwb_obj.data

        self.ind_start = np.searchsorted(self.data_timestamps, interval_range[0])
        self.ind_end = np.searchsorted(self.data_timestamps, interval_range[1])
        self.data_timestamps = self.data_timestamps[self.ind_start : self.ind_end]
        self.sample_rate = int(
            1 / np.median(np.diff(self.data_timestamps))
        )  # TODO: estimate this better

    def update_displayed_groups(self):
        if self.channel_groups is None:
            self.displayed_groups = None
            return

        self.displayed_groups = [
            group
            for group in self.channel_groups.keys()
            if any(ch in self.channel_index for ch in self.channel_groups[group])
        ]

    def compile(self, viewer=None):
        self.viewer = viewer if viewer else napari.Viewer()
        # initial display settings
        window = int(1 * self.sample_rate)  # 1 second window
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
                np.ones(self.n_plot) * self.tracks_scale
                + self.tracks_spacing * i  # null initialization
            )
        # add the layer
        self.tracks_layer = self.viewer.add_tracks(
            self.tracks_data,
            name="channel " + str(self.channel_index[i]),
        )
        self.set_properties()

    def add_additional_widgets(self, playback_widget):
        # create the channel selection widget, with link to the payback widget for update
        if self.channel_groups is None:
            channel_widget = self.create_channel_checkbox_widget_no_groups(
                playback_widget
            )
        else:
            channel_widget = self.create_channel_checkbox_widget_groups(playback_widget)

        self.viewer.window.add_dock_widget(channel_widget.native)

    def update(
        self,
        time,
        window,
        **kwargs,
    ):
        frame_data = self.get_data(time, window)

        ind_track = np.arange(self.n_plot)
        for i in range(frame_data.shape[1]):
            shift = i * self.n_plot
            if frame_data.shape[0] < ind_track.shape[0]:
                self.tracks_data[ind_track[-frame_data.shape[0]] + shift :, 2] = (
                    frame_data[:, i] * self.tracks_scale + self.tracks_spacing * i
                )
            else:
                self.tracks_data[ind_track + shift, 2] = (
                    frame_data[:, i] * self.tracks_scale + self.tracks_spacing * i
                )
        if (j_points := ind_track[-1] + shift) < self.tracks_data.shape[0] - 1:
            self.tracks_data[j_points + 1 :, 2] = -100
        self.tracks_layer.data = self.tracks_data
        self.set_properties()
        self.tracks_layer.refresh()

    # data streaming tools
    def get_data(self, time, window):
        """load data from the nwb object. Uses chunked caching to speed up return
        when calling in sequence.

        Parameters
        ----------
        loc : int
            The location of the data to load.
        window : int
            The window around loc to load. (in seconds)
        """
        target_times = np.linspace(time - window, time + window, self.n_plot)

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
            ][idx_array - idx_array[0] - 1]

        return np.concatenate(
            [self.get_data_chunk(chunk_index_st)[self.get_chunk_frame_index(ind_st) :]]
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
        )[idx_array - idx_array[0] - 1]

    @functools.lru_cache(maxsize=int(1e6 // CHUNK_SIZE))
    def get_data_chunk(self, chunk_index):
        if len(self.data.shape) == 1:
            return self.data[
                CHUNK_SIZE * chunk_index : CHUNK_SIZE * (chunk_index + 1)
            ][:,None]
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

    # ----------- Widget Creation -------------------
    def create_channel_checkbox_widget_no_groups(self, play_widget):
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
            new_indexes = [
                int(w.text) for w in container if isinstance(w, CheckBox) and w.value
            ]
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

    def create_channel_checkbox_widget_groups(self, play_widget):
        container = Container(layout="vertical")
        # Create a checkbox for each channel option.
        for group in list(self.channel_groups):
            # Each checkbox is pre-set True if the channel is in the current selection.
            gr = CheckBox(text=str(group), value=(group in self.channel_groups))
            container.append(gr)
        # Create an Apply button
        apply_btn = PushButton(text="Apply")
        container.append(apply_btn)

        def apply_changes(event):
            # Gather the channels that are checked
            new_groups = [
                int(w.text) for w in container if isinstance(w, CheckBox) and w.value
            ]
            # Limit to max_channels if too many are selected
            display_channels = list(
                np.concatenate([self.channel_groups[group] for group in new_groups])
            )[: self.max_channels]

            self.channel_index = display_channels
            self.update_displayed_groups()

            # Clear the cached data so that new channels are used
            self.get_data_chunk.cache_clear()
            # Rebuild the layer data with the updated channel indices
            self.update(
                play_widget.time.value + self.data_timestamps[0],
                play_widget.window_scale.value,
            )

        apply_btn.changed.connect(apply_changes)
        return container

    def set_properties(self, set_color=True):
        properties = self.tracks_layer.properties
        color_prop = "track_id"
        if self.channel_groups is not None:
            properties = {**properties, **self.get_group_property()}
            color_prop = "group"
        self.tracks_layer.properties = properties
        # Set the color property
        if set_color:
            self.tracks_layer.color_by = color_prop
        return properties

    def get_group_property(self):
        if self.channel_groups is None:
            return

        # Create a new group_prop array based on updated self.channel_index
        group_prop = np.ones(self.n_plot * self.max_channels, dtype=int) * -1
        group_lookup = {}

        # Update the group lookup for the currently selected channels
        for group_num, (group, ch_list) in enumerate(self.channel_groups.items()):
            for ch in ch_list:
                group_lookup[ch] = group_num

        # Update group_prop based on the new channel selection
        for i, ch in enumerate(self.channel_index):
            group_num = group_lookup.get(
                ch, -1
            )  # Default to -1 if channel isn't in any group
            group_prop[i * self.n_plot : (i + 1) * self.n_plot] = group_num

        properties = {"group": group_prop}
        return properties


class NwbDigitalSeriesViewer(ViewerMixin):
    def __init__(self, nwb_obj, interval_range):
        super().__init__(interval_range)
        self.n_plot = 1000
        self.nwb_obj = nwb_obj


        self.timestamps = nwb_obj.timestamps
        self.ind_start = np.searchsorted(self.timestamps, interval_range[0])
        self.ind_end = np.searchsorted(self.timestamps, interval_range[1])
        self.initial_state = self.nwb_obj.data[0] if self.ind_start == 0 else self.nwb_obj.data[self.ind_start - 1]
        self.timestamps = self.timestamps[self.ind_start : self.ind_end]

    def compile(self, viewer=None):
        self.viewer = viewer if viewer else napari.Viewer()
        self.tracks_data = np.zeros((self.n_plot,4))
        self.tracks_data[:,3] = np.linspace(-300, -20, self.n_plot)
        self.tracks_layer = self.viewer.add_tracks(
            self.tracks_data,
            name=self.nwb_obj.name,
        )

    def update(
        self,
        time,
        window,
        **kwargs,
    ):
        t_window = np.linspace(time - window, time + window, self.n_plot)
        mark_inds = np.searchsorted(self.timestamps, t_window, side="left")
        mark_inds = np.clip(mark_inds, 0, len(self.timestamps) - 1)
        mark_inds += self.ind_start
        unique_inds = np.unique(mark_inds)
        accessed_values = self.nwb_obj.data[unique_inds]
        # Create a mask for the accessed values
        for ind, val in zip(unique_inds,accessed_values):
            self.tracks_data[mark_inds == ind, 2] = val * 10
        self.tracks_layer.data = self.tracks_data
        self.tracks_layer.refresh()