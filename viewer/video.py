import napari
import numpy as np
from PyQt5.QtCore import QTimer
import imageio

import os
import glob
import tempfile
import subprocess
import threading
import time

from .viewer_mixin import ViewerMixin

class VideoFramePreloader:
    def __init__(self, video_path):
        """
        Launch ffmpeg asynchronously to extract every frame from video_path into a temp directory.
        The frames are not guaranteed to be ready until ffmpeg completes,
        so calls to get_frame or get_frame_path might find frames missing if ffmpeg hasn't produced them yet.
        """
        self.video_path = video_path

        # Create a dedicated temp directory to store extracted frames
        self.temp_dir = tempfile.mkdtemp(prefix="video_frame_cache_")

        # ffmpeg command to decode every frame into individual .png files
        # -vsync 0 prevents ffmpeg from duplicating or dropping frames
        out_pattern = os.path.join(self.temp_dir, "frame_%010d.jpg")
        command = [
            "ffmpeg",
            "-i", self.video_path,
            "-vsync", "0",
            # "-vf", "scale=640:-1",
            out_pattern
        ]

        # Launch ffmpeg in the background (asynchronously)
        # We store the Popen object so we can check if it's still running
        self.ffmpeg_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Keep track of frame paths. Initially empty; it will fill up as ffmpeg extracts more files.
        self.frame_paths = []

    def _refresh_frame_list(self):
        """
        Updates the internal list of frame paths by checking the temp directory.
        This should be called each time a frame is requested, in case new frames have appeared.
        """
        # If ffmpeg has finished, we have a final set of frames.
        # If ffmpeg is still running, we have a partial set.
        # In either case, we sort them so they remain in correct numerical order.
        current_list = glob.glob(os.path.join(self.temp_dir, "frame_*.jpg"))
        # Sort them numerically based on the frame_XXXXXXXXXX number
        current_list.sort()
        self.frame_paths = current_list

    def is_done_extraction(self):
        """
        Returns True if the ffmpeg process has completed, False if it's still running.
        """
        return (self.ffmpeg_process.poll() is not None)

    def get_frame_path(self, frame_index):
        """
        Returns the file path to the requested frame index.
        If the requested frame isn't available yet, raises a BufferingError.
        """
        # Update the list of frames extracted so far
        self._refresh_frame_list()

        if 0 <= frame_index < len(self.frame_paths):
            return self.frame_paths[frame_index]
        else:
            # Check if ffmpeg is completely done. If done, it means frame_index is out of range.
            if self.is_done_extraction():
                # If out of range and we're done extracting, the user has requested a frame that doesn't exist
                return None
            else:
                # If ffmpeg is not done, we might be waiting on frames that haven't been created yet
                Warning(f"Buffering: Frame {frame_index} not yet extracted. Still buffering...")

    def get_frame(self, frame_index):
        """
        Loads and returns the image data for the requested frame (as a numpy array).
        Raises BufferingError if the requested frame is not yet available.
        Returns None if the index is out of range but extraction is complete.
        """
        # path = self.get_frame_path(frame_index)
        # if path is None:
        #     # That means ffmpeg finished, but frame_index doesn't exist
        #     return None

        # If we got a valid path, load the image
        path = f"{self.temp_dir}/frame_{frame_index:010d}.jpg"
        if not os.path.exists(path):
            raise ValueError(f"Frame {frame_index} not yet available. Still buffering...")
        return imageio.imread(path)

    def total_frames(self):
        """
        If ffmpeg is still running, returns how many frames have been extracted so far.
        If it's done, returns the total number of frames in the video.
        """
        self._refresh_frame_list()
        return len(self.frame_paths)


class NwbVideoViewer(ViewerMixin):
    def __init__(self, video_path, interval_range, frame_timestamps):
        super().__init__(interval_range)
        self.video_path = video_path
        self.frame_timestamps = frame_timestamps
        # Start the background preloader (asynchronous extraction)
        self.loader = VideoFramePreloader(video_path)
        self.current_frame = 1

        # Create a napari viewer
        self.viewer = napari.Viewer()


    def compile(self):
        # Initialize an empty image layer (e.g., 3-channel color)
        null_image = None
        while null_image is None:
            # Attempt to load the first frame
            try:
                null_image = self.loader.get_frame(self.current_frame)
            except ValueError as e:
                # If the frame is not yet available, wait and retry
                print(f"Buffering: {e}")
                time.sleep(0.1)
        image_layer = self.viewer.add_image(null_image, rgb=True, name="video")

    def run(self):
        playback_widget = self.create_playback_widget()
        self.viewer.window.add_dock_widget(playback_widget.native)
        napari.run()

    def update(self, time, window):
        new_frame_index = np.searchsorted(self.frame_timestamps, time)
        new_frame_index = min(new_frame_index, len(self.frame_timestamps) - 1)
        if self.current_frame == new_frame_index:
            return
        self.current_frame = new_frame_index
        # Update the image layer with the new frame
        # Check if the frame is available
        try:
            frame_data = self.loader.get_frame(self.current_frame)
        except ValueError as e:
            # If the frame is not yet available, wait and retry
            print(f"Buffering: {e}")
            return
        # Update the image layer with the new frame
        self.viewer.layers["video"].data = frame_data