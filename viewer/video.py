import napari
import numpy as np
from magicgui import magicgui
from magicgui.widgets import PushButton
from PyQt5.QtCore import QTimer
import imageio

import os
import glob
import tempfile
import subprocess
import threading
import time
# imageio is needed for get_frame if you want to load the PNG as an array
import imageio

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


def napari_video(video_path):
    # 1) Start the background preloader (asynchronous extraction)
    loader = VideoFramePreloader(video_path)

    # 2) Create a napari viewer
    viewer = napari.Viewer()

    # 3) Initialize an empty image layer (e.g., 3-channel color)
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    image_layer = viewer.add_image(dummy_image, rgb=True, name="video")

    # 4) Build a slider controlling the frame index.
    #    The maximum is set arbitrarily to 1000. You can update it dynamically
    #    by polling loader.total_frames() in a separate thread or QTimer.
    @magicgui(
        auto_call=True,
        frame_index={"widget_type": "Slider", "min": 0, "max": 1000, "step": 1},
    )
    def frame_slider(frame_index: int = 0):
        """
        Called automatically each time the user moves the slider.
        Attempts to display the selected frame.
        """
        try:
            total = loader.total_frames()  # how many frames have been extracted so far
            if frame_index >= total:
                # The user asked for a frame beyond what's extracted so far.
                return
            # Load the requested frame (may raise an error if not yet available)
            frame_data = loader.get_frame(frame_index)
            image_layer.data = frame_data

        except ValueError as e:
            # Custom error if frame not yet extracted
            print(f"Buffering: {e}")
        except ValueError as e:
            # Catch any other read errors
            print(e)

    # Add the slider to napari
    viewer.window.add_dock_widget(frame_slider, area="right")

    # 5) Add a Play button that toggles playback using a QTimer
    play_button = PushButton(label="Play")   # Magicgui also works, but a simple widget is enough
    timer = QTimer()
    playback_state = {"is_playing": False}

    def on_timer_tick():
        """
        Advance the slider forward by 1 frame on each timer tick.
        Loop back to frame 0 if we reach the last extracted frame.
        """
        current_val = frame_slider.frame_index.value
        total = loader.total_frames()  # total frames extracted so far

        if current_val < total - 1:
            frame_slider.frame_index.value = current_val + 1
        else:
            # Reset to the first frame. Alternatively, you could stop playback here.
            frame_slider.frame_index.value = 0

    def toggle_play(event):
        """
        Start or stop the timer.
        """
        playback_state["is_playing"] = not playback_state["is_playing"]
        if playback_state["is_playing"]:
            play_button.text = "Pause"
            timer.start(5)  # update ~20 frames per second (adjust as needed)
        else:
            play_button.text = "Play"
            timer.stop()

    play_button.changed.connect(toggle_play)
    timer.timeout.connect(on_timer_tick)

    # Add the play button to napari
    viewer.window.add_dock_widget(play_button, area="right")

    # 6) Start the napari event loop
    napari.run()
