# NWB Data Viewer

Takes nwb objects from an open nwb file and streams the data for video-like playback
and interaction.

**Example Interactive Output (Video + eseries data)**
![](./example.gif)

## Basic Usage

- `ViewerMixin`: Base mixin class providing playback controls (play/pause, time slider, window scale) using napari and Qt timers.
- `MultiViewer`: manages multiple nwb object viewers for synchronized playback in a single napari window

- `NwbTimeSeriesViewer`:  for viewing multi-channel NWB time series data with channel selection and interval control.
- `NwbDigitalSeriesViewer`: for displaying timeseries data storing changes of a binary state (e.g. light status)
- `NwbSpatialSeriesViewer`: class extending time series viewer for spatial data visualization (position).
- `NwbPoseViewer`: class generalizing from spatial series for playback of bodyparts and connecting skeleton stored in a `ndx_pose` object
- `NwbVideoViewer`: class for synchronous playback of external video files in conjunction with other nwb objects

## Installation

**Pypi Release**

Coming soon

**Developer Installation**

1. Install miniconda (or anaconda) if it isn't already installed.
2. git clone <package.git>
3. Setup editiable package with dependencies

```bash
cd nwb_data_viewer
conda env create -f environment.yml
conda activate nwb_data_viewer
pip install -e .
```
