import math
import warnings
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

warnings.filterwarnings(action='ignore', category=FutureWarning)


epsilon = 1e-9 # Add small epsilon to avoid division by zero


cm_to_inch: Final = 1 / 2.54
page_size: Final = (13 * cm_to_inch, 18 * cm_to_inch)

#* Parameters for the analysis

# Point of the tail chosen as the last one to be considered in the analysis, using a tracking with 15 points on the tail.
chosen_tail_point: Final = 16 - 1

# To define the whole extent of trials, i.e., regions in time before and after a stimulus.
time_bef_ms: Final = -45000 # ms
time_aft_ms: Final = 45000 # ms
time_bef: Final = -45 # ms
time_aft: Final = 45 # ms


baseline_window = 15 # s

# To crop data, before pooling.
t_crop_data_bef_s = -45 #s
t_crop_data_aft_s = 45 #s

# To check whether any frame was lost.
number_frames_discard_beg: Final = 20*700 # around 60 s at 700 FPS (number of frames)
buffer_size: Final = 700 # frames
max_interval_between_frames: Final = 0.005 # ms
lag_thr: Final = 50 # ms

# To check for tracking errors.
single_point_tracking_error_thr: Final = 2 * 180/np.pi # deg

# Parameteres for filtering raw data.
#! Need to add here the units.
#? If you change this, you will have to change the rest of the code.
# All below in number of frames at expected_framerate (700 FPS)
space_bcf_window: Final = 3 # number segments of the tail
time_bcf_window: Final = 10
time_max_window: Final = 20
time_min_window: Final = 400
filtering_window: Final = 350 # frames

# Parameters for 'Bout' detection.
bout_detection_thr_1: Final = 4 # deg/ms   # Changed from 1 to 4 on 17 Jan 2023.
min_interbout_time: Final = 10 # frames
min_bout_duration: Final = 40 # frames
bout_detection_thr_2: Final = 1 # deg/ms

# Interpolate data to this framerate.
expected_framerate: Final = 700 # FPS

# After interpolating, time_bef and time_aft can be used in number of frames.
time_bef_frame: Final = int(np.ceil(time_bef_ms * expected_framerate/1000)) # frames
time_aft_frame: Final = int(np.ceil(time_aft_ms * expected_framerate/1000)) # frames


# Width of the bins used to group 'Bout beg (ms)' and 'Bout end (ms)' data, in s.
binning_window: Final = 0.5 # s

binning_window_long: Final = 2 # s

time_bins_short = list(np.arange(-binning_window/2, t_crop_data_bef_s-binning_window, -binning_window))[::-1] + list(np.arange(binning_window/2, t_crop_data_aft_s+binning_window, binning_window))

time_bins_long = list(np.arange(-binning_window_long, time_bef-binning_window_long, -binning_window_long))[::-1] + list(np.arange(0, time_aft+binning_window_long, binning_window_long))

bin_or_window_name = str(binning_window) + '-s window'


# Estimated duration of stimuli, for pooling data.
us_duration: Final = 0.1 # s

us_struggle_window = 15 # s


#* Parameters for clipping scaled vigor
clip_low = 0
clip_high = 1


#* Parameters for plotting data

# Donwsampling step to plot the whole experimental data, using plot_cropped_experiment function.
downsampling_step: Final = 5

cs_color: Final = [i/255 for i in [13, 129, 54]] # green
us_color: Final = [i/255 for i in [112, 46, 120]] # purple


mean_bef_onset = 'Mean '+str(baseline_window)+' s before'


#* Variables containing strings

segments_analysis: Final = [mean_bef_onset, '', 'Normalized vigor']

tail_angle: Final = 'Angle of point {} (deg)'.format(chosen_tail_point)

cols_to_use_orig = ['FrameID']
for i in range(chosen_tail_point+int(math.floor(space_bcf_window/2))):
    cols_to_use_orig.append('angle' + str(i))

cols = [0] * (len(cols_to_use_orig) - 1)
cols = ['Angle of point {} (deg)'.format(i) for i in range(len(cols))]


time_trial_frame: Final = 'Trial time (frame) [{} FPS]'.format(expected_framerate)