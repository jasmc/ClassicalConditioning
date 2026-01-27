"""Shared defaults and derived settings for the analysis pipeline."""
import math
from dataclasses import dataclass, field
from typing import Final, List, Tuple

import numpy as np

# Constants that are truly global/physics related
CM_TO_INCH: Final[float] = 1 / 2.54

@dataclass
class ValidationConfig:
    epsilon: float = 1e-9
    number_frames_discard_beg: int = 20 * 700  # ~60s at 700 FPS
    buffer_size: int = 700
    max_interval_between_frames: float = 0.005  # ms
    lag_thr: float = 50  # ms
    single_point_tracking_error_thr: float = 2 * 180 / np.pi  # deg

@dataclass
class FilteringConfig:
    space_bcf_window: int = 3
    time_bcf_window: int = 10
    time_max_window: int = 20
    time_min_window: int = 400
    filtering_window: int = 350
    # Note: All above are in frames/segments

@dataclass
class BoutDetectionConfig:
    bout_detection_thr_1: float = 4  # deg/ms
    bout_detection_thr_2: float = 1  # deg/ms
    min_interbout_time: int = 10     # frames
    min_bout_duration: int = 40      # frames

@dataclass
class PlottingConfig:
    downsampling_step: int = 5
    page_size: Tuple[float, float] = (13 * CM_TO_INCH, 18 * CM_TO_INCH)
    
    # Colors
    cs_color: List[float] = field(default_factory=lambda: [13/255, 129/255, 54/255]) # green
    us_color: List[float] = field(default_factory=lambda: [112/255, 46/255, 120/255]) # purple
    
    clip_low: int = 0
    clip_high: int = 1
    
    @property
    def segments_analysis(self) -> List[str]:
        # This requires context from the main config usually, but for now we put placeholders
        return ['Mean window', '', 'Normalized vigor']

@dataclass
class GeneralConfig:
    """
    Central configuration for the analysis pipeline.
    Replaces `my_general_variables.py`.
    """
    
    # Sub-configs
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    bout_detection: BoutDetectionConfig = field(default_factory=BoutDetectionConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)

    # Experiment Constants
    expected_framerate: int = 700  # FPS
    chosen_tail_point: int = 15    # 0-indexed (originally 16-1)
    
    # Timing - Analysis Windows
    # Defined in ms, converted to sec/frames as needed
    time_bef_ms: float = -45000
    time_aft_ms: float = 45000
    
    # Note: The original file had time_bef = -45 with comment # ms, but logic suggests it might be seconds or a shorter window?
    # Context clue: "To define the whole extent of trials". -45000 ms is -45s.
    # ORIGINAL: time_bef: Final = -45 # ms
    # If the original code meant seconds for 'time_bef' and ms for 'time_bef_ms', appropriate conversions are needed.
    # Assuming -45 and 45 are SECONDS based on typical behavioral analysis windows matching the 45000ms.
    
    time_bef_s: float = -45
    time_aft_s: float = 45
    
    # Baseline
    baseline_window: float = 15 # s
    
    # Cropping
    t_crop_data_bef_s: float = -45
    t_crop_data_aft_s: float = 45
    
    # Binning
    binning_window: float = 0.5 # s
    binning_window_long: float = 2.0 # s
    
    # Stimuli
    us_duration: float = 0.1 # s
    us_struggle_window: float = 15 # s
    
    def __post_init__(self):
        # Derive frame-based windows and labels from the scalar settings.
        # Derived values
        self.time_bef_frame = int(np.ceil(self.time_bef_ms * self.expected_framerate / 1000))
        self.time_aft_frame = int(np.ceil(self.time_aft_ms * self.expected_framerate / 1000))
        
        self.bin_or_window_name = f"{self.binning_window}-s window"
        self.mean_bef_onset = f"Mean {self.baseline_window} s before"
        
        self.plotting.segments_analysis[0] = self.mean_bef_onset
        
        # Calculate time bins
        # This mirrors the logic: list(np.arange(...))[::-1] + list(np.arange(...))
        self.time_bins_short = (
            list(np.arange(-self.binning_window/2, 
                          self.t_crop_data_bef_s - self.binning_window, 
                          -self.binning_window))[::-1] + 
            list(np.arange(self.binning_window/2, 
                          self.t_crop_data_aft_s + self.binning_window, 
                          self.binning_window))
        )
        
        self.time_bins_long = (
            list(np.arange(-self.binning_window_long, 
                          self.time_bef_s - self.binning_window_long, 
                          -self.binning_window_long))[::-1] + 
            list(np.arange(0, 
                          self.time_aft_s + self.binning_window_long, 
                          self.binning_window_long))
        )
        
        # Strings
        self.tail_angle_label = f'Angle of point {self.chosen_tail_point} (deg)'
        self.time_trial_frame_label = f'Trial time (frame) [{self.expected_framerate} FPS]'

    @property
    def cols_to_use_orig(self) -> List[str]:
        # 'angle0', 'angle1', ... up to chosen point + half window
        limit = self.chosen_tail_point + int(math.floor(self.filtering.space_bcf_window / 2))
        return ['FrameID'] + [f'angle{i}' for i in range(limit)]

# Instance for easy import and usage, replicating the module-level access pattern but cleaner
config = GeneralConfig()
