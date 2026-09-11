"""Experiment-specific configuration definitions and lookup helpers."""
import os
import sys
import unittest
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Constants for default paths (can be overridden by environment variables)
# Suggestion 3: Dynamic Path Handling
DEFAULT_BASE_PATH = os.getenv('EXPERIMENT_BASE_PATH', r'D:\Experiments')
DEFAULT_SAVE_PATH = os.getenv('EXPERIMENT_SAVE_PATH', r'F:\Results (paper)')

@dataclass
class ExperimentConfig:
    """
    Configuration for a specific experiment type.
    Encapsulates all experiment-specific variables.
    """
    path_home: Path
    path_save: Path
    cond_dict: Dict[str, Any]
    
    # Trial settings
    min_number_cs_trials: int
    min_number_us_trials: int
    time_aft_last_trial: int # minutes or seconds depending on context
    
    # Block definitions
    trials_cs_blocks_10: List[List[int]]
    trials_us_blocks_10: List[List[int]]
    names_cs_blocks_10: List[str]
    names_us_blocks_10: List[str]
    
    trials_cs_blocks_phases: List[Any] # Often numpy arrays
    trials_us_blocks_phases: List[Any]
    names_cs_blocks_phases: List[str]
    names_us_blocks_phases: List[str]

    # Timing / Analysis windows
    cs_duration: float = 10.0 # seconds
    cr_window: List[float] = field(default_factory=lambda: [0, 9])
    
    # Block Analysis settings
    number_trials_block: int = 5
    min_number_trials_with_bouts_per_block: int = 3
    us_window_qc: int = 5
    
    # Post-init to validate or derive simple things if needed
    block_names: List[str] = field(default_factory=lambda: [
        'Early Pre-Train', 'Late Pre-Train', 'Early Train', 'Train 2', 'Train 3',
        'Train 4', 'Train 5', 'Train 6', 'Train 7', 'Train 8', 'Train 9',
        'Late Train', 'Early Test', 'Test 2', 'Test 3', 'Test 4', 'Test 5',
        'Late Test'
    ])
    blocks_chosen: List[str] = field(default_factory=lambda: ['Late Pre-Train', 'Early Test', 'Late Test'])

    @property
    def cond_types(self) -> List[str]:
        return list(self.cond_dict.keys())

    @property
    def exp_types_names(self) -> List[str]:
        return [self.cond_dict[k]['name in original path'] for k in self.cond_types]

    @property
    def color_palette(self) -> List[Tuple[float, float, float]]:
        # Logic from original script: color/256
        return [tuple([y/256 for y in self.cond_dict[x]['color']]) for x in self.cond_types]

    @property
    def blocks_dict(self) -> Dict[str, Any]:
        """
        Generates the dictionary defining block structures for analysis/plotting.
        Mirroring the logic at the end of original script.
        """
        blocks_cs_single_trials = [[x] for x in np.arange(1, self.min_number_cs_trials + 1)]
        blocks_us_single_trials = [[x] for x in np.arange(1, self.min_number_us_trials + 1)]
        
        number_rows_blocks_10 = max(len(self.trials_cs_blocks_10), len(self.trials_us_blocks_10))
        number_rows_blocks_phases = max(len(self.trials_cs_blocks_phases), len(self.trials_us_blocks_phases))
        
        return {
            'single trials': {
                'CS': {
                    'trials in each block': blocks_cs_single_trials,
                    'names of blocks': [str(x[0]) for x in blocks_cs_single_trials]
                },
                'US': {
                    'trials in each block': blocks_us_single_trials,
                    'names of blocks': [str(x[0]) for x in blocks_us_single_trials]
                },
                'number of cols or rows': self.min_number_cs_trials,
                'figure size': (15, 2 * self.min_number_cs_trials / 3)
            },
            'blocks 10 trials': {
                'CS': {
                    'trials in each block': self.trials_cs_blocks_10,
                    'names of blocks': self.names_cs_blocks_10
                },
                'US': {
                    'trials in each block': self.trials_us_blocks_10,
                    'names of blocks': self.names_us_blocks_10
                },
                'number of cols or rows': number_rows_blocks_10,
                'figure size': (60 * number_rows_blocks_10 / 4, 15)
            },
            'blocks phases': {
                'CS': {
                    'trials in each block': self.trials_cs_blocks_phases,
                    'names of blocks': self.names_cs_blocks_phases
                },
                'US': {
                    'trials in each block': self.trials_us_blocks_phases,
                    'names of blocks': self.names_us_blocks_phases
                },
                'number of cols or rows': number_rows_blocks_phases,
            }
        }

class ExperimentType(Enum):
    # Enum values must match the names used in data folders and scripts.
    RESP_TO_US = 'respToUS'
    MOVING_CS_4COND = 'movingCS_4cond'
    ONLY_STIMULATION = 'only stimulation'
    MANY_DELAY_TRAINING_TRIALS = 'manyDelayTrainingTrials'
    FIRST_DELAY = 'firstDelay'
    ALL_DELAY = 'allDelay'
    FIXED_VS_INCREASING_TRACE = 'fixedVsIncreasingTrace'
    ALL_INC_TRACE = 'allIncTrace'
    ALL_3S_TRACE = 'all3sTrace'
    FIXED_TRACE_FULLY_REINFORCED = 'fixedTraceFullyReinforced'
    TRACE_10S = '10sTrace'
    ALL_10S_TRACE = 'all10sTrace'
    DELAY_MK801 = 'delaymk801'
    TRACE_MK801 = 'tracemk801'
    DELAY_LONG_TERM_SPACED_VS_MASSED_1 = 'delayLongTermSpacedVsMassed-1'
    DELAY_LONG_TERM_SPACED_VS_MASSED_2 = 'delayLongTermSpacedVsMassed-2'
    DELAY_LONG_TERM_SPACED_VS_MASSED_ALL = 'delayLongTermSpacedVsMassed-all'
    TRACE_LONG_TERM_SPACED_VS_MASSED_1 = 'traceLongTermSpacedVsMassed-1'
    TRACE_LONG_TERM_SPACED_VS_MASSED_2 = 'traceLongTermSpacedVsMassed-2'
    TRACE_LONG_TERM_SPACED_VS_MASSED_ALL = 'traceLongTermSpacedVsMassed-all'
    TRACE_PARTIAL_REINFORCEMENT = 'tracePartialReinforcement'
    TRACE_PUROMYCIN_SHORT = 'tracepuromycinshort'
    DELAY_LONG_TERM_SPACED_PUROMYCIN_5MG_1 = 'delayLongTermSpacedpuromycin5mg-1'
    DELAY_LONG_TERM_SPACED_PUROMYCIN_5MG_2 = 'delayLongTermSpacedpuromycin5mg-2'
    DELAY_LONG_TERM_SPACED_PUROMYCIN_5MG_ALL = 'delayLongTermSpacedpuromycin5mg-all'
    DELAY_LONG_TERM_SPACED_PUROMYCIN_10MG_1 = 'delayLongTermSpacedpuromycin10mg-1'
    DELAY_LONG_TERM_SPACED_PUROMYCIN_10MG_2 = 'delayLongTermSpacedpuromycin10mg-2'
    DELAY_LONG_TERM_SPACED_PUROMYCIN_10MG_ALL = 'delayLongTermSpacedpuromycin10mg-all'
    DELAY_LONG_TERM_NEW_1 = 'delayLongTermNew-1'
    DELAY_LONG_TERM_NEW_2 = 'delayLongTermNew-2'
    DELAY_LONG_TERM_NEW_3 = 'delayLongTermNew-3'
    DELAY_LONG_TERM_NEW_ALL = 'delayLongTermNew-all'
    TWO_P_MULTIPLE_PLANES_TOP = '2-P multiple planes top'
    TWO_P_MULTIPLE_PLANES_BOTTOM = '2-P multiple planes bottom'
    TWO_P_MULTIPLE_PLANES_ZOOM_IN = '2-P multiple planes zoom in'
    TWO_P_MULTIPLE_PLANES_CA8 = '2-P multiple planes ca8'
    TWO_P_SINGLE_PLANE = '2-P single plane'
    CA8_ABLATION = 'ca8ablation'
    # Add others as needed

def get_experiment_config(experiment_name: str) -> ExperimentConfig:
    """
    Factory function to retrieve the configuration for a given experiment name.
    """
    print(f"Retrieving configuration for experiment: {experiment_name}")
    
    if experiment_name == ExperimentType.RESP_TO_US.value:
        return ExperimentConfig(
            path_home=Path(r'D:\2022 07-08_Responses to US'),
            path_save=Path(r'F:\Results (paper)\2022 07-08_Responses to US'),
            cond_dict={
                'us10msnoopt': {'color': (0, 0, 0), 'name': '10 ms, 0 μM opt.', 'name in original path': 'US10msNoOpt'},
                'us10ms5umopt': {'color': (0, 0, 0), 'name': '10 ms, 5 μM opt.', 'name in original path': 'US10ms5umOpt'},
                'us10ms10umopt': {'color': (0, 0, 0), 'name': '10 ms, 10 μM opt.', 'name in original path': 'US10ms10umOpt'},
                'us50ms10umopt': {'color': (0, 0, 0), 'name': '50 ms, 10 μM opt.', 'name in original path': 'US50ms10umOpt'},
                'us100ms10umopt': {'color': (0, 0, 0), 'name': '100 ms, 10 μM opt.', 'name in original path': 'US100ms10umOpt'}
            },
            min_number_cs_trials=0,
            min_number_us_trials=51,
            time_aft_last_trial=1,
            cs_duration=4, 
            trials_cs_blocks_10=[],
            trials_us_blocks_10=[
                [*range(1, 11)], [*range(11, 21)], [*range(21, 31)], [*range(31, 41)], [*range(41, 51)]
            ],
            names_cs_blocks_10=[],
            names_us_blocks_10=['Block 1', 'Block 2', 'Block 3', 'Block 4', 'Block 5'],
            trials_cs_blocks_phases=[],
            trials_us_blocks_phases=[np.arange(1, 51)],
            names_cs_blocks_phases=[],
            names_us_blocks_phases=['Block']
        )

    elif experiment_name == ExperimentType.MOVING_CS_4COND.value:
        return ExperimentConfig(
            path_home=Path(r'D:\2022 06_Last version w moving CS 4 cond\Raw data'),
            path_save=Path(r'F:\Results (paper)\2022 06_Last version w moving CS 4 cond'),
            cond_dict={
                'control': {'color': (0,174,239), 'name': 'Control', 'name in original path': 'control'},
                'delaynoopt': {'color': (189,0,112), 'name': 'No opt. (delay)', 'name in original path': 'delayNoOpt', 'US latency': [3]*50},
                'delay': {'color': (236,0,140), 'name': 'Delay', 'name in original path': 'delay', 'US latency': [3]*50},
                'trace': {'color': (241, 90, 41), 'name': 'Trace', 'name in original path': 'trace', 'US latency': [6.5]*50}
            },
            min_number_cs_trials=72,
            min_number_us_trials=52,
            time_aft_last_trial=1,
            cs_duration=4,
            trials_cs_blocks_10=[
                [*range(3, 13)], [*range(13, 23)], [*range(23, 33)], [*range(33, 43)],
                [*range(43, 53)], [*range(53, 63)], [*range(63, 73)]
            ],
            trials_us_blocks_10=[
                [*range(2, 12)], [*range(12, 22)], [*range(22, 32)], [*range(32, 42)], [*range(42, 52)]
            ],
            names_cs_blocks_10=['Pre-train', 'Train 1', 'Train 2', 'Train 3', 'Train 4', 'Train 5', 'Test'],
            names_us_blocks_10=['Train 1', 'Train 2', 'Train 3', 'Train 4', 'Train 5'],
            trials_cs_blocks_phases=[np.arange(3, 13), np.arange(13, 63), np.arange(63, 73)],
            trials_us_blocks_phases=[np.arange(2, 52)],
            names_cs_blocks_phases=['Pre', 'Train', 'Test'],
            names_us_blocks_phases=['Train']
        )

    elif experiment_name == ExperimentType.ONLY_STIMULATION.value:
        return ExperimentConfig(
            path_home=Path(r'D:\2022 10_Only stimulation and pre-conditioning\Raw data'),
            path_save=Path(r'F:\Results (paper)\2022 10_Only stimulation and pre-conditioning'),
            cond_dict={
                'opt-uv': {'color': (0,0,0), 'name': 'both VL and opt.', 'name in original path': 'Opt-UV'},
                'opt-none': {'color': (0,0,0), 'name': 'no VL, only opt.', 'name in original path': 'Opt-None'},
                'none-uv': {'color': (0,0,0), 'name': 'only VL, no opt.', 'name in original path': 'None-UV'},
                'none-none': {'color': (0,0,0), 'name': 'no VL, no opt.', 'name in original path': 'None-None'}
            },
            min_number_cs_trials=14,
            min_number_us_trials=18,
            time_aft_last_trial=1,
            cs_duration=10,
            cr_window=[0, 3],
            trials_cs_blocks_10=[[*range(1, 15)]],
            trials_us_blocks_10=[[*range(1, 19)]],
            names_cs_blocks_10=['Stimulation'],
            names_us_blocks_10=['Stimulation'],
            trials_cs_blocks_phases=[np.arange(1, 14)],
            trials_us_blocks_phases=[np.arange(1, 18)],
            names_cs_blocks_phases=['Stimulation'],
            names_us_blocks_phases=['Stimulation']
        )

    elif experiment_name == ExperimentType.MANY_DELAY_TRAINING_TRIALS.value:
        return ExperimentConfig(
            path_home=Path(r'D:\2025 10_Delay long train\Raw data'),
            path_save=Path(r'F:\Results (paper)\2025 10_Delay long train'),
            cond_dict={
                'control': {'color': (0,174,239), 'name': 'Control', 'name in original path': 'control'},
                'delay': {'color': (236,0,140), 'name': 'Delay', 'name in original path': 'delay', 'US latency': [9]*46}
            },
            min_number_cs_trials=94,
            min_number_us_trials=78,
            time_aft_last_trial=1,
            cs_duration=10,
            cr_window=[0, 9],
            trials_cs_blocks_10=[
                [*range(5,15)], [*range(15,25)], [*range(25,35)], [*range(35,45)], [*range(45,55)],
                [*range(55,65)], [*range(65,75)], [*range(75,85)], [*range(85,95)], [*range(95,105)],
                [*range(105,115)], [*range(115,120)], [*range(120,130)], [*range(130,140)], [*range(140,150)]
            ],
            trials_us_blocks_10=[
                [*range(18,28)], [*range(28,37)], [*range(37,46)], [*range(46,55)], [*range(55,64)],
                [*range(64,74)], [*range(74,84)], [*range(84,94)], [*range(94,104)], [*range(103,111)],
                [*range(111,117)]
            ],
            names_cs_blocks_10=[
                'Pre-train', 'Train 1', 'Train 2', 'Train 3', 'Train 4', 'Train 5', 'Train 6', 'Train 7',
                'Train 8', 'Train 9', 'Train 10', 'Train 11', 'Test 1', 'Test 2', 'Test 3'
            ],
            names_us_blocks_10=[
                'Train 1', 'Train 2', 'Train 3', 'Train 4', 'Train 5', 'Train 6', 'Train 7', 'Train 8',
                'Train 9', 'Train 10', 'Train 11'
            ],
            trials_cs_blocks_phases=[np.arange(5,15), np.arange(15,120), np.arange(120,150)],
            trials_us_blocks_phases=[np.arange(18,64+53)],
            names_cs_blocks_phases=['Pre', 'Train', 'Test'],
            names_us_blocks_phases=['Train']
        )

    elif experiment_name == ExperimentType.ALL_DELAY.value:
         return ExperimentConfig(
            path_home=Path(r''),
            path_save=Path(r'F:\Results (paper)\2025_delay'),
            cond_dict={
                'control': {'color': (0,174,239), 'name': 'Control', 'name in original path': 'control'},
                'delay': {'color': (236,0,140), 'name': 'Delay', 'name in original path': 'delay', 'US latency': [9]*46}
            },
            min_number_cs_trials=94,
            min_number_us_trials=78,
            time_aft_last_trial=1,
            cs_duration=10,
            cr_window=[0, 9],
            trials_cs_blocks_10=[
                [*range(5,15)], [*range(15,25)], [*range(25,35)], [*range(35,45)],
                [*range(45,55)], [*range(55,65)], [*range(65,75)], [*range(75,85)], [*range(85,95)]
            ],
            trials_us_blocks_10=[
                [*range(18,28)], [*range(28,37)], [*range(37,46)], [*range(46,55)], [*range(55,64)]
            ],
            names_cs_blocks_10=['Pre-train', 'Train 1', 'Train 2', 'Train 3', 'Train 4', 'Train 5', 'Test 1', 'Test 2', 'Test 3'],
            names_us_blocks_10=['Train 1', 'Train 2', 'Train 3', 'Train 4', 'Train 5'],
            trials_cs_blocks_phases=[np.arange(5,15), np.arange(15,65), np.arange(65,95)],
            trials_us_blocks_phases=[np.arange(18,64)],
            names_cs_blocks_phases=['Pre', 'Train', 'Test'],
            names_us_blocks_phases=['Train']
        )
        
    elif experiment_name == ExperimentType.FIRST_DELAY.value:
        # Based on 'firstDelay' in logic
        min_trace_interval = 10
        # ... logic for US latency omitted for brevity in config but available if needed
        
        return ExperimentConfig(
            path_home=Path(r'D:\2022 11_Basic delay and (increasing) trace CC paradigm\Raw data'),
            path_save=Path(r'F:\Results (paper)\2022 11_Basic delay and (increasing) trace CC paradigm'),
            cond_dict={
                'control': {'color': (0,174,239), 'name': 'Control', 'name in original path': 'control'},
                'delay': {'color': (236,0,140), 'name': 'Delay', 'name in original path': 'delay', 'US latency': [9]*46},
                'trace': {'color': (255, 135, 94), 'name': 'Trace (inc.)', 'name in original path': 'trace', 'US latency': []} # Placeholder for complex calc
            },
            min_number_cs_trials=94,
            min_number_us_trials=78,
            time_aft_last_trial=1,
            cs_duration=10,
            cr_window=[0, 9],
            trials_cs_blocks_10=[
                [*range(5,15)], [*range(15,25)], [*range(25,35)], [*range(35,45)], [*range(45,55)],
                [*range(55,65)], [*range(65,75)], [*range(75,85)], [*range(85,95)]
            ],
            trials_us_blocks_10=[
                [*range(18,28)], [*range(28,37)], [*range(37,46)], [*range(46,55)], [*range(55,64)]
            ],
            names_cs_blocks_10=['Pre-train', 'Train 1', 'Train 2', 'Train 3', 'Train 4', 'Train 5', 'Test 1', 'Test 2', 'Test 3'],
            names_us_blocks_10=['Train 1', 'Train 2', 'Train 3', 'Train 4', 'Train 5'],
            trials_cs_blocks_phases=[np.arange(5,15), np.arange(15,65), np.arange(65,95)],
            trials_us_blocks_phases=[np.arange(18,64)],
             names_cs_blocks_phases=['Pre', 'Train', 'Test'],
            names_us_blocks_phases=['Train']
        )

    elif experiment_name == ExperimentType.ALL_INC_TRACE.value:
        # Based on 'allIncTrace' in my_experiment_specific_variables.py
        min_trace_interval_stable_numb_trials = 10
        max_trace_interval_stable_numb_trials = 10
        us_latency_trace_min = 0.5  # s
        us_latency_trace_max = 3.0  # s
        number_us_trials_increasing_trace = 26

        cs_duration = 10  # s
        us_latency_increasing_trace = (
            cs_duration
            + np.array(
                [us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1)
                + list(
                    np.linspace(
                        us_latency_trace_min,
                        us_latency_trace_max,
                        num=number_us_trials_increasing_trace + 1,
                        endpoint=False,
                        dtype=float,
                    )
                )
                + [us_latency_trace_max] * max_trace_interval_stable_numb_trials
            )
        ).tolist()

        return ExperimentConfig(
            path_home=Path(r''),
            path_save=Path(r'F:\Results (paper)\2025_incTrace'),
            cond_dict={
                'control': {
                    'color': (0, 174, 239),
                    'name': 'Control',
                    'name in original path': 'control',
                },
                'trace': {
                    'color': (255, 135, 94),
                    'name': 'incTrace',
                    'name in original path': 'increasingTrace',
                    'US latency': us_latency_increasing_trace,
                },
            },
            min_number_cs_trials=94,
            min_number_us_trials=78,
            time_aft_last_trial=1,
            cs_duration=cs_duration,
            cr_window=[0, 10.5],
            trials_cs_blocks_10=[
                [*range(5, 15)],
                [*range(15, 25)],
                [*range(25, 35)],
                [*range(35, 45)],
                [*range(45, 55)],
                [*range(55, 65)],
                [*range(65, 75)],
                [*range(75, 85)],
                [*range(85, 95)],
            ],
            trials_us_blocks_10=[
                [*range(18, 28)],
                [*range(28, 37)],
                [*range(37, 46)],
                [*range(46, 55)],
                [*range(55, 64)],
            ],
            names_cs_blocks_10=[
                'Pre-train',
                'Train 1',
                'Train 2',
                'Train 3',
                'Train 4',
                'Train 5',
                'Test 1',
                'Test 2',
                'Test 3',
            ],
            names_us_blocks_10=['Train 1', 'Train 2', 'Train 3', 'Train 4', 'Train 5'],
            trials_cs_blocks_phases=[np.arange(5, 15), np.arange(15, 65), np.arange(65, 95)],
            trials_us_blocks_phases=[np.arange(15, 65)],
            names_cs_blocks_phases=['Pre', 'Train', 'Test'],
            names_us_blocks_phases=['Train'],
        )

    elif experiment_name == ExperimentType.ALL_3S_TRACE.value:
        return ExperimentConfig(
            path_home=Path(r''),
            path_save=Path(r'F:\Results (paper)\2025_3sTrace'),
            cond_dict={
                'control': {'color': (0,174,239), 'name': 'Control', 'name in original path': 'control'},
                'trace': {'color': (241,90,41), 'name': '3sTrace', 'name in original path': 'trace', 'US latency': [9]*46}
            },
            min_number_cs_trials=94,
            min_number_us_trials=78,
            time_aft_last_trial=1,
            cs_duration=10,
            cr_window=[0, 13],
            trials_cs_blocks_10=[
                [*range(5,15)], [*range(15,25)], [*range(25,35)], [*range(35,45)], [*range(45,55)],
                [*range(55,65)], [*range(65,75)], [*range(75,85)], [*range(85,95)]
            ],
            trials_us_blocks_10=[
                [*range(18,28)], [*range(28,37)], [*range(37,46)], [*range(46,55)], [*range(55,64)]
            ],
            names_cs_blocks_10=[
                'Pre-train', 'Train 1', 'Train 2', 'Train 3', 'Train 4', 'Train 5', 'Test 1', 'Test 2', 'Test 3'
            ],
            names_us_blocks_10=['Train 1', 'Train 2', 'Train 3', 'Train 4', 'Train 5'],
            trials_cs_blocks_phases=[np.arange(5,15), np.arange(15,65), np.arange(65,95)],
            trials_us_blocks_phases=[np.arange(15,65)],
            names_cs_blocks_phases=['Pre', 'Train', 'Test'],
            names_us_blocks_phases=['Train']
        )

    elif experiment_name == ExperimentType.TRACE_10S.value:
        return ExperimentConfig(
            path_home=Path(r'D:\2023 03_3-s vs 10-s fixed trace\Raw data'),
            path_save=Path(r'F:\Results (paper)\2023 03_3-s vs 10-s fixed trace'),
            cond_dict={
                'control': {'color': (0,174,239), 'name': 'Control', 'name in original path': 'control'},
                '3sfixedtrace': {'color': (241,90,41), 'name': 'Trace (3 s)', 'name in original path': '3sFixedTrace', 'US latency': [13]*46},
                'trace': {'color': (145,54,25), 'name': '10sTrace', 'name in original path': '10sFixedTrace', 'US latency': [20]*46}
            },
            min_number_cs_trials=94,
            min_number_us_trials=78,
            time_aft_last_trial=1,
            cs_duration=10,
            cr_window=[0, 13],
            trials_cs_blocks_10=[
                [*range(5,15)], [*range(15,25)], [*range(25,35)], [*range(35,45)], [*range(45,55)],
                [*range(55,65)], [*range(65,75)], [*range(75,85)], [*range(85,95)]
            ],
            trials_us_blocks_10=[
                [*range(18,28)], [*range(28,37)], [*range(37,46)], [*range(46,55)], [*range(55,64)]
            ],
            names_cs_blocks_10=[
                'Pre-train', 'Train 1', 'Train 2', 'Train 3', 'Train 4', 'Train 5', 'Test 1', 'Test 2', 'Test 3'
            ],
            names_us_blocks_10=['Train 1', 'Train 2', 'Train 3', 'Train 4', 'Train 5'],
            trials_cs_blocks_phases=[np.arange(5,15), np.arange(15,65), np.arange(65,95)],
            trials_us_blocks_phases=[np.arange(15,65)],
            names_cs_blocks_phases=['Pre', 'Train', 'Test'],
            names_us_blocks_phases=['Train']
        )

    elif experiment_name == ExperimentType.ALL_10S_TRACE.value:
        return ExperimentConfig(
            path_home=Path(r''),
            path_save=Path(r'F:\Results (paper)\2025_10sTrace'),
            cond_dict={
                'control': {'color': (0,174,239), 'name': 'Control', 'name in original path': 'control'},
                'trace': {'color': (145,54,25), 'name': '10sTrace', 'name in original path': '10sFixedTrace', 'US latency': [20]*46}
            },
            min_number_cs_trials=94,
            min_number_us_trials=78,
            time_aft_last_trial=1,
            cs_duration=10,
            cr_window=[0, 20],	
            trials_cs_blocks_10=[
                [*range(5,15)], [*range(15,25)], [*range(25,35)], [*range(35,45)], [*range(45,55)],
                [*range(55,65)], [*range(65,75)], [*range(75,85)], [*range(85,95)]
            ],
            trials_us_blocks_10=[
                [*range(18,28)], [*range(28,37)], [*range(37,46)], [*range(46,55)], [*range(55,64)]
            ],
            names_cs_blocks_10=[
                'Pre-train', 'Train 1', 'Train 2', 'Train 3', 'Train 4', 'Train 5', 'Test 1', 'Test 2', 'Test 3'
            ],
            names_us_blocks_10=['Train 1', 'Train 2', 'Train 3', 'Train 4', 'Train 5'],
            trials_cs_blocks_phases=[np.arange(5,15), np.arange(15,65), np.arange(65,95)],
            trials_us_blocks_phases=[np.arange(15,65)],
            names_cs_blocks_phases=['Pre', 'Train', 'Test'],
            names_us_blocks_phases=['Train']
        )

    elif experiment_name == ExperimentType.CA8_ABLATION.value:
        return ExperimentConfig(
            path_home=Path(r'D:\2025 09_delay and 3sTrace ca8 ablation'),
            path_save=Path(r'F:\Results (paper)\2025 09_delay and 3sTrace ca8 ablation'),
            cond_dict={
                'controlca8negnfp': {'color': (0,174,239), 'name': 'Ctrl Ca8-', 'name in original path': 'controlCa8negNfp'},
                'controlca8posnfp': {'color': (236,0,140), 'name': 'Ctrl Ca8+', 'name in original path': 'controlCa8posNfp', 'US latency': [9]*46},
                'delayca8negnfp': {'color': (0, 139, 191), 'name': 'Delay Ca8-', 'name in original path': 'delayCa8negNfp'},
                'delayca8posnfp': {'color': (189, 0, 112), 'name': 'Delay Ca8+', 'name in original path': 'delayCa8posNfp', 'US latency': [9]*46},
                'traceca8negnfp': {'color': (0, 104, 143), 'name': 'Trace Ca8-', 'name in original path': 'traceCa8negNfp'},
                'traceca8posnfp': {'color': (142, 0, 84), 'name': 'Trace Ca8+', 'name in original path': 'traceCa8posNfp', 'US latency': [9]*46}
            },
            min_number_cs_trials=94,
            min_number_us_trials=78,
            time_aft_last_trial=1,
            cs_duration=10,
            cr_window=[0, 9],
            trials_cs_blocks_10=[
                [*range(5,15)], [*range(15,25)], [*range(25,35)], [*range(35,45)], [*range(45,55)],
                [*range(55,65)], [*range(65,75)], [*range(75,85)], [*range(85,95)]
            ],
            trials_us_blocks_10=[
                [*range(18,28)], [*range(28,37)], [*range(37,46)], [*range(46,55)], [*range(55,64)]
            ],
            names_cs_blocks_10=['Pre-train', 'Train 1', 'Train 2', 'Train 3', 'Train 4', 'Train 5', 'Test 1', 'Test 2', 'Test 3'],
            names_us_blocks_10=['Train 1', 'Train 2', 'Train 3', 'Train 4', 'Train 5'],
            trials_cs_blocks_phases=[np.arange(5,15), np.arange(15,65), np.arange(65,95)],
            trials_us_blocks_phases=[np.arange(15,64)],
            names_cs_blocks_phases=['Pre', 'Train', 'Test'],
            names_us_blocks_phases=['Train']
        )

    elif experiment_name in {exp.value for exp in ExperimentType}:
        print(
            f"Experiment '{experiment_name}' is listed in ExperimentType but not yet configured."
        )
        raise NotImplementedError(
            f"Experiment '{experiment_name}' is not yet configured in experiment_configuration.py."
        )

    else:
        # Fallback or raise error
        print(f"Experiment '{experiment_name}' not found in configuration.")
        raise ValueError(f"Experiment name '{experiment_name}' not recognized.")

# Suggestion 13: Add Unit Tests
class TestExperimentConfig(unittest.TestCase):
    
    def test_resp_to_us_config(self):
        config = get_experiment_config(ExperimentType.RESP_TO_US.value)
        self.assertIsInstance(config, ExperimentConfig)
        self.assertEqual(config.min_number_cs_trials, 0)
        self.assertEqual(len(config.cond_types), 5)
        self.assertTrue('us10msnoopt' in config.cond_types)
        
    def test_blocks_dict_property(self):
        # Using respToUS which implies specific block structure
        config = get_experiment_config(ExperimentType.RESP_TO_US.value)
        blocks_dict = config.blocks_dict
        self.assertIn('single trials', blocks_dict)
        self.assertEqual(blocks_dict['single trials']['number of cols or rows'], 0)
        
    def test_invalid_experiment(self):
        with self.assertRaises(ValueError):
            get_experiment_config("INVALID_NAME")

if __name__ == '__main__':
    unittest.main()
