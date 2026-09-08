import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D
from pandas.api.types import CategoricalDtype
from scipy.spatial import distance
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

import my_functions as f

# # Read experiment type from external txt file
# path = r"C:\Users\joaquim\Desktop\experiment_to_process.txt.txt"
# try:
#     with open(path, 'r') as file:
#         experiment_ = file.read().strip()
#         print(f"Using experiment type from file: '{experiment_}'")
#             os.remove(path)
#             print(f"File '{path}' has been deleted.")
#         else:
#             print(f"File '{path}' doesn't exist.")
# except FileNotFoundError:
#     print(f"File 'experiment_type.txt' not found.")
# del file

experiment_ =  'allDelay'
# 'all3sTrace'
# Run discard again: # 
# Run discard again with cr_window = 10.5: # 

# 'all10sTrace'

# 'allIncTrace'




# '10sTrace'
# 'fixedVsIncreasingTrace'
# 'tracemk801'
# 'tracepuromycinshort'
# 'tracePartialReinforcement'

# 'firstDelay'
# 'delaymk801'

# 'ca8ablation'











# 'delayLongTermSpacedpuromycin5mg-all'







# 'only stimulation'
# 'respToUS'
# 'movingCS_4cond'








# 'manyDelayTrainingTrials'

# '2-P multiple planes zoom in'






# '2-P multiple planes top'
# '2-P single plane'
# '2-P multiple planes ca8'
# '2-P multiple planes bottom'

# 'delayLongTermNew-all'

# 'delayLongTermSpacedpuromycin10mg-1'


# 'traceLongTermSpacedVsMassed-all'
# 'delayLongTermSpacedVsMassed-all'

# 'delayLongTermSpacedpuromycin5mg-2'


# 'delayLongTermSpacedpuromycin10mg-all'



# 'cfos'
# 'ablation'

# 'delayDanionella'

match experiment_:

    case 'respToUS':

        path_home = Path(r'D:\2022 07-08_Responses to US')
        path_save = Path(r'F:\Results (paper)\2022 07-08_Responses to US')

        cond0 = 'us10msnoopt'
        cond1 = 'us10ms5umopt'
        cond2 = 'us10ms10umopt'
        cond3 = 'us50ms10umopt'
        cond4 = 'us100ms10umopt'

        cond_dict = {
            cond0: {'color': (0,0,0), 'name': '10 ms, 0 μM opt.', 'name in original path': 'US10msNoOpt'},
            cond1: {'color': (0,0,0), 'name': '10 ms, 5 μM opt.', 'name in original path': 'US10ms5umOpt'},
            cond2: {'color': (0,0,0), 'name': '10 ms, 10 μM opt.', 'name in original path': 'US10ms10umOpt'},
            cond3: {'color': (0,0,0), 'name': '50 ms, 10 μM opt.', 'name in original path': 'US50ms10umOpt'},
            cond4: {'color': (0,0,0), 'name': '100 ms, 10 μM opt.', 'name in original path': 'US100ms10umOpt'}}

        # cs_duration = 4 # s
        # cr_window = 3 # s

        min_number_cs_trials = 0
        # There are 50 stim as wanted to test and a viability test of 200 ms as the last trial.
        min_number_us_trials = 51
        time_aft_last_trial = 1  # s

        trials_cs_blocks_10 = []

        trials_us_blocks_10 = [
            [*range(1, 11)],
            [*range(11, 21)],
            [*range(21, 31)],
            [*range(31, 41)],
            [*range(41, 51)]]

        names_cs_blocks_10 = []

        names_us_blocks_10 = [
            'Block 1',
            'Block 2',
            'Block 3',
            'Block 4',
            'Block 5']

        trials_cs_blocks_phases = []
        # There are 50 stim as wanted to test and a viability test of 200 ms as the last trial.
        trials_us_blocks_phases = [np.arange(1, 51)]
        names_cs_blocks_phases = []
        names_us_blocks_phases = ['Block']

    case 'movingCS_4cond':

        path_home = Path(r'D:\2022 06_Last version w moving CS 4 cond\Raw data')
        path_save = Path(r'F:\Results (paper)\2022 06_Last version w moving CS 4 cond')

        cond0 = 'control'
        cond1 = 'delaynoopt'
        cond2 = 'delay'
        cond3 = 'trace'

        cond_dict = {
            cond0: {'color': (0,174,239), 'name': 'Control', 'name in original path': 'control'},
            cond1: {'color': (189,0,112), 'name': 'No opt. (delay)', 'name in original path': 'delayNoOpt', 'US latency': [3]*50},
            cond2: {'color': (236,0,140), 'name': 'Delay', 'name in original path': 'delay', 'US latency': [3]*50},
            cond3: {'color': (241, 90, 41), 'name': 'Trace', 'name in original path': 'trace', 'US latency': [6.5]*50}
        }

        relevant_trial_numbers = np.array([10, 20, 60, 70])
        where_v_lines_1 = [10.5, 60.5]
        relevant_session_numbers = np.array([0, 3, 6])
        where_v_lines_2 = np.array([0.5, 5.5])

        cs_duration = 4 # s
        # cr_window = 3 # s

        min_number_cs_trials = 72
        min_number_us_trials = 52
        time_aft_last_trial = 1

        trials_cs_blocks_10 = [
            [*range(3, 13)],
            [*range(13, 23)],
            [*range(23, 33)],
            [*range(33, 43)],
            [*range(43, 53)],
            [*range(53, 63)],
            [*range(63, 73)]]

        trials_us_blocks_10 = [
            [*range(2, 12)],
            [*range(12, 22)],
            [*range(22, 32)],
            [*range(32, 42)],
            [*range(42, 52)]]

        names_cs_blocks_10 = [
            'Pre-train',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
            'Test']

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5']

        trials_cs_blocks_phases = [np.arange(3, 13), np.arange(13, 63), np.arange(63, 73)]
        trials_us_blocks_phases = [np.arange(2, 52)]
        names_cs_blocks_phases = ['Pre', 'Train', 'Test']
        names_us_blocks_phases = ['Train']

    case 'only stimulation':

        path_home = Path(r'D:\2022 10_Only stimulation and pre-conditioning\Raw data')
        path_save = Path(r'F:\Results (paper)\2022 10_Only stimulation and pre-conditioning')

        cond0 = 'opt-uv'
        cond1 = 'opt-none'
        cond2 = 'none-uv'
        cond3 = 'none-none'

        cond_dict = {cond0: {'color': (0,0,0), 'name': 'both VL and opt.', 'name in original path': 'Opt-UV'},
                    cond1: {'color': (0,0,0), 'name': 'no VL, only opt.', 'name in original path': 'Opt-None'},
                    cond2: {'color': (0,0,0), 'name': 'only VL, no opt.', 'name in original path': 'None-UV'},
                    cond3: {'color': (0,0,0), 'name': 'no VL, no opt.', 'name in original path': 'None-None'}}

        cs_duration = 10 # s
        cr_window = 3 # s

        min_number_cs_trials = 14
        min_number_us_trials = 18
        time_aft_last_trial = 1

        trials_cs_blocks_10 = [[*range(1, 15)]]

        trials_us_blocks_10 = [[*range(1, 19)]]

        names_cs_blocks_10 = ['Stimulation']

        names_us_blocks_10 = ['Stimulation']

        trials_cs_blocks_phases = [np.arange(1,14)]
        trials_us_blocks_phases = [np.arange(1, 18)]
        names_cs_blocks_phases = ['Stimulation']
        names_us_blocks_phases = ['Stimulation']



    case 'manyDelayTrainingTrials':


        #* Path where the raw data is.
        path_home = Path(r'D:\2025 10_Delay long train\Raw data')
        path_save = Path(r'F:\Results (paper)\2025 10_Delay long train')

        cond0 = 'control'
        cond1 = 'delay'

        #TODO add more stuff to this dict.
        cond_dict = {
            cond0 : {'color' : (0,174,239), 'name' : 'Control', 'name in original path' : 'control'},
            cond1 : {'color' :  (236,0,140), 'name' : 'Delay', 'name in original path' : 'delay', 'US latency' : [9]*46}}

        relevant_trial_numbers = np.array([10, 20, 60, 70, 90])
        where_v_lines_1 = [10.5, 60.5]
        relevant_session_numbers = np.array([0, 3, 7])
        where_v_lines_2 = np.array([0.5, 5.5])

        cs_duration    = 10 # s
        cr_window = 9  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78
        time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        catch_trials_training = [25, 39, 53, 59, 93+15, 99+15]

        # catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]
        # list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))



        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],
                                [*range(65,75)],
                                [*range(75,85)],
                                [*range(85,95)],
                                [*range(95,105)],
                                [*range(105,115)],
                                [*range(115,120)],
                                [*range(120,130)],
                                [*range(130,140)],
                                [*range(140,150)]]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                [*range(55,64)],
                                [*range(64,74)],
                                [*range(74,84)],
                                [*range(84,94)],
                                [*range(94,104)],
                                [*range(103,111)],
                                [*range(111,117)]]

        names_cs_blocks_10 = [
            'Pre-train',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
            'Train 6',
            'Train 7',
            'Train 8',
            'Train 9',
            'Train 10',
            'Train 11',
            'Test 1',
            'Test 2',
            'Test 3']

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
            'Train 6',
            'Train 7',
            'Train 8',
            'Train 9',
            'Train 10',
            'Train 11']

        trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,120), np.arange(120,150)]

        trials_us_blocks_phases = [np.arange(18,64+53)]

        names_cs_blocks_phases = [
            'Pre',
            'Train',
            'Test']

        names_us_blocks_phases = [
            'Train']



    case 'firstDelay':

    #! November 2022 basic delay and trace 'Exp.'

        #* Path where the raw data is.
        path_home = Path(r'D:\2022 11_Basic delay and (increasing) trace CC paradigm\Raw data')
        path_save = Path(r'F:\Results (paper)\2022 11_Basic delay and (increasing) trace CC paradigm')

        cond0 = 'control'
        cond1 = 'delay'
        cond2 = 'trace'

        min_trace_interval_stable_numb_trials = 10
        max_trace_interval_stable_numb_trials = 10

        us_latency_trace_min = 0.5 # s
        us_latency_trace_max = 3  # s

        #! Careful with the number of catch trials!
        number_us_trials_increasing_trace = 26

        # us_latency_conditioning_list = cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace_max, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace_max] * max_trace_interval_stable_numb_trials)/1000

        cs_duration    = 10 # s

        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : (0,174,239), 'name' : 'Control', 'name in original path' : 'control'},
                  cond1 : {'color' :  (236,0,140), 'name' : 'Delay', 'name in original path' : 'delay', 'US latency' : [9]*46},
                  cond2 : {'color' : (255, 135, 94), 'name' : 'Trace (inc.)', 'name in original path' : 'trace', 'US latency' :
           cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace_max, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace_max] * max_trace_interval_stable_numb_trials)}}

        relevant_trial_numbers = np.array([10, 20, 60, 70, 90])
        where_v_lines_1 = [10.5, 60.5]
        relevant_session_numbers = np.array([0, 3, 7])
        where_v_lines_2 = np.array([0.5, 5.5])

#! Exception
        cr_window = 9  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78
        time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        catch_trials_training = [25, 39, 53, 59]

        catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]
        # list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))



        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],
                                [*range(65,75)],
                                [*range(75,85)],
                                [*range(85,95)],
                                ]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                [*range(55,64)]]

        names_cs_blocks_10 = [
            'Pre-train',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
            'Test 1',
            'Test 2',
            'Test 3']

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5']



        trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65,95)]

        trials_us_blocks_phases = [np.arange(18,64)]

        names_cs_blocks_phases = [
            'Pre',
            'Train',
            'Test']

        names_us_blocks_phases = [
            'Train']


    case 'allDelay':

    #! November 2022 basic delay and trace 'Exp.'

        #* Path where the raw data is.
        path_home = Path(r'')
        path_save = Path(r'F:\Results (paper)\2025_delay')

        cond0 = 'control'
        cond1 = 'delay'

        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : (0,174,239), 'name' : 'Control', 'name in original path' : 'control'},
                  cond1 : {'color' :  (236,0,140), 'name' : 'Delay', 'name in original path' : 'delay', 'US latency' : [9]*46}}

        relevant_trial_numbers = np.array([10, 20, 60, 70, 90])
        where_v_lines_1 = [10.5, 60.5]
        relevant_session_numbers = np.array([0, 3, 7])
        where_v_lines_2 = np.array([0.5, 5.5])

        cs_duration    = 10 # s
        cr_window = 9  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78
        time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        catch_trials_training = [25, 39, 53, 59]

        catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]
        # list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))



        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],
                                [*range(65,75)],
                                [*range(75,85)],
                                [*range(85,95)],
                                ]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                [*range(55,64)]]

        names_cs_blocks_10 = [
            'Pre-train',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
            'Test 1',
            'Test 2',
            'Test 3']

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5']



        trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65,95)]

        trials_us_blocks_phases = [np.arange(18,64)]

        names_cs_blocks_phases = [
            'Pre',
            'Train',
            'Test']

        names_us_blocks_phases = [
            'Train']

    case 'fixedVsIncreasingTrace':

    #! End February-March 2023 comparison between 3-s fixed trace and 3-s increasing trace

        #* Path where the raw data is.
        path_home = Path(r'D:\2023 02-03_Fixed vs increasing trace (3 s)\Raw data')
        path_save = Path(r'F:\Results (paper)\2023 02-03_Fixed vs increasing trace (3 s)')

        cond0 = 'control'
        cond1 = 'fixedtrace'
        cond2 = 'increasingtrace'

        min_trace_interval_stable_numb_trials = 10
        max_trace_interval_stable_numb_trials = 10

        us_latency_trace_min = 0.5  # s
        us_latency_trace_max = 3  # s

        #! Careful with the number of catch trials!
        number_us_trials_increasing_trace = 26

        cs_duration = 10 # s

        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : (0,174,239), 'name' : 'Control', 'name in original path' : 'control'},
                  cond1 : {'color' :  (241,90,41), 'name' : 'Trace CC fixed', 'name in original path' : 'fixedTrace', 'US latency' : [9]*46},
                  cond2 : {'color' : (255, 135, 94), 'name' : 'Trace CC increasing', 'name in original path' : 'increasingTrace', 'US latency' :
           cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace_max, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace_max] * max_trace_interval_stable_numb_trials)}
                  }

        relevant_trial_numbers = np.array([10, 20, 60, 70, 90])
        where_v_lines_1 = [10.5, 60.5]
        relevant_session_numbers = np.array([0, 3, 7])
        where_v_lines_2 = np.array([0.5, 5.5])

        cr_window = 13  # s


        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78
        time_aft_last_trial = 1 # min

        catch_trials_training = [25, 39, 53, 59]

        catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]


        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],
                                [*range(65,75)],
                                [*range(75,85)],
                                [*range(85,95)]]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                [*range(55,64)]]

        names_cs_blocks_10 = [
            'Pre-train',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
            'Test 1',
            'Test 2',
            'Test 3']

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5'
        ]



        trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65,95)]

        trials_us_blocks_phases = [np.arange(15,65)]

        names_cs_blocks_phases = [
            'Pre',
            'Train',
            'Test'
        ]

        names_us_blocks_phases = [
            'Train'
        ]

    case 'allIncTrace':

    #! End February-March 2023 comparison between 3-s fixed trace and 3-s increasing trace

        #* Path where the raw data is.
        path_home = Path(r'')
        path_save = Path(r'F:\Results (paper)\2025_incTrace')

        cond0 = 'control'
        cond1 = 'trace'

        min_trace_interval_stable_numb_trials = 10
        max_trace_interval_stable_numb_trials = 10

        us_latency_trace_min = 0.5  # s
        us_latency_trace_max = 3  # s

        #! Careful with the number of catch trials!
        number_us_trials_increasing_trace = 26

        cs_duration = 10 # s

        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : (0,174,239), 'name' : 'Control', 'name in original path' : 'control'},
                  cond1 : {'color' : (255, 135, 94), 'name' : 'incTrace', 'name in original path' : 'increasingTrace', 'US latency' :
           cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace_max, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace_max] * max_trace_interval_stable_numb_trials)}
                  }

        relevant_trial_numbers = np.array([10, 20, 60, 70, 90])
        where_v_lines_1 = [10.5, 60.5]
        relevant_session_numbers = np.array([0, 3, 7])
        where_v_lines_2 = np.array([0.5, 5.5])

        cr_window = 13  # s


        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78
        time_aft_last_trial = 1 # min

        catch_trials_training = [25, 39, 53, 59]

        catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]


        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],
                                [*range(65,75)],
                                [*range(75,85)],
                                [*range(85,95)],
                                ]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                [*range(55,64)]]

        names_cs_blocks_10 = [
            'Pre-train',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
            'Test 1',
            'Test 2',
            'Test 3']

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5'
        ]



        trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65,95)]

        trials_us_blocks_phases = [np.arange(15,65)]

        names_cs_blocks_phases = [
            'Pre',
            'Train',
            'Test'
        ]

        names_us_blocks_phases = [
            'Train'
        ]

    case 'all3sTrace':

    #! End February-March 2023 comparison between 3-s fixed trace and 3-s increasing trace

        #* Path where the raw data is.
        path_home = Path(r'')
        path_save = Path(r'F:\Results (paper)\2025_3sTrace')

        cond0 = 'control'
        cond1 = 'trace'

        cond_dict = {cond0 : {'color' : (0,174,239), 'name' : 'Control', 'name in original path' : 'control'},
                  cond1 : {'color' :  (241,90,41), 'name' : '3sTrace', 'name in original path' : 'trace', 'US latency' : [9]*46}}

        cs_duration = 10 # s
        cr_window = 13  # s

        relevant_trial_numbers = np.array([10, 20, 60, 70, 90])
        where_v_lines_1 = [10.5, 60.5]
        relevant_session_numbers = np.array([0, 3, 7])
        where_v_lines_2 = np.array([0.5, 5.5])



        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78
        time_aft_last_trial = 1 # min

        catch_trials_training = [25, 39, 53, 59]

        catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]


        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],
                                [*range(65,75)],
                                [*range(75,85)],
                                [*range(85,95)],
                                ]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                [*range(55,64)]]

        names_cs_blocks_10 = [
            'Pre-train',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
            'Test 1',
            'Test 2',
            'Test 3']

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5'
        ]



        trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65,95)]

        trials_us_blocks_phases = [np.arange(15,65)]

        names_cs_blocks_phases = [
            'Pre',
            'Train',
            'Test'
        ]

        names_us_blocks_phases = [
            'Train'
        ]

    case 'fixedTraceFullyReinforced':

    #! End February-March 2023 comparison between 3-s fixed trace and 3-s increasing trace

        #* Path where the raw data is.
        path_home = Path(r'D:\2023 08_Fully reinforced fixed 3-s trace vs partially reinforced\All')
        path_save = Path(r'F:\Results (paper)\2023 08_Fully reinforced fixed 3-s trace vs partially reinforced\All')

        cond0 = 'control'
        cond1 = 'trace'

        min_trace_interval_stable_numb_trials = 10
        max_trace_interval_stable_numb_trials = 10

        us_latency_trace_min = 0.5  # s
        us_latency_trace_max = 3  # s

        #! Careful with the number of catch trials!
        number_us_trials_increasing_trace = 26

        cs_duration = 10 # s

        cond_dict = {cond0 : {'color' : (0,174,239), 'name' : 'Control', 'name in original path' : 'control'},
                  cond1 : {'color' :  (241,90,41), 'name' : 'Trace (3 s)', 'name in original path' : 'trace', 'US latency' : [9]*46}}


        relevant_trial_numbers = np.array([10, 20, 60, 70, 90])
        where_v_lines_1 = [10.5, 60.5]
        relevant_session_numbers = np.array([0, 3, 7])
        where_v_lines_2 = np.array([0.5, 5.5])

        cr_window = 13  # s


        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78
        time_aft_last_trial = 1 # min

        catch_trials_training = [25, 39, 53, 59]

        catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]


        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],
                                [*range(65,75)],
                                [*range(75,85)],
                                [*range(85,95)],
                                ]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                [*range(55,64)]]

        names_cs_blocks_10 = [
            'Pre-train',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
            'Test 1',
            'Test 2',
            'Test 3']

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5'
        ]



        trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65,95)]

        trials_us_blocks_phases = [np.arange(15,65)]

        names_cs_blocks_phases = [
            'Pre',
            'Train',
            'Test'
        ]

        names_us_blocks_phases = [
            'Train'
        ]

    case '10sTrace':

    #! March 2023 3-s trace vs 10-s trace 'Exp.'

        #* Path where the raw data is.
        path_home = Path(r'D:\2023 03_3-s vs 10-s fixed trace\Raw data')
        path_save = Path(r'F:\Results (paper)\2023 03_3-s vs 10-s fixed trace')


        cond0 = 'control'
        cond1 = '3sfixedtrace'
        cond2 = 'trace'

        cs_duration = 10 # s

        #* Colors for data in the plots.

        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : (0,174,239), 'name' : 'Control', 'name in original path' : 'control'},
                  cond1 : {'color' : (241,90,41), 'name' : 'Trace (3 s)', 'name in original path' : '3sFixedTrace', 'US latency' : [13]*46},
                  cond2 : {'color' : (145, 54, 25), 'name' : '10sTrace', 'name in original path' : '10sFixedTrace', 'US latency' : [20]*46}
                  }

        relevant_trial_numbers = np.array([10, 20, 60, 70, 90])
        where_v_lines_1 = [10.5, 60.5]
        relevant_session_numbers = np.array([0, 3, 7])
        where_v_lines_2 = np.array([0.5, 5.5])

        cr_window = 13  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78
        time_aft_last_trial = 1 # min

        catch_trials_training = [25, 39, 53, 59]

        catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]



        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],
                                [*range(65,75)],
                                [*range(75,85)],
                                [*range(85,95)],
                                ]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                [*range(55,64)]]

        names_cs_blocks_10 = [
            'Pre-train',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
            'Test 1',
            'Test 2',
            'Test 3']

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5'
        ]



        trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65,95)]

        trials_us_blocks_phases = [np.arange(15,65)]

        names_cs_blocks_phases = [
            'Pre',
            'Train',
            'Test'
        ]

        names_us_blocks_phases = [
            'Train'
        ]

    case 'all10sTrace':

        #* Path where the raw data is.
        path_home = Path(r'')
        path_save = Path(r'F:\Results (paper)\2025_10sTrace')

        cond0 = 'control'
        cond1 = 'trace'

        cs_duration = 10 # s

        #* Colors for data in the plots.

        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : (0,174,239), 'name' : 'Control', 'name in original path' : 'control'}, cond1 : {'color' : (145, 54, 25), 'name' : '10sTrace', 'name in original path' : '10sFixedTrace', 'US latency' : [20]*46}}

        relevant_trial_numbers = np.array([10, 20, 60, 70, 90])
        where_v_lines_1 = [10.5, 60.5]
        relevant_session_numbers = np.array([0, 3, 7])
        where_v_lines_2 = np.array([0.5, 5.5])

        cr_window = 13  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78
        time_aft_last_trial = 1 # min

        catch_trials_training = [25, 39, 53, 59]

        catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]



        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],
                                [*range(65,75)],
                                [*range(75,85)],
                                [*range(85,95)],
                                ]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                [*range(55,64)]]

        names_cs_blocks_10 = [
            'Pre-train',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
            'Test 1',
            'Test 2',
            'Test 3']

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5'
        ]

        trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65,95)]

        trials_us_blocks_phases = [np.arange(15,65)]

        names_cs_blocks_phases = [
            'Pre',
            'Train',
            'Test'
        ]

        names_us_blocks_phases = [
            'Train'
        ]

    case 'delaymk801':

    #! November 2022 basic delay and delay 'Exp.'

        #* Path where the raw data is.
        # path_home = Path(r'I:\Joaquim\data')
        path_home = Path(r'D:\2024 06_Delay with MK 801 (10 vs 100 μM)')
        path_save = Path(r'F:\Results (paper)\2024 06_Delay with MK 801 (10 vs 100 μM)')

        cond0 = 'controlnomk801'
        cond1 = 'delaynomk801'

        cond2 = 'controllowmk801'
        cond3 = 'delaylowmk801'

        cond4 = 'controlhighmk801'
        cond5 = 'delayhighmk801'

        min_delay_interval_stable_numb_trials = 10
        max_delay_interval_stable_numb_trials = 10

        # us_latency_delay = 3  # s

        #! Careful with the number of catch trials!
        number_us_trials_increasing_delay = 26

        cs_duration = 10 # 10 s

        #TODO add more stuff to this dict.
        cond_dict = {
                    cond0 : {'color' : (0,174,239), 'name' : 'Ctrl 0 μM', 'name in original path' : 'controlnomk801'},
                    cond1 : {'color' : (236,0,140), 'name' : 'Delay 0 μM', 'name in original path' : 'delaynomk801', 'US latency' : [9]*46},
                    cond2 : {'color' : (0, 139, 191), 'name' : 'Ctrl 10 μM', 'name in original path' : 'controllowmk801'},
                    cond3 : {'color' : (189, 0, 112), 'name' : 'Delay 10 μM', 'name in original path' : 'delaylowmk801', 'US latency' : [9]*46},
                    cond4 : {'color' : (0, 104, 143), 'name' : 'Ctrl 100 μM', 'name in original path' : 'controlhighmk801'},
                    cond5 : {'color' : (142, 0, 84), 'name' : 'Delay 100 μM', 'name in original path' : 'delayhighmk801', 'US latency' : [9]*46}}

        relevant_trial_numbers = np.array([10, 20, 60, 70, 90])
        where_v_lines_1 = [10.5, 60.5]
        relevant_session_numbers = np.array([0, 3, 7])
        where_v_lines_2 = np.array([0.5, 5.5])

#! Exception
        cr_window = 9  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78
        time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        catch_trials_training = [25, 39, 53, 59]

        catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]
        # list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))



        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],
                                [*range(65,75)],
                                [*range(75,85)],
                                [*range(85,95)],
                                ]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                [*range(55,64)]]

        names_cs_blocks_10 = [
            'Pre-train',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
            'Test 1',
            'Test 2',
            'Test 3']

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5']



        trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65,95)]

        trials_us_blocks_phases = [np.arange(15,64)]

        names_cs_blocks_phases = [
            'Pre',
            'Train',
            'Test']

        names_us_blocks_phases = [
            'Train']

    case 'tracemk801':

    #! April 2023 trace with μM MK801 'Exp.'

        #* Path where the raw data is.
        path_home = Path(r'D:\2023 04_Fixed 3-s trace with mk 801 (100 μM)\Raw data')
        path_save = Path(r'F:\Results (paper)\2023 04_Fixed 3-s trace with mk 801 (100 μM)')

        cond0 = 'controlnomk801'
        cond1 = 'tracenomk801'
        cond2 = 'controlmk801'
        cond3 = 'tracemk801'

        min_trace_interval_stable_numb_trials = 10
        max_trace_interval_stable_numb_trials = 10

        # us_latency_trace = 3  # s

        #! Careful with the number of catch trials!
        number_us_trials_increasing_trace = 26

        cs_duration = 10 # s

        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : (0,174,239), 'name' : 'Control 0 μM', 'name in original path' : 'controlnomk801'},
                    cond1 : {'color' : (241, 90, 41), 'name' : 'Trace 0 μM', 'name in original path' : 'tracenomk801', 'US latency' : [13]*46
        #    cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace] * max_trace_interval_stable_numb_trials)
           },
           cond2 : {'color' : (0, 139, 191), 'name' : 'Control 100 μM', 'name in original path' : 'controlmk801',},
           cond3 : {'color' : (193, 72, 33), 'name' : 'Trace 100 μM', 'name in original path' : 'tracemk801',
           'US latency' : [13]*46
        #    cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace] * max_trace_interval_stable_numb_trials)
           },}

        relevant_trial_numbers = np.array([10, 20, 60, 70, 90])
        where_v_lines_1 = [10.5, 60.5]
        relevant_session_numbers = np.array([0, 3, 7])
        where_v_lines_2 = np.array([0.5, 5.5])

        cr_window = 10  # s


        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78
        time_aft_last_trial = 1 # min

        catch_trials_plot = list(np.arange(5,15)) + [25, 39, 53, 59] + list(np.arange(65,95))



        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],
                                [*range(65,75)],
                                [*range(75,85)],
                                [*range(85,95)],
                                ]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                [*range(55,64)]]

        names_cs_blocks_10 = [
            'Pre-train',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
            'Test 1',
            'Test 2',
            'Test 3']

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5'
        ]



        trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65,95)]

        trials_us_blocks_phases = [np.arange(15,65)]

        names_cs_blocks_phases = [
            'Pre',
            'Train',
            'Test'
        ]

        names_us_blocks_phases = [
            'Train'
        ]


    case 'delayLongTermSpacedVsMassed-1':

    #! August 2023 trace with massed or spaced schedule

        #* Path where the raw data is.
        path_home = Path(r'D:\2023 10_Long-term memory spaced vs massed delay\Block 1\Raw data')
        path_save = Path(r'F:\Results (paper)\2023 10_Long-term memory spaced vs massed delay\Block 1')

        cond0 = 'controlmassed'
        cond1 = 'controlspaced'
        cond2 = 'delaymassed'
        cond3 = 'delayspaced'

        us_latency_delay = -1  # s

        cs_duration = 10  # s

        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : [58, 129, 195], 'name' : 'Control Massed', 'name in original path' : 'controlmassed'},
                  cond1 : {'color' :  [245, 130, 48], 'name' : 'Control Spaced', 'name in original path' : 'controlspaced'},
                  cond2 : {'color' : [0, 0, 139], 'name' : 'Delay Massed', 'name in original path' : 'delaymassed'},
                  cond3 : {'color' : [246, 190, 0], 'name' : 'Delay Spaced', 'name in original path' : 'delayspaced'}
                  }

        cr_window = 9  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 64
        min_number_us_trials = 59

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 64
        min_number_us_trials = 59
        # time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        # catch_trials_training = [25, 39, 53, 59]

        # catch_trials_plot = list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))

        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],
                                # [*range(65,70)],
                                # [*range(75,85)],
                                # [*range(85,95)],
                                ]


        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                [*range(55,64)]]

                                # [[*range(13,22)],
                                # [*range(23,32)],
                                # [*range(32,41)],
                                # [*range(41,50)],
                                # [*range(50,59)]]

        names_cs_blocks_10 = [
            'Pre-train',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
        ]

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5'
        ]



        trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65)]

        trials_us_blocks_phases = [np.arange(18,64)]
        # [np.arange(15,65)]

        names_cs_blocks_phases = [
            'Pre',
            'Train',
        ]

        names_us_blocks_phases = [
            'Train'
        ]

    case 'delayLongTermSpacedVsMassed-2':

    #! August 2023 delay with massed or spaced schedule

        #* Path where the raw data is.
        path_home = Path(r'D:\2023 10_Long-term memory spaced vs massed delay\Block 2\Raw data')
        path_save = Path(r'F:\Results (paper)\2023 10_Long-term memory spaced vs massed delay\Block 2')

        cond0 = 'controlmassed'
        cond1 = 'controlspaced'
        cond2 = 'delaymassed'
        cond3 = 'delayspaced'

        us_latency_delay = -1  # s

        cs_duration = 10  # s

        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : [58, 129, 195], 'name' : 'Control Massed', 'name in original path' : 'controlmassed'},
                  cond1 : {'color' :  [245, 130, 48], 'name' : 'Control Spaced', 'name in original path' : 'controlspaced'},
                  cond2 : {'color' : [0, 0, 139], 'name' : 'delay Massed', 'name in original path' : 'delaymassed'},
                  cond3 : {'color' : [246, 190, 0], 'name' : 'delay Spaced', 'name in original path' : 'delayspaced'}
                  }

        cr_window = 9  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 60
        min_number_us_trials = 44

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 60
        min_number_us_trials = 44
        # time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        catch_trials_training = [25, 39, 53, 59]

        # catch_trials_plot = list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))

        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(1,11)],
                                [*range(11,21)],
                                [*range(21,31)],
                                [*range(31,41)],
                                [*range(41,51)],
                                [*range(51,61)],
                                # [*range(61,71)],
                                # [*range(71,81)],
                                # [*range(81,91)],
                                # [*range(91,101)],
                                # [*range(101,111)]
                                ]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [
                                [*range(17,27)],
                                 [*range(27,36)],
                                [*range(36,45)],
                                ]

        names_cs_blocks_10 = [
                        'Test 1',
                        'Test 2',
                        'Test 3',
                        'Retrain 1',
                        'Retrain 2',
                        'Retrain 3',
                        # 'Retrain 4',
                        # 'Retrain 5',
                        # 'Post-retrain 1',
                        # 'Post-retrain 2',
                        # 'Post-retrain 3'
                        ]

        names_us_blocks_10 = [
                        'Retrain 1',
                        'Retrain 2',
                        'Retrain 3',
                        # 'Retrain 4',
                        # 'Retrain 5'
                        ]



        trials_cs_blocks_phases = [np.arange(1,31), np.arange(31,61),]
                            #  np.arange(81,111)]

        trials_us_blocks_phases = [np.arange(16,45)]

        names_cs_blocks_phases = ['Test',
                            'Retrain',]
                            # 'Post-retrain']

        names_us_blocks_phases = ['Retrain']

    case 'delayLongTermSpacedVsMassed-all':

    #! August 2023 delay with massed or spaced schedule

        #* Path where the raw data is.
        path_home = Path(r'D:\2023 10_Long-term memory spaced vs massed delay\All blocks\Raw data')
        path_save = Path(r'F:\Results (paper)\2023 10_Long-term memory spaced vs massed delay\All blocks')

        cond0 = 'delaymassed'
        cond1 = 'delayspaced'

        us_latency_delay = -1  # s

        cs_duration = 10  # s

        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : (236,0,140), 'name' : 'delay Massed', 'name in original path' : 'delaymassed'},
                    cond1 : {'color' : sns.color_palette('colorblind')[-4], 'name' : 'delay Spaced', 'name in original path' : 'delayspaced'}}

        cr_window = 9  # s

        relevant_trial_numbers = np.array([10, 20, 60, 70, 90])
        where_v_lines_1 = [10.5, 60.5]
        relevant_session_numbers = np.array([0, 3, 7, 10])
        where_v_lines_2 = np.array([0.5, 5.5, 8.5])


        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 64 + 60
        min_number_us_trials = 59 + 44

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 64 + 60
        min_number_us_trials = 59 + 44
        # time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        # catch_trials_training = [25, 39, 53, 59] + [25 + 64, 39 + 64, 53 + 64, 59 + 64]

        # catch_trials_plot = list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))

        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],

                                [*range(65,75)],
                                [*range(75,85)],
                                [*range(85,95)],
                                [*range(95,105)],
                                [*range(105,115)],
                                [*range(115,125)],
                                # [*range(125,135)],
                                # [*range(135,145)],
                                # [*range(145,155)],
                                # [*range(155,165)],
                                # [*range(165,175)]
                                ]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                [*range(55,64)],

                                [*range(81,91)],
                                 [*range(91,100)],
                                [*range(100,109)]
                                ]

        names_cs_blocks_10 = [
            'Pre-train',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
            'Test 1',
            'Test 2',
            'Test 3',
            'Retrain 1',
            'Retrain 2',
            'Retrain 3',
            # 'Retrain 4',
            # 'Retrain 5',
            # 'Post-retrain 1',
            # 'Post-retrain 2',
            # 'Post-retrain 3'
        ]

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
            'Retrain 1',
            'Retrain 2',
            'Retrain 3',
            # 'Retrain 4',
            # 'Retrain 5'
        ]



        trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65, 95), np.arange(95, 125)]

        trials_us_blocks_phases = [np.arange(18,64), np.arange(81, 109)]



        names_cs_blocks_phases = [
            'Pre',
            'Train',
            'Test',
            'Retrain',
            # 'Post-retrain'
            ]

        names_us_blocks_phases = [
            'Train',
            'Retrain']



    case 'traceLongTermSpacedVsMassed-1':

    #! August 2023 trace with massed or spaced schedule

        #* Path where the raw data is.
        path_home = Path(r'D:\2023 08-09_Long-term memory spaced vs massed fixed 3-s trace\Block 1\Raw data')
        path_save = Path(r'F:\Results (paper)\2023 08-09_Long-term memory spaced vs massed fixed 3-s trace\Block 1')

        cond0 = 'controlmassed'
        cond1 = 'controlspaced'
        cond2 = 'delaymassed'
        cond3 = 'delayspaced'

        us_latency_delay = -1  # s

        cs_duration = 10  # s

        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : [58, 129, 195], 'name' : 'Control Massed', 'name in original path' : 'controlmassed'},
                  cond1 : {'color' :  [245, 130, 48], 'name' : 'Control Spaced', 'name in original path' : 'controlspaced'},
                  cond2 : {'color' : [0, 0, 139], 'name' : 'Delay Massed', 'name in original path' : 'delaymassed'},
                  cond3 : {'color' : [246, 190, 0], 'name' : 'Delay Spaced', 'name in original path' : 'delayspaced'}
                  }

        cr_window = 9  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 64
        min_number_us_trials = 59

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 64
        min_number_us_trials = 59
        # time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        # catch_trials_training = [25, 39, 53, 59]

        # catch_trials_plot = list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))

        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],
                                # [*range(65,70)],
                                # [*range(75,85)],
                                # [*range(85,95)],
                                ]


        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                [*range(55,64)]]

                                # [[*range(13,22)],
                                # [*range(23,32)],
                                # [*range(32,41)],
                                # [*range(41,50)],
                                # [*range(50,59)]]

        names_cs_blocks_10 = [
            'Pre-train',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
        ]

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5'
        ]



        trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65)]

        trials_us_blocks_phases = [np.arange(18,64)]
        # [np.arange(15,65)]

        names_cs_blocks_phases = [
            'Pre',
            'Train',
        ]

        names_us_blocks_phases = [
            'Train'
        ]

    case 'traceLongTermSpacedVsMassed-2':

    #! August 2023 delay with massed or spaced schedule

        #* Path where the raw data is.
        path_home = Path(r'D:\2023 08-09_Long-term memory spaced vs massed fixed 3-s trace\Block 2\Raw data')
        path_save = Path(r'F:\Results (paper)\2023 08-09_Long-term memory spaced vs massed fixed 3-s trace\Block 2')

        cond0 = 'controlmassed'
        cond1 = 'controlspaced'
        cond2 = 'delaymassed'
        cond3 = 'delayspaced'

        us_latency_delay = -1  # s

        cs_duration = 10  # s

        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : [58, 129, 195], 'name' : 'Control Massed', 'name in original path' : 'controlmassed'},
                  cond1 : {'color' :  [245, 130, 48], 'name' : 'Control Spaced', 'name in original path' : 'controlspaced'},
                  cond2 : {'color' : [0, 0, 139], 'name' : 'delay Massed', 'name in original path' : 'delaymassed'},
                  cond3 : {'color' : [246, 190, 0], 'name' : 'delay Spaced', 'name in original path' : 'delayspaced'}
                  }

        cr_window = 9  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 60
        min_number_us_trials = 44

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 60
        min_number_us_trials = 44
        # time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        catch_trials_training = [25, 39, 53, 59]

        # catch_trials_plot = list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))

        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(1,11)],
                                [*range(11,21)],
                                [*range(21,31)],
                                [*range(31,41)],
                                [*range(41,51)],
                                [*range(51,61)],
                                # [*range(61,71)],
                                # [*range(71,81)],
                                # [*range(81,91)],
                                # [*range(91,101)],
                                # [*range(101,111)]
                                ]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [
                                [*range(17,27)],
                                 [*range(27,36)],
                                [*range(36,45)],
                                ]

        names_cs_blocks_10 = [
                        'Test 1',
                        'Test 2',
                        'Test 3',
                        'Retrain 1',
                        'Retrain 2',
                        'Retrain 3',
                        # 'Retrain 4',
                        # 'Retrain 5',
                        # 'Post-retrain 1',
                        # 'Post-retrain 2',
                        # 'Post-retrain 3'
                        ]

        names_us_blocks_10 = [
                        'Retrain 1',
                        'Retrain 2',
                        'Retrain 3',
                        # 'Retrain 4',
                        # 'Retrain 5'
                        ]



        trials_cs_blocks_phases = [np.arange(1,31), np.arange(31,61),]
                            #  np.arange(81,111)]

        trials_us_blocks_phases = [np.arange(16,45)]

        names_cs_blocks_phases = ['Test',
                            'Retrain',]
                            # 'Post-retrain']

        names_us_blocks_phases = ['Retrain']

    case 'traceLongTermSpacedVsMassed-all':

    #! August 2023 delay with massed or spaced schedule

        #* Path where the raw data is.
        path_home = Path(r'')
        path_save = Path(r'F:\Results (paper)\2023 08-09_Long-term memory spaced vs massed fixed 3-s trace\All blocks')

        cond0 = 'tracemassed'
        cond1 = 'tracespaced'

        us_latency_trace = 3  # s

        cs_duration = 10  # s

        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : (241, 90, 41), 'name' : 'trace Massed', 'name in original path' : 'tracemassed'},
                    cond1 : {'color' : (241,90,41), 'name' : 'trace Spaced', 'name in original path' : 'tracespaced'}}

        cr_window = 9  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 64 + 60
        min_number_us_trials = 59 + 44

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 64 + 60
        min_number_us_trials = 59 + 44
        # time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        # catch_trials_training = [25, 39, 53, 59] + [25 + 64, 39 + 64, 53 + 64, 59 + 64]

        # catch_trials_plot = list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))

        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],

                                [*range(65,75)],
                                [*range(75,85)],
                                [*range(85,95)],
                                [*range(95,105)],
                                [*range(105,115)],
                                [*range(115,125)],
                                # [*range(125,135)],
                                # [*range(135,145)],
                                # [*range(145,155)],
                                # [*range(155,165)],
                                # [*range(165,175)]
                                ]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                [*range(55,64)],

                                [*range(81,91)],
                                 [*range(91,100)],
                                [*range(100,109)]
                                ]

        names_cs_blocks_10 = [
            'Pre-train',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
            'Test 1',
            'Test 2',
            'Test 3',
            'Retrain 1',
            'Retrain 2',
            'Retrain 3',
            # 'Retrain 4',
            # 'Retrain 5',
            # 'Post-retrain 1',
            # 'Post-retrain 2',
            # 'Post-retrain 3'
        ]

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
            'Retrain 1',
            'Retrain 2',
            'Retrain 3',
            # 'Retrain 4',
            # 'Retrain 5'
        ]



        trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65, 95), np.arange(95, 125)]

        trials_us_blocks_phases = [np.arange(18,64), np.arange(81, 109)]



        names_cs_blocks_phases = [
            'Pre',
            'Train',
            'Test',
            'Retrain',
            # 'Post-retrain'
            ]

        names_us_blocks_phases = [
            'Train',
            'Retrain']



    case 'tracePartialReinforcement':

    #! August 2023 trace with partial and full reinforcement

        #* Path where the raw data is.
        path_home = Path(r'D:\2023 08_Fully reinforced fixed 3-s trace vs partially reinforced\All')
        path_save = Path(r'F:\Results (paper)\2023 08_Fully reinforced fixed 3-s trace vs partially reinforced')

        cond0 = 'control'
        cond1 = 'tracepartiallyreinforced'
        cond2 = 'tracefullyreinforced'

        us_latency_trace = 3  # s

        cr_window = 13  # s

        cs_duration = 10 # s

        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : (0,174,239), 'name' : 'Control', 'name in original path' : 'control'},
                  cond1 : {'color' :  (236,0,140), 'name' : 'Trace Partially Reinforced', 'name in original path' : 'tracepartiallyreinforced'},
                  cond2 : {'color' : sns.color_palette('colorblind')[-4], 'name' : 'Trace Fully Reinforced', 'name in original path' : 'tracefullyreinforced'}}

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 79

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 79
        time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        # catch_trials_training = [25, 39, 53, 59]

        # catch_trials_plot = list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))

        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],
                                [*range(65,75)],
                                [*range(75,85)],
                                [*range(85,95)],
                                ]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                [*range(55,64)]]

        names_cs_blocks_10 = [
            'Pre-train',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
            'Test 1',
            'Test 2',
            'Test 3']

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5'
        ]



        trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65,95)]

        trials_us_blocks_phases = [np.arange(18,65)]

        names_cs_blocks_phases = [
            'Pre',
            'Train',
            'Test'
        ]

        names_us_blocks_phases = [
            'Train'
        ]



    case 'tracepuromycinshort':

    #! May 2023 trace with μM MK801 'Exp.'

        #* Path where the raw data is.
        path_home = Path(r'D:\2023 05 & 08_Fixed 3-s trace with Puromycin (5 mg per L)\Raw data')
        path_save = Path(r'F:\Results (paper)\2023 05 & 08_Fixed 3-s trace with Puromycin (5 mg per L)')

        cond0 = 'control'
        cond1 = 'trace'
        # cond2 = 'controlpuromycin'
        # cond3 = 'tracepuromycin'

        cs_duration = 10 # s

        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : (0,174,239), 'name' : 'Control', 'name in original path' : 'control'},
               cond1 : {'color' : (241, 90, 41), 'name' : 'Trace', 'name in original path' : 'trace', 'US latency' : [13]*46},

                #   cond2 : {'color' :  (0, 139, 191), 'name' : 'Control Puromycin', 'name in original path' : 'controlpuromycin'},

                   #   cond3 : {'color' : (193, 72, 33), 'name' : 'Trace Puromycin', 'name in original path' : 'tracepuromycin', 'US latency' : [13]*46},
               }


        relevant_trial_numbers = np.array([10, 20, 60, 70, 90])
        where_v_lines_1 = [10.5, 60.5]
        relevant_session_numbers = np.array([0, 3, 7])
        where_v_lines_2 = np.array([0.5, 5.5])


        cr_window = 13  # s


        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78
        time_aft_last_trial = 1 # min

        catch_trials_plot = list(np.arange(5,15)) + [25, 39, 53, 59] + list(np.arange(65,95))



        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],
                                [*range(65,75)],
                                [*range(75,85)],
                                [*range(85,95)],
                                ]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                [*range(55,64)]]

        names_cs_blocks_10 = [
            'Pre-train',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
            'Test 1',
            'Test 2',
            'Test 3']

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5'
        ]



        trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65,95)]

        trials_us_blocks_phases = [np.arange(15,65)]

        names_cs_blocks_phases = [
            'Pre',
            'Train',
            'Post'
        ]

        names_us_blocks_phases = [
            'Train'
        ]




    case 'delayLongTermSpacedpuromycin5mg-1':

    #! August 2023 trace with massed or spaced schedule

        #* Path where the raw data is.
        path_home = Path(r'D:\2023 11_Long-term memory control, delay spaced training with Puromycin (5 mg per L)\2\Block 1\Raw data')
        path_save = Path(r'F:\Results (paper)\2023 11_Long-term memory control, delay spaced training with Puromycin (5 mg per L)\2\Block 1')

        cond0 = 'controlspacednopuromycin'
        cond1 = 'controlspacedpuromycin'
        cond2 = 'delayspacednopuromycin'
        cond3 = 'delayspacedpuromycin'

        cs_duration = 10  # s



        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : (0, 174, 239), 'name' : 'Control Spaced No Puromycin', 'name in original path' : 'controlspacednopuromycin'},
                  cond2 : {'color' :  (236, 0, 140), 'name' : 'Control Spaced Puromycin', 'name in original path' : 'controlspacedpuromycin'},
                  cond1 : {'color' : (0, 139, 191), 'name' : 'Delay Spaced No Puromycin', 'name in original path' : 'delayspacednopuromycin'},
                  cond3 : {'color' : (189, 0, 112), 'name' : 'Delay Spaced Puromycin', 'name in original path' : 'delayspacedpuromycin'}
                  }

        cr_window = 9  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 64
        min_number_us_trials = 59

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 64
        min_number_us_trials = 59
        # time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        # catch_trials_training = [25, 39, 53, 59]

        # catch_trials_plot = list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))

        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],
                                # [*range(65,70)],
                                # [*range(75,85)],
                                # [*range(85,95)],
                                ]


        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                [*range(55,64)]]

                                # [[*range(13,22)],
                                # [*range(23,32)],
                                # [*range(32,41)],
                                # [*range(41,50)],
                                # [*range(50,59)]]

        names_cs_blocks_10 = [
            'Pre-train',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
        ]

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5'
        ]



        trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65)]

        trials_us_blocks_phases = [np.arange(18,64)]
        # [np.arange(15,65)]

        names_cs_blocks_phases = [
            'Pre',
            'Train',
        ]

        names_us_blocks_phases = [
            'Train'
        ]

    case 'delayLongTermSpacedpuromycin5mg-2':

    #! August 2023 delay with massed or spaced schedule

        #* Path where the raw data is.
        path_home = Path(r'D:\2023 11_Long-term memory control, delay spaced training with Puromycin (5 mg per L)\2\Block 2\Raw data')
        path_save = Path(r'F:\Results (paper)\2023 11_Long-term memory control, delay spaced training with Puromycin (5 mg per L)\2\Block 2')
        # path_home = Path(r'D:\Pilot studies\2024 01_delay with Puromycin (10 mg per L)\Block 2\Raw data')
        # path_save = Path(r'D:\Pilot studies\2024 01_delay with Puromycin (10 mg per L)\Block 2')

        cond0 = 'controlspacednopuromycin'
        cond1 = 'controlspacedpuromycin'
        cond2 = 'delayspacednopuromycin'
        cond3 = 'delayspacedpuromycin'

        us_latency_delay = -1  # s

        cs_duration = 10  # s


        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : (0, 174, 239), 'name' : 'Control Spaced No Puromycin', 'name in original path' : 'controlspacednopuromycin'},
                  cond1 : {'color' :  (236, 0, 140), 'name' : 'Delay Spaced No Puromycin', 'name in original path' : 'delayspacednopuromycin'},
                  cond2 : {'color' : (0, 139, 191), 'name' : 'Control Spaced Puromycin', 'name in original path' : 'controlspacedpuromycin'},
                  cond3 : {'color' : (189, 0, 112), 'name' : 'Delay Spaced Puromycin', 'name in original path' : 'delayspacedpuromycin'}
                  }

        cr_window = 9  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 60
        min_number_us_trials = 44

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 60
        min_number_us_trials = 44
        # time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        catch_trials_training = [25, 39, 53, 59]

        # catch_trials_plot = list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))

        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(1,11)],
                                [*range(11,21)],
                                [*range(21,31)],
                                [*range(31,41)],
                                [*range(41,51)],
                                [*range(51,61)],
                                # [*range(61,71)],
                                # [*range(71,81)],
                                # [*range(81,91)],
                                # [*range(91,101)],
                                # [*range(101,111)]
                                ]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [
                                [*range(17,27)],
                                 [*range(27,36)],
                                [*range(36,45)],
                                ]

        names_cs_blocks_10 = [
                        'Test 1',
                        'Test 2',
                        'Test 3',
                        'Retrain 1',
                        'Retrain 2',
                        'Retrain 3',
                        # 'Retrain 4',
                        # 'Retrain 5',
                        # 'Test-retrain 1',
                        # 'Test-retrain 2',
                        # 'Test-retrain 3'
                        ]

        names_us_blocks_10 = [
                        'Retrain 1',
                        'Retrain 2',
                        'Retrain 3',
                        # 'Retrain 4',
                        # 'Retrain 5'
                        ]



        trials_cs_blocks_phases = [np.arange(1,31), np.arange(31,61),]
                            #  np.arange(81,111)]

        trials_us_blocks_phases = [np.arange(16,45)]

        names_cs_blocks_phases = ['Test',
                            'Retrain',]
                            # 'Test-retrain']

        names_us_blocks_phases = ['Retrain']

    case 'delayLongTermSpacedpuromycin5mg-all':

    #! August 2023 delay with massed or spaced schedule

        #* Path where the raw data is.
        path_home = Path(r'D:\2023 11_Long-term memory control, delay spaced training with Puromycin (5 mg per L)\All blocks\Raw data')
        path_save = Path(r'F:\Results (paper)\2023 11_Long-term memory control, delay spaced training with Puromycin (5 mg per L)\All blocks')


        cond0 = 'controlspacednopuromycin'
        cond1 = 'delayspacednopuromycin'
        cond2 = 'controlspacedpuromycin'
        cond3 = 'delayspacedpuromycin'

        cs_duration = 10  # s

        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : (0, 174, 239), 'name' : 'Control 0 mg/L', 'name in original path' : 'controlspacednopuromycin'},
                  cond1 : {'color' :  (236, 0, 140), 'name' : 'Delay 0 mg/L', 'name in original path' : 'delayspacednopuromycin'},
                  cond2 : {'color' : (0, 139, 191), 'name' : 'Control 5 mg/L', 'name in original path' : 'controlspacedpuromycin'},
                  cond3 : {'color' : (189, 0, 112), 'name' : 'Delay 5 mg/L', 'name in original path' : 'delayspacedpuromycin'}
                  }


        relevant_trial_numbers = np.array([10, 20, 60, 70, 90])
        where_v_lines_1 = [10.5, 60.5, 90.5]
        relevant_session_numbers = np.array([0, 3, 7, 10])
        where_v_lines_2 = np.array([0.5, 5.5, 8.5])



        cr_window = 9  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 64 + 60
        min_number_us_trials = 59 + 44

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 64 + 60
        min_number_us_trials = 59 + 44
        # time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        # catch_trials_training = [25, 39, 53, 59] + [25 + 64, 39 + 64, 53 + 64, 59 + 64]

        # catch_trials_plot = list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))

        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],

                                [*range(65,75)],
                                [*range(75,85)],
                                [*range(85,95)],
                                [*range(95,105)],
                                [*range(105,115)],
                                [*range(115,125)],
                                # [*range(125,135)],
                                # [*range(135,145)],
                                # [*range(145,155)],
                                # [*range(155,165)],
                                # [*range(165,175)]
                                ]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                [*range(55,64)],

                                [*range(81,91)],
                                 [*range(91,100)],
                                [*range(100,109)]
                                ]

        names_cs_blocks_10 = [
            'Pre-train',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
            'Test 1',
            'Test 2',
            'Test 3',
            'Retrain 1',
            'Retrain 2',
            'Retrain 3',
            # 'Retrain 4',
            # 'Retrain 5',
            # 'Test-retrain 1',
            # 'Test-retrain 2',
            # 'Test-retrain 3'
        ]

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
            'Retrain 1',
            'Retrain 2',
            'Retrain 3',
            # 'Retrain 4',
            # 'Retrain 5'
        ]



        trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65, 95), np.arange(95, 125)]

        trials_us_blocks_phases = [np.arange(18,64), np.arange(81, 109)]



        names_cs_blocks_phases = [
            'Pre',
            'Train',
            'Test',
            'Re',
            # 'Test-retrain'
            ]

        names_us_blocks_phases = [
            'Train',
            'Retrain']




    case 'delayLongTermSpacedpuromycin10mg-1':

    #! August 2023 trace with massed or spaced schedule

        #* Path where the raw data is.
        path_home = Path(r'E:\Pilot studies\2024 01_Long-term memory control, delay spaced training with Puromycin (10 mg per L)\Block 1\Raw data')
        path_save = Path(r'F:\Results (paper)\2024 01_Long-term memory control, delay spaced training with Puromycin (10 mg per L)\Block 1')


        cond0 = 'delayspacednopuromycin'
        cond1 = 'delayspacedpuromycin'

        cond_dict = {cond0 : {'color' : (236, 0, 140), 'name' : 'Delay Spaced No Puromycin', 'name in original path' : 'delayspacednopuromycin'},
               cond1 : {'color' : (189, 0, 112), 'name' : 'Delay Spaced Puromycin', 'name in original path' : 'delayspacedpuromycin'}}

        cr_window = 9  # s
        us_latency_delay = -1  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 64
        min_number_us_trials = 59

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 64
        min_number_us_trials = 59
        # time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        # catch_trials_training = [25, 39, 53, 59]

        # catch_trials_plot = list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))

        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],
                                # [*range(65,70)],
                                # [*range(75,85)],
                                # [*range(85,95)],
                                ]


        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                [*range(55,64)]]

                                # [[*range(13,22)],
                                # [*range(23,32)],
                                # [*range(32,41)],
                                # [*range(41,50)],
                                # [*range(50,59)]]

        names_cs_blocks_10 = [
            'Pre-train',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
        ]

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5'
        ]



        trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65)]

        trials_us_blocks_phases = [np.arange(18,64)]
        # [np.arange(15,65)]

        names_cs_blocks_phases = [
            'Pre',
            'Train',
        ]

        names_us_blocks_phases = [
            'Train'
        ]

    case 'delayLongTermSpacedpuromycin10mg-2':

    #! August 2023 delay with massed or spaced schedule

        #* Path where the raw data is.
        path_home = Path(r'E:\Pilot studies\2024 01_Long-term memory control, delay spaced training with Puromycin (10 mg per L)\Block 2\Raw data')
        path_save = Path(r'F:\Results (paper)\2024 01_Long-term memory control, delay spaced training with Puromycin (10 mg per L)\Block 2')

        cond0 = 'delayspacednopuromycin'
        cond1 = 'delayspacedpuromycin'

        cond_dict = {cond0 : {'color' : (236, 0, 140), 'name' : 'Delay Spaced No Puromycin', 'name in original path' : 'delayspacednopuromycin'},
               cond1 : {'color' : (189, 0, 112), 'name' : 'Delay Spaced Puromycin', 'name in original path' : 'delayspacedpuromycin'}}


        cr_window = 9  # s
        us_latency_delay = -1  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 60
        min_number_us_trials = 44

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 60
        min_number_us_trials = 44
        # time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        catch_trials_training = [25, 39, 53, 59]

        # catch_trials_plot = list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))

        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(1,11)],
                                [*range(11,21)],
                                [*range(21,31)],
                                [*range(31,41)],
                                [*range(41,51)],
                                [*range(51,61)],
                                # [*range(61,71)],
                                # [*range(71,81)],
                                # [*range(81,91)],
                                # [*range(91,101)],
                                # [*range(101,111)]
                                ]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [
                                [*range(17,27)],
                                 [*range(27,36)],
                                [*range(36,45)],
                                ]

        names_cs_blocks_10 = [
                        'Test 1',
                        'Test 2',
                        'Test 3',
                        'Retrain 1',
                        'Retrain 2',
                        'Retrain 3',
                        # 'Retrain 4',
                        # 'Retrain 5',
                        # 'Test-retrain 1',
                        # 'Test-retrain 2',
                        # 'Test-retrain 3'
                        ]

        names_us_blocks_10 = [
                        'Retrain 1',
                        'Retrain 2',
                        'Retrain 3',
                        # 'Retrain 4',
                        # 'Retrain 5'
                        ]



        trials_cs_blocks_phases = [np.arange(1,31), np.arange(31,61),]
                            #  np.arange(81,111)]

        trials_us_blocks_phases = [np.arange(16,45)]

        names_cs_blocks_phases = ['Test',
                            'Retrain',]
                            # 'Test-retrain']

        names_us_blocks_phases = ['Retrain']

    case 'delayLongTermSpacedpuromycin10mg-all':

    #! August 2023 delay with massed or spaced schedule

        #* Path where the raw data is.
        path_home = Path(r'E:\Pilot studies\2024 01_Long-term memory control, delay spaced training with Puromycin (10 mg per L)\All blocks\Raw data')
        path_save = Path(r'F:\Results (paper)\2024 01_Long-term memory control, delay spaced training with Puromycin (10 mg per L)\All blocks')

        cond0 = 'delayspacednopuromycin'
        cond1 = 'delayspacedpuromycin'

        cond_dict = {cond0 : {'color' : (236, 0, 140), 'name' : 'Delay Spaced No Puromycin', 'name in original path' : 'delayspacednopuromycin'},
               cond1 : {'color' : (189, 0, 112), 'name' : 'Delay Spaced Puromycin', 'name in original path' : 'delayspacedpuromycin'}}

        cr_window = 9  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 64 + 60
        min_number_us_trials = 59 + 44

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 64 + 60
        min_number_us_trials = 59 + 44
        # time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        # catch_trials_training = [25, 39, 53, 59] + [25 + 64, 39 + 64, 53 + 64, 59 + 64]

        # catch_trials_plot = list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))

        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],

                                [*range(65,75)],
                                [*range(75,85)],
                                [*range(85,95)],
                                [*range(95,105)],
                                [*range(105,115)],
                                [*range(115,125)],
                                # [*range(125,135)],
                                # [*range(135,145)],
                                # [*range(145,155)],
                                # [*range(155,165)],
                                # [*range(165,175)]
                                ]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                [*range(55,64)],

                                [*range(81,91)],
                                 [*range(91,100)],
                                [*range(100,109)]
                                ]

        names_cs_blocks_10 = [
            'Pre-train',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
            'Test 1',
            'Test 2',
            'Test 3',
            'Retrain 1',
            'Retrain 2',
            'Retrain 3',
            # 'Retrain 4',
            # 'Retrain 5',
            # 'Test-retrain 1',
            # 'Test-retrain 2',
            # 'Test-retrain 3'
        ]

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
            'Retrain 1',
            'Retrain 2',
            'Retrain 3',
            # 'Retrain 4',
            # 'Retrain 5'
        ]



        trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65, 95), np.arange(95, 125)]

        trials_us_blocks_phases = [np.arange(18,64), np.arange(81, 109)]



        names_cs_blocks_phases = [
            'Pre',
            'Train',
            'Test',
            'Retrain',
            # 'Test-retrain'
            ]

        names_us_blocks_phases = [
            'Train',
            'Retrain']






    case 'delayLongTermNew-1':

    #! July 2024 trace with spaced schedule

        #* Path where the raw data is.
        path_home = Path(r'D:\2024 07_Long-term memory control, delay new\All\Block 1')
        path_save = Path(r'F:\Results (paper)\2024 07_Long-term memory control, delay new\All\Block 1')


        cond0 = 'control'
        cond1 = 'delay'

        cs_duration = 10  # s

        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : (0, 174, 239), 'name' : 'Control', 'name in original path' : 'control'},
                  cond1 : {'color' :  (236, 0, 140), 'name' : 'Delay', 'name in original path' : 'delay'},
                  }

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 64
        min_number_us_trials = 59

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 64
        min_number_us_trials = 59
        # time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        # catch_trials_training = [25, 39, 53, 59]

        # catch_trials_plot = list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))

        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],
                                ]


        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                [*range(55,64)]]

        names_cs_blocks_10 = [
            'Pre-train',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
        ]

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5'
        ]



        trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65)]

        trials_us_blocks_phases = [np.arange(18,64)]
        # [np.arange(15,65)]

        names_cs_blocks_phases = [
            'Pre',
            'Train',
        ]

        names_us_blocks_phases = [
            'Train'
        ]

    case 'delayLongTermNew-2':

        #* Path where the raw data is.
        path_home = Path(r'D:\2024 07_Long-term memory control, delay new\All\Block 2')
        path_save = Path(r'F:\Results (paper)\2024 07_Long-term memory control, delay new\All\Block 2')


        cond0 = 'control'
        cond1 = 'delay'

        cs_duration = 10  # s

        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : (0, 174, 239), 'name' : 'Control', 'name in original path' : 'control'},
                  cond1 : {'color' :  (236, 0, 140), 'name' : 'Delay', 'name in original path' : 'delay'},
                  }
        cr_window = 9  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 70
        min_number_us_trials = 52

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 70
        min_number_us_trials = 52
        # time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        # catch_trials_training = [25, 39, 53, 59]

        # catch_trials_plot = list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))

        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(1,11)],
                                [*range(11,21)],
                                [*range(21,31)],
                                [*range(31,41)],
                                [*range(41,51)],
                                [*range(51,61)],
                                [*range(61,71)],
                                [*range(71,81)],
                                ]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [
                                [*range(16,26)],
                                [*range(26,35)],
                                [*range(35,44)],
                                [*range(44,53)],
                                [*range(53,62)],
                                ]

        names_cs_blocks_10 = [
                        'Test 1',
                        'Test 2',
                        'Test 3',
                        'Retrain 1',
                        'Retrain 2',
                        'Retrain 3',
                        'Retrain 4',
                        'Retrain 5',
                        ]

        names_us_blocks_10 = [
                        'Retrain 1',
                        'Retrain 2',
                        'Retrain 3',
                        'Retrain 4',
                        'Retrain 5',
                        ]

        trials_cs_blocks_phases = [np.arange(1,31), np.arange(31,51), np.arange(51,81)]

        trials_us_blocks_phases = [np.arange(15,34), np.arange(34,62)]

        names_cs_blocks_phases = ['Test 1', 'Retrain 1', 'Retrain 2']

        names_us_blocks_phases = ['Retrain 1', 'Retrain 2']

    case 'delayLongTermNew-3':

        #* Path where the raw data is.
        path_home = Path(r'D:\2024 07_Long-term memory control, delay spaced training with Puromycin (25 mg per L)\2+3\Block 3')
        path_save = Path(r'E:\Results (paper)\2024 07_Long-term memory control, delay spaced training with Puromycin (25 mg per L)\2+3\Block 3')

        cond0 = 'controlspacednopuromycin'
        cond1 = 'delayspacednopuromycin'
        cond2 = 'controlspacedWithPuromycin'
        cond3 = 'delayspacedWithPuromycin'

        us_latency_delay = -1  # s

        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : 'lightblue', 'name' : 'Control Spaced No Puromycin', 'name in original path' : 'controlspacednopuromycin'},
                  cond1 : {'color' :  'pink', 'name' : 'Control Spaced Puromycin', 'name in original path' : 'controlspacedpuromycin'},
                  cond2 : {'color' : 'blue', 'name' : 'Delay Spaced No Puromycin', 'name in original path' : 'delayspacednopuromycin'},
                  cond3 : {'color' : 'red', 'name' : 'Delay Spaced Puromycin', 'name in original path' : 'delayspacedpuromycin'}
                  }

        cr_window = 9  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 50
        min_number_us_trials = 35

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 50
        min_number_us_trials = 35
        # time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        # catch_trials_training = [25, 39, 53, 59]

        # catch_trials_plot = list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))

        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(1,11)],
                                [*range(11,21)],
                                [*range(21,31)],
                                [*range(31,41)],
                                [*range(41,51)],
                                ]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [
                                [*range(16,26)],
                                [*range(26,35)],
                                ]

        names_cs_blocks_10 = [
                        'Test 4',
                        'Test 5',
                        'Test 6',
                        'Retrain 6',
                        'Retrain 7',
                        ]

        names_us_blocks_10 = [
                        'Retrain 6',
                        'Retrain 7',
                        ]



        trials_cs_blocks_phases = [np.arange(1,31), np.arange(31,51)]

        trials_us_blocks_phases = [np.arange(16,34)]

        names_cs_blocks_phases = ['Test 2', 'Retrain 3']

        names_us_blocks_phases = ['Retrain 3']

    case 'delayLongTermNew-all':

    #! August 2023 delay with massed or spaced schedule

        #* Path where the raw data is.
        path_home = Path(r'D:\2024 07_Long-term memory control, delay new\All\All blocks')
        path_save = Path(r'F:\Results (paper)\2024 07_Long-term memory control, delay new\All\All blocks')


        cond0 = 'control'
        cond1 = 'delay'

        cs_duration = 10  # s

        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : (0, 174, 239), 'name' : 'Control', 'name in original path' : 'control'},
                  cond1 : {'color' :  (236, 0, 140), 'name' : 'Delay', 'name in original path' : 'delay'},
                  }
        cr_window = 9  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 64 + 70 + 50
        min_number_us_trials = 59 + 52 + 35

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 64 + 70 + 50
        min_number_us_trials = 59 + 52 + 35
        # time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        # catch_trials_training = [25, 39, 53, 59] + [25 + 64, 39 + 64, 53 + 64, 59 + 64]

        # catch_trials_plot = list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))

        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],

                                [*range(65,75)],
                                [*range(75,85)],
                                [*range(85,95)],
                                [*range(95,105)],
                                [*range(105,115)],
                                [*range(115,125)],
                                [*range(125,135)],

                                # [*range(135,145)],
                                # [*range(145,155)],
                                # [*range(155,165)],
                                # [*range(165,175)],
                                # [*range(175,185)]
                                ]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                [*range(55,64)],

                                [*range(82,92)],
                                [*range(92,101)],
                                [*range(101,110)],
                                [*range(110,119)],

                                # [*range(119,129)],
                                # [*range(129,138)],
                                # [*range(138,147)],
                                # [*range(147,156)],
                                ]

        names_cs_blocks_10 = [

            'Pre-train',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',

            'Test 1',
            'Test 2',
            'Test 3',
            'Retrain 1',
            'Retrain 2',
            'Retrain 3',
            'Retrain 4',
            # 'Retrain 5',

            # 'Test 4',
            # 'Test 5',
            # 'Test 6',
            # 'Retrain 6',
            # 'Retrain 7',
        ]

        names_us_blocks_10 = [

            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',

            'Retrain 1',
            'Retrain 2',
            'Retrain 3',
            'Retrain 4',
            # 'Retrain 5',

            # 'Retrain 6',
            # 'Retrain 7',
        ]



        trials_cs_blocks_phases = [

            np.arange(5,15), np.arange(15,65),

            np.arange(65, 95), np.arange(95, 115), np.arange(115, 135),

            # np.arange(135, 165), np.arange(165, 185)
        ]

        trials_us_blocks_phases = [

            np.arange(18,64),

            np.arange(82, 101), np.arange(101, 119),

            np.arange(119, 156)
        ]



        names_cs_blocks_phases = [
            'Pre',
            'Train',
            'Test',
            'Retrain',
            # 'Retrain 2',
            # 'Test 2',
            # 'Retrain 3'
            ]

        names_us_blocks_phases = [
            'Train',
            'Retrain 1',
            'Retrain 2',
            'Retrain 3'
            ]


    case '2-P multiple planes top':

        #* Path where the raw data is.

        path_home = Path(r'D:\2024 03_Delay 2-P 15 planes top part\Tail')
        path_save = Path(r'F:\Results (paper)\2024 03_Delay 2-P 15 planes top part\Tail')

        cond0 = 'control'
        cond1 = 'delay'
        cond2 = 'trace'

        cs_duration	= 10 # s

        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : (0,174,239), 'name' : 'Control', 'name in original path' : 'control'},
               cond1 : {'color' :  (236,0,140), 'name' : 'Delay', 'name in original path' : 'delay', 'US latency' : [9]*46},
               cond1 : {'color' :  (255,135,94), 'name' : 'Trace', 'name in original path' : 'trace', 'US latency' : [9]*46},}

        cr_window = 9  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 70
        min_number_us_trials = 71

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 70
        min_number_us_trials = 71
        time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        # catch_trials_training = [25, 39, 53, 59]

        # catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]
        # list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))



        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                 [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],
                                [*range(65,75)]]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [[*range(22,32)],
                                [*range(32,42)],
                                [*range(42,52)],
                                [*range(52,62)]]

        names_cs_blocks_10 = [
            'Pre-train 1',
            'Pre-train 2',
            'Pre-train 3',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            # 'Train 5',
            # 'Test 1',
            # 'Test 2',
            # 'Test 3'
            ]

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            # 'Train 5'
        ]



        trials_cs_blocks_phases = [[*range(5,35)], [*range(35,75)]]

        trials_us_blocks_phases = [[*range(22,62)]]

        names_cs_blocks_phases = [
            'Pre',
            'Train',
            # 'Test'
        ]

        names_us_blocks_phases = [
            'Train'
        ]

    case '2-P multiple planes bottom':

        #* Path where the raw data is.

        path_home = Path(r'D:\2024 10_Delay 2-P 15 planes bottom part\Tail')
        path_save = Path(r'F:\Results (paper)\2024 10_Delay 2-P 15 planes bottom part\Tail')

        cond0 = 'control'
        cond1 = 'delay'
        cond2 = 'trace'

        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : (0,174,239), 'name' : 'Control', 'name in original path' : 'control'},
               cond1 : {'color' :  (236,0,140), 'name' : 'Delay', 'name in original path' : 'delay', 'US latency' : [9]*46},
               cond2 : {'color' :  (255,135,94), 'name' : 'Trace', 'name in original path' : 'trace', 'US latency' : [9]*46},}

        cr_window = 9  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 70
        min_number_us_trials = 71

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 70
        min_number_us_trials = 71
        time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        # catch_trials_training = [25, 39, 53, 59]

        # catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]
        # list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))



        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                 [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],
                                [*range(65,75)]]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [[*range(22,32)],
                                [*range(32,42)],
                                [*range(42,52)],
                                [*range(52,62)]]

        names_cs_blocks_10 = [
            'Pre-train 1',
            'Pre-train 2',
            'Pre-train 3',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            # 'Train 5',
            # 'Test 1',
            # 'Test 2',
            # 'Test 3'
            ]

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            # 'Train 5'
        ]



        trials_cs_blocks_phases = [[*range(5,35)], [*range(35,75)]]

        trials_us_blocks_phases = [[*range(22,62)]]

        names_cs_blocks_phases = [
            'Pre',
            'Train',
            # 'Test'
        ]

        names_us_blocks_phases = [
            'Train'
        ]

    case '2-P multiple planes zoom in':

        #* Path where the raw data is.


        path_home = Path(r'D:\2024 09_Delay 2-P 4 planes JC neurons\Tail')
        path_save = Path(r'F:\Results (paper)\2024 09_Delay 2-P 4 planes JC neurons\Tail')

        cond0 = 'control'
        cond1 = 'delay'
        cond2 = 'trace'

        cs_duration	= 10 # s

        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : (0,174,239), 'name' : 'Control', 'name in original path' : 'control'},
               cond1 : {'color' :  (236,0,140), 'name' : 'Delay', 'name in original path' : 'delay', 'US latency' : [9]*46},
               cond2 : {'color' :  (255,135,94), 'name' : 'Trace', 'name in original path' : 'trace', 'US latency' : [9]*46},}

        cr_window = 9  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 84
        min_number_us_trials = 69

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 84
        min_number_us_trials = 69
        time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        catch_trials_training = [12, 26, 34]

        # catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]
        # list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))



        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                 [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],
                                [*range(65,75)],
                                [*range(75,85)]
                                ]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                # [*range(55,64)]
                                ]

        names_cs_blocks_10 = [
            'Pre-train 1',
            # 'Pre-train 2',
            # 'Pre-train 3',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            # 'Train 5',
            'Test 1',
            'Test 2',
            'Test 3'
            ]

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            # 'Train 5'
        ]



        trials_cs_blocks_phases = [[*range(5,15)], [*range(15,55)], [*range(55,85)]]

        trials_us_blocks_phases = [[*range(18,55)]]

        names_cs_blocks_phases = [
            'Pre',
            'Train',
            'Test'
        ]

        names_us_blocks_phases = [
            'Train'
        ]

    case '2-P multiple planes ca8':

        #* Path where the raw data is.


        path_home = Path(r'D:\2024 03_Delay 2-P 15 planes top part\Tail')
        path_save = Path(r'F:\Results (paper)\2024 03_Delay 2-P 15 planes top part\Tail')

        cond0 = 'control'
        cond1 = 'delay'
        cond2 = 'trace'

        cs_duration	= 10 # s

        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : (0,174,239), 'name' : 'Control', 'name in original path' : 'control'},
               cond1 : {'color' :  (236,0,140), 'name' : 'Delay', 'name in original path' : 'delay', 'US latency' : [9]*46},
               cond1 : {'color' :  (255,135,94), 'name' : 'Trace', 'name in original path' : 'trace', 'US latency' : [9]*46},}

        cr_window = 9  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 70
        min_number_us_trials = 71

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 70
        min_number_us_trials = 71
        time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        # catch_trials_training = [25, 39, 53, 59]

        # catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]
        # list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))



        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                 [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],
                                [*range(65,75)]]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [[*range(22,32)],
                                [*range(32,42)],
                                [*range(42,52)],
                                [*range(52,62)]]

        names_cs_blocks_10 = [
            'Pre-train 1',
            'Pre-train 2',
            'Pre-train 3',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            # 'Train 5',
            # 'Test 1',
            # 'Test 2',
            # 'Test 3'
            ]

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            # 'Train 5'
        ]



        trials_cs_blocks_phases = [[*range(5,35)], [*range(35,75)]]

        trials_us_blocks_phases = [[*range(22,62)]]

        names_cs_blocks_phases = [
            'Pre',
            'Train',
            # 'Test'
        ]

        names_us_blocks_phases = [
            'Train'
        ]

    case '2-P single plane':

        #* Path where the raw data is.

        path_home = Path(r'D:\2024 10_Delay 2-P single plane\Tail')
        path_save = Path(r'F:\Results (paper)\2024 10_Delay 2-P single plane\Tail')


        cond0 = 'control'
        cond1 = 'delay'
        cond2 = 'trace'

        #TODO add more stuff to this dict.
        cond_dict = {cond0 : {'color' : (0,174,239), 'name' : 'Control', 'name in original path' : 'control'},
               cond1 : {'color' :  (236,0,140), 'name' : 'Delay', 'name in original path' : 'delay', 'US latency' : [9]*46},
               cond2 : {'color' :  (255,135,94), 'name' : 'Trace', 'name in original path' : 'trace', 'US latency' : [9]*46},}

        cs_duration = 10  # s

        cr_window = 9  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 84
        min_number_us_trials = 69

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 84
        min_number_us_trials = 69
        time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        catch_trials_training = [12, 26, 34]

        # catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]
        # list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))



        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                 [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],
                                [*range(65,75)],
                                [*range(75,85)]
                                ]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                # [*range(55,64)]
                                ]

        names_cs_blocks_10 = [
            'Pre-train 1',
            # 'Pre-train 2',
            # 'Pre-train 3',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            # 'Train 5',
            'Test 1',
            'Test 2',
            'Test 3'
            ]

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            # 'Train 5'
        ]



        trials_cs_blocks_phases = [[*range(5,15)], [*range(15,55)], [*range(55,85)]]

        trials_us_blocks_phases = [[*range(18,55)]]

        names_cs_blocks_phases = [
            'Pre',
            'Train',
            'Test'
        ]

        names_us_blocks_phases = [
            'Train'
        ]


    case 'ca8ablation':

    #! 09 2025'

        #* Path where the raw data is.
        # path_home = Path(r'I:\Joaquim\data')
        path_home = Path(r'D:\2025 09_delay and 3sTrace ca8 ablation')
        path_save = Path(r'F:\Results (paper)\2025 09_delay and 3sTrace ca8 ablation')

        cond0 = 'controlca8negnfp'
        cond1 = 'controlca8posnfp'

        cond2 = 'delayca8negnfp'
        cond3 = 'delayca8posnfp'

        cond4 = 'traceca8negnfp'
        cond5 = 'traceca8posnfp'

        min_delay_interval_stable_numb_trials = 10
        max_delay_interval_stable_numb_trials = 10

        # us_latency_delay = 3  # s

        #! Careful with the number of catch trials!
        number_us_trials_increasing_delay = 26

        cs_duration = 10 # 10 s

        #TODO add more stuff to this dict.
        cond_dict = {
                    cond0 : {'color' : (0,174,239), 'name' : 'Ctrl Ca8-', 'name in original path' : 'controlCa8negNfp'},
                    cond1 : {'color' : (236,0,140), 'name' : 'Ctrl Ca8+', 'name in original path' : 'controlCa8posNfp', 'US latency' : [9]*46},
                    cond2 : {'color' : (0, 139, 191), 'name' : 'Delay Ca8-', 'name in original path' : 'delayCa8negNfp'},
                    cond3 : {'color' : (189, 0, 112), 'name' : 'Delay Ca8+', 'name in original path' : 'delayCa8posNfp', 'US latency' : [9]*46},
                    cond4 : {'color' : (0, 104, 143), 'name' : 'Trace Ca8-', 'name in original path' : 'traceCa8negNfp'},
                    cond5 : {'color' : (142, 0, 84), 'name' : 'Trace Ca8+', 'name in original path' : 'traceCa8posNfp', 'US latency' : [9]*46}}

        relevant_trial_numbers = np.array([10, 20, 60, 70, 90])
        where_v_lines_1 = [10.5, 60.5]
        relevant_session_numbers = np.array([0, 3, 7])
        where_v_lines_2 = np.array([0.5, 5.5])

#! Exception
        cr_window = 9  # s

        # Minimum number of trials to not discard an 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78

        # To estimate maximum duration of the 'Exp.'.
        min_number_cs_trials = 94
        min_number_us_trials = 78
        time_aft_last_trial = 1 # min


#TODO add this to the other experiments
        catch_trials_training = [25, 39, 53, 59]

        catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]
        # list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))



        # To plot pooled trials.

        # 9*10 trials
        trials_cs_blocks_10 = [[*range(5,15)],
                                [*range(15,25)],
                                [*range(25,35)],
                                [*range(35,45)],
                                [*range(45,55)],
                                [*range(55,65)],
                                [*range(65,75)],
                                [*range(75,85)],
                                [*range(85,95)],
                                ]

        # Starts at 18 because the first US is discarded in the analysis.
        trials_us_blocks_10 = [[*range(18,28)],
                                [*range(28,37)],
                                [*range(37,46)],
                                [*range(46,55)],
                                [*range(55,64)]]

        names_cs_blocks_10 = [
            'Pre-train',
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5',
            'Test 1',
            'Test 2',
            'Test 3']

        names_us_blocks_10 = [
            'Train 1',
            'Train 2',
            'Train 3',
            'Train 4',
            'Train 5']



        trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65,95)]

        trials_us_blocks_phases = [np.arange(15,64)]

        names_cs_blocks_phases = [
            'Pre',
            'Train',
            'Test']

        names_us_blocks_phases = [
            'Train']



path_lost_frames, path_summary_exp, path_summary_beh, path_processed_data, path_cropped_exp_with_bout_detection, path_tail_angle_fig_cs, path_tail_angle_fig_us, path_raw_vigor_fig_cs, path_raw_vigor_fig_us, path_scaled_vigor_fig_cs, path_scaled_vigor_fig_us, path_normalized_fig_cs, path_normalized_fig_us, path_pooled_vigor_fig, path_analysis_protocols, path_orig_pkl, path_all_fish, path_pooled = f.create_folders(path_save)




cond_types = [*cond_dict.keys()]
exp_types_names = [cond_dict[k]['name in original path'] for k in [*cond_dict.keys()]]
exp_types_order = [cond.lower() for cond in cond_dict.keys()]



color_palette = [tuple([y/256 for y in cond_dict[x]['color']]) for x in cond_dict.keys()]




#!!!!!!!!!!!!!!!! uncomment
# for c in cond_dict.keys():
#     cond_dict[c]['color'][:3] = [i / 255 for i in cond_dict[c]['color'][:3]]



# To plot single trials.
blocks_cs_single_trials = [[x] for x in np.arange(1, min_number_cs_trials+1)]

blocks_us_single_trials = [[x] for x in np.arange(1, min_number_us_trials+1)]

number_rows_single_trials = max(len(blocks_cs_single_trials), len(blocks_us_single_trials))


number_rows_blocks_10 = max(len(trials_cs_blocks_10), len(trials_us_blocks_10))

number_rows_blocks_phases = max(len(trials_cs_blocks_phases), len(trials_us_blocks_phases))




blocks_dict = {

    'single trials' : {
        'CS' : {
            'trials in each block' : blocks_cs_single_trials,
            'names of blocks' : [str(x[0]) for x in blocks_cs_single_trials]
            },
        'US' : {
            'trials in each block' : blocks_us_single_trials,
            'names of blocks' : [str(x[0]) for x in blocks_us_single_trials]
            },
        'number of cols or rows' : min_number_cs_trials,
        # horizontal_fig : False,
        'figure size' : (15,2*min_number_cs_trials/3)
        },

    'blocks 10 trials' : {
        'CS' : {
            'trials in each block' : trials_cs_blocks_10,
            'names of blocks' : names_cs_blocks_10
            },
        'US' : {
            'trials in each block' : trials_us_blocks_10,
            'names of blocks' : names_us_blocks_10
            },
        'number of cols or rows' : number_rows_blocks_10,
        # horizontal_fig : True,
        'figure size' : (60*number_rows_blocks_10/4,15)
        },

    'blocks phases' : {
        'CS' : {
            'trials in each block' : trials_cs_blocks_phases,
            'names of blocks' : names_cs_blocks_phases
            },
        'US' : {
            'trials in each block' : trials_us_blocks_phases,
            'names of blocks' : names_us_blocks_phases
            },
        'number of cols or rows' : number_rows_blocks_phases,
        }
    }