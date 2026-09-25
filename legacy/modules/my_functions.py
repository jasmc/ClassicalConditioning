# %%
import os
import sys
from pathlib import Path
from timeit import default_timer as timer

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly as py
import seaborn as sns
from cycler import cycler
# from numba import jit, prange
from pandas.api.types import CategoricalDtype
from plotly import graph_objs as go
from plotly.subplots import make_subplots
from scipy import interpolate

# Add the directory containing 'my_functions' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import my_general_variables as gen_var


def set_plot_style():
    """
    Configures matplotlib to produce high-quality scientific plots.
    This function sets default parameters for font size, line width, colors, etc.
    """
    sns.set_context("paper")
    sns.set_style("ticks")

    # Enable LaTeX rendering for text in the figure
    plt.rcParams['text.usetex'] = False
    # Set SVG font type to 'none' to export text as text
    plt.rcParams['svg.fonttype'] = 'none'

    # General font settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['axes.titleweight'] = 'normal'
    plt.rcParams['axes.labelweight'] = 'bold'

    # Line settings
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams['lines.linestyle'] = '-'
    plt.rcParams['lines.color'] = 'C0'
    plt.rcParams['lines.marker'] = 'None'
    plt.rcParams['lines.markerfacecolor'] = 'black'
    plt.rcParams['lines.markeredgecolor'] = 'black'
    plt.rcParams['lines.markeredgewidth'] = 1.0
    plt.rcParams['lines.markersize'] = 5
    plt.rcParams['lines.dash_joinstyle'] = 'round'
    plt.rcParams['lines.dash_capstyle'] = 'butt'
    plt.rcParams['lines.solid_joinstyle'] = 'round'
    plt.rcParams['lines.solid_capstyle'] = 'projecting'
    plt.rcParams['lines.antialiased'] = True

    # Patch settings
    plt.rcParams['patch.linewidth'] = 1.0
    plt.rcParams['patch.facecolor'] = 'none'
    plt.rcParams['patch.edgecolor'] = 'black'
    plt.rcParams['patch.force_edgecolor'] = False
    plt.rcParams['patch.antialiased'] = True

    # Hatch settings
    plt.rcParams['hatch.color'] = 'black'
    plt.rcParams['hatch.linewidth'] = 1.0

    # Boxplot settings
    plt.rcParams['boxplot.notch'] = False
    plt.rcParams['boxplot.vertical'] = True
    plt.rcParams['boxplot.whiskers'] = 1.5
    plt.rcParams['boxplot.bootstrap'] = None
    plt.rcParams['boxplot.patchartist'] = False
    plt.rcParams['boxplot.showmeans'] = False
    plt.rcParams['boxplot.showcaps'] = True
    plt.rcParams['boxplot.showbox'] = True
    plt.rcParams['boxplot.showfliers'] = True
    plt.rcParams['boxplot.meanline'] = False

    plt.rcParams['boxplot.flierprops.color'] = 'black'
    plt.rcParams['boxplot.flierprops.marker'] = 'o'
    plt.rcParams['boxplot.flierprops.markerfacecolor'] = 'none'
    plt.rcParams['boxplot.flierprops.markeredgecolor'] = 'none'
    plt.rcParams['boxplot.flierprops.markeredgewidth'] = 1.0
    plt.rcParams['boxplot.flierprops.markersize'] = 6
    plt.rcParams['boxplot.flierprops.linestyle'] = 'none'
    plt.rcParams['boxplot.flierprops.linewidth'] = 1.0

    plt.rcParams['boxplot.boxprops.color'] = 'none'
    plt.rcParams['boxplot.boxprops.linewidth'] = 1.0
    plt.rcParams['boxplot.boxprops.linestyle'] = '-'

    plt.rcParams['boxplot.whiskerprops.color'] = 'black'
    plt.rcParams['boxplot.whiskerprops.linewidth'] = 1.0
    plt.rcParams['boxplot.whiskerprops.linestyle'] = '-'

    plt.rcParams['boxplot.capprops.color'] = 'black'
    plt.rcParams['boxplot.capprops.linewidth'] = 1.0
    plt.rcParams['boxplot.capprops.linestyle'] = '-'

    plt.rcParams['boxplot.medianprops.color'] = 'black'
    plt.rcParams['boxplot.medianprops.linewidth'] = 1.0
    plt.rcParams['boxplot.medianprops.linestyle'] = '-'

    plt.rcParams['boxplot.meanprops.color'] = 'C2'
    plt.rcParams['boxplot.meanprops.marker'] = '^'
    plt.rcParams['boxplot.meanprops.markerfacecolor'] = 'C2'
    plt.rcParams['boxplot.meanprops.markeredgecolor'] = 'C2'
    plt.rcParams['boxplot.meanprops.markersize'] = 6
    plt.rcParams['boxplot.meanprops.linestyle'] = '--'
    plt.rcParams['boxplot.meanprops.linewidth'] = 1.0

    # Axes settings
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.grid.axis'] = 'both'
    plt.rcParams['axes.grid.which'] = 'major'
    plt.rcParams['axes.titlelocation'] = 'left'
    plt.rcParams['axes.titlecolor'] = 'black'
    plt.rcParams['axes.titlepad'] = 6.0
    plt.rcParams['axes.labelpad'] = 4.0
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['axes.axisbelow'] = 'line'
    plt.rcParams['axes.formatter.limits'] = (-4, 4)
    plt.rcParams['axes.formatter.use_locale'] = False
    plt.rcParams['axes.formatter.use_mathtext'] = False
    plt.rcParams['axes.formatter.min_exponent'] = 0
    plt.rcParams['axes.formatter.useoffset'] = True
    plt.rcParams['axes.formatter.offset_threshold'] = 4
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.unicode_minus'] = True
    plt.rcParams['axes.prop_cycle'] = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    plt.rcParams['axes.xmargin'] = 0.05
    plt.rcParams['axes.ymargin'] = 0.05
    plt.rcParams['axes.zmargin'] = 0.05
    plt.rcParams['axes.autolimit_mode'] = 'data'

    # Ticks settings
    plt.rcParams['xtick.top'] = False
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['xtick.labeltop'] = False
    plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.major.size'] = 2
    plt.rcParams['xtick.minor.size'] = 2
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['xtick.minor.width'] = 0.5
    plt.rcParams['xtick.major.pad'] = 2
    plt.rcParams['xtick.minor.pad'] = 2
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['xtick.labelcolor'] = 'inherit'
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['xtick.minor.visible'] = False
    plt.rcParams['xtick.major.top'] = True
    plt.rcParams['xtick.major.bottom'] = True
    plt.rcParams['xtick.minor.top'] = True
    plt.rcParams['xtick.minor.bottom'] = True
    plt.rcParams['xtick.alignment'] = 'center'

    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.right'] = False
    plt.rcParams['ytick.labelleft'] = True
    plt.rcParams['ytick.labelright'] = False
    plt.rcParams['ytick.major.size'] = 2
    plt.rcParams['ytick.minor.size'] = 2
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['ytick.minor.width'] = 0.25
    plt.rcParams['ytick.major.pad'] = 2
    plt.rcParams['ytick.minor.pad'] = 2
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['ytick.labelcolor'] = 'inherit'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['ytick.minor.visible'] = False
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.minor.left'] = True
    plt.rcParams['ytick.minor.right'] = True
    plt.rcParams['ytick.alignment'] = 'center_baseline'

    # Legend settings
    plt.rcParams['legend.loc'] = 'upper right'
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.facecolor'] = 'inherit'
    plt.rcParams['legend.edgecolor'] = 'black'
    plt.rcParams['legend.fancybox'] = False
    plt.rcParams['legend.shadow'] = False
    plt.rcParams['legend.numpoints'] = 1
    plt.rcParams['legend.scatterpoints'] = 1
    plt.rcParams['legend.markerscale'] = 1.0
    plt.rcParams['legend.labelcolor'] = 'black'
    plt.rcParams['legend.title_fontsize'] = 10
    plt.rcParams['legend.borderpad'] = 0.4
    plt.rcParams['legend.labelspacing'] = 0.5
    plt.rcParams['legend.handlelength'] = 2.0
    plt.rcParams['legend.handleheight'] = 0.7
    plt.rcParams['legend.handletextpad'] = 0.8
    plt.rcParams['legend.borderaxespad'] = 0.1
    plt.rcParams['legend.columnspacing'] = 2.0

    # Figure settings
    plt.rcParams['figure.titlesize'] = 11
    plt.rcParams['figure.titleweight'] = 'normal'
    plt.rcParams['figure.labelsize'] = 10
    plt.rcParams['figure.labelweight'] = 'normal'
    plt.rcParams['figure.figsize'] = (5/2.54, 8/2.54)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['figure.edgecolor'] = 'white'
    plt.rcParams['figure.frameon'] = False
    plt.rcParams['figure.subplot.left'] = 0.125
    plt.rcParams['figure.subplot.right'] = 0.9
    plt.rcParams['figure.subplot.bottom'] = 0.11
    plt.rcParams['figure.subplot.top'] = 0.88
    plt.rcParams['figure.subplot.wspace'] = 0.2
    plt.rcParams['figure.subplot.hspace'] = 0.2
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['figure.constrained_layout.h_pad'] = 0.04167
    plt.rcParams['figure.constrained_layout.w_pad'] = 0.04167
    plt.rcParams['figure.constrained_layout.hspace'] = 0.02
    plt.rcParams['figure.constrained_layout.wspace'] = 0.02

    # Savefig settings
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['savefig.edgecolor'] = 'white'
    plt.rcParams['savefig.format'] = 'svg'
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.05
    plt.rcParams['savefig.transparent'] = False
    plt.rcParams['savefig.orientation'] = 'portrait'

    # PS backend settings
    plt.rcParams['ps.papersize'] = 'letter'
    plt.rcParams['ps.useafm'] = False
    plt.rcParams['ps.usedistiller'] = False
    plt.rcParams['ps.distiller.res'] = 6000
    plt.rcParams['ps.fonttype'] = 3

    # PDF backend settings
    plt.rcParams['pdf.compression'] = 6
    plt.rcParams['pdf.fonttype'] = 3
    plt.rcParams['pdf.use14corefonts'] = False
    plt.rcParams['pdf.inheritcolor'] = False

    # SVG backend settings
    # plt.rcParams['svg.image_inline'] = True
    # plt.rcParams['svg.fonttype'] = 'path'
    # plt.rcParams['svg.hashsalt'] = None


set_plot_style()




# @jit(nopython=True, parallel=True)
def calculate_vigor_fast(angles, framerate):
    """Optimized vigor calculation using Numba."""
    n = len(angles)
    vigor = np.empty(n, dtype=np.float32)
    vigor[0] = 0.0
    
    for i in range(1, n):
        vigor[i] = abs(angles[i] - angles[i-1]) * (framerate / 1000)
    
    return vigor


def create_folders(path_home):

    path_lost_frames = path_home / 'Lost frames'
    path_lost_frames.mkdir(parents=True, exist_ok=True)

    path_summary_exp = path_home / 'Summary of protocol actually run'
    path_summary_exp.mkdir(parents=True, exist_ok=True)

    path_summary_beh = path_home / 'Summary of behavior'
    path_summary_beh.mkdir(parents=True, exist_ok=True)


    #* Path to save processed data; create folder to save processed data if it does not exist yet.
    path_processed_data = path_home / 'Processed data'
    path_processed_data.mkdir(parents=True, exist_ok=True)


    path_cropped_exp_with_bout_detection = path_processed_data / '1. summary of exp.'
    path_cropped_exp_with_bout_detection.mkdir(parents=True, exist_ok=True)

    path_tail_angle_fig_cs = path_processed_data / '2. single fish_tail angle' / 'aligned to CS'
    path_tail_angle_fig_cs.mkdir(parents=True, exist_ok=True)

    path_tail_angle_fig_us = path_processed_data / '2. single fish_tail angle' / 'aligned to US'
    path_tail_angle_fig_us.mkdir(parents=True, exist_ok=True)

    path_raw_vigor_fig_cs = path_processed_data / '3. single fish_raw vigor heatmap' / 'aligned to CS'
    path_raw_vigor_fig_cs.mkdir(parents=True, exist_ok=True)
    
    path_raw_vigor_fig_us = path_processed_data / '3. single fish_raw vigor heatmap' / 'aligned to US'
    path_raw_vigor_fig_us.mkdir(parents=True, exist_ok=True)

    path_scaled_vigor_fig_cs = path_processed_data / '4. single fish_scaled vigor heatmap' / 'aligned to CS'
    path_scaled_vigor_fig_cs.mkdir(parents=True, exist_ok=True)
    
    path_scaled_vigor_fig_us = path_processed_data / '4. single fish_scaled vigor heatmap' / 'aligned to US'
    path_scaled_vigor_fig_us.mkdir(parents=True, exist_ok=True)

    path_normalized_fig_cs = path_processed_data / '5. single fish_suppression ratio vigor trial' / 'aligned to CS'
    path_normalized_fig_cs.mkdir(parents=True, exist_ok=True)

    path_normalized_fig_us = path_processed_data / '5. single fish_suppression ratio vigor trial' / 'aligned to US'
    path_normalized_fig_us.mkdir(parents=True, exist_ok=True)
    
    path_pooled_vigor_fig = path_processed_data / 'All fish'
    path_pooled_vigor_fig.mkdir(parents=True, exist_ok=True)

    path_analysis_protocols = path_processed_data / 'Analysis of protocols'
    path_analysis_protocols.mkdir(parents=True, exist_ok=True)

    path_pkl = path_processed_data / 'pkl files'
    path_pkl.mkdir(parents=True, exist_ok=True)


    path_orig_pkl = path_pkl / '1. Original'
    path_orig_pkl.mkdir(parents=True, exist_ok=True)

    path_all_fish = path_pkl / '2. All fish by condition'
    path_all_fish.mkdir(parents=True, exist_ok=True)

    path_pooled = path_pkl / '3. Pooled data'
    path_pooled.mkdir(parents=True, exist_ok=True)

    return path_lost_frames, path_summary_exp, path_summary_beh, path_processed_data, path_cropped_exp_with_bout_detection, path_tail_angle_fig_cs, path_tail_angle_fig_us, path_raw_vigor_fig_cs, path_raw_vigor_fig_us, path_scaled_vigor_fig_cs, path_scaled_vigor_fig_us, path_normalized_fig_cs, path_normalized_fig_us, path_pooled_vigor_fig, path_analysis_protocols, path_orig_pkl, path_all_fish, path_pooled





def msg(stem_fish_path_orig, message):
    
    if type(message) is list:
        message = '\t'.join([str(i) for i in message])

    return [stem_fish_path_orig] + ['\t' + message + '\n']

    # return [stem_fish_path_orig + '\t' + message + '\n']

def save_info(protocol_info_path, stem_fish_path_orig, message):

    message = msg(stem_fish_path_orig, message)
    print(message)

    with open(protocol_info_path, 'a') as file:
        file.writelines(message)

def fish_id(stem_path):
    # Info about a specific 'Fish'.
    
    stem_fish_path = stem_path.lower()
    stem_fish_path = stem_fish_path.split('_')
    day = stem_fish_path[0]

    # strain = stem_fish_path[1]
    # age = stem_fish_path[2].replace('dpf', '')
    # cond_type = stem_fish_path[3]
    # rig = stem_fish_path[4]
    # fish_number = stem_fish_path[5].replace('fish', '')

    fish_number = stem_fish_path[1]
    cond_type = stem_fish_path[2]
    rig = stem_fish_path[3]
    strain = stem_fish_path[4]
    age = stem_fish_path[5].replace('dpf_', '')
    

    return day, strain, age, cond_type, rig, fish_number

def read_initial_abs_time(camera_path):
    # Read the absolute time at the beginning of the 'Exp.'.

    try:
        with open(camera_path, 'r') as f:
            f.readline()
        # Previous version.
            # first_frame_absolute_time = int(float(f.readline().strip('\n').split('\t')[2]))

        # return first_frame_absolute_time

            return int(float(f.readline().strip('\n').split('\t')[2]))

    except:
        print('No absolute time in cam file.')

        return None

def read_camera(camera_path: str) -> pd.DataFrame:

    try:
        start = timer()

        camera = pd.read_csv(str(camera_path), sep=' ', decimal='.', header=0, skiprows=[*range(1, gen_var.number_frames_discard_beg)])

        if len(camera.columns) == 1:
            camera = pd.read_csv(str(camera_path), sep='\t', decimal=',', header=0, skiprows=[*range(1, gen_var.number_frames_discard_beg)])

        camera.rename(columns={'TotalTime' : 'ElapsedTime'}, inplace=True)
        camera.rename(columns={'ID' : 'FrameID'}, inplace=True)
            
        camera = camera.astype({'FrameID' : 'int', 'ElapsedTime' : 'float', 'AbsoluteTime' : 'int'})
        
        print('Time to read cam.txt: {} (s)'.format(timer()-start))

        return camera

    except:

        print('Cannot read camera file.')
        
        return None


def read_sync_reader(sync_reader_path):

    try:
        start = timer()
        
        sync_reader = pd.read_csv(str(sync_reader_path), sep=' ', header=0, decimal='.')
        
        print('Time to read scape sync reader.txt: {} (s)'.format(timer()-start))

        # sync_reader.rename(columns={'Time' : 'ElapsedTime'}, inplace=True)
        
        return sync_reader

    except:
        # print('Cannot read scape sync file.')

        return None


def framerate_and_reference_frame(camera, stem_fish_path_orig, fig_camera_name):
    # This is complicated. It calculates the true framerate of the camera, estimates the accumulation of lag to capture the frames, checks if any frames were not caught by the computer based on that.
    # It also returns a reference frame at the beginning of the 'Exp.', where the computer was catching the frames at the expected time (interval between frames similar to what would be if recording exactly at 700 frames per second).
    # predicted_framerate should be almost the expected framerate (700 frames per second).
    # predicted_framerate, reference_frame_id, lost_f = f.framerate_and_reference_frame(camera.drop(columns='AbsoluteTime', errors='ignore'), first_frame_absolute_time, protocol_info_path, stem_fish_path_orig, fig_camera_name)
    camera = camera.drop(columns='AbsoluteTime', errors='ignore')
    
    camera.loc[:,['ElapsedTime']] = camera.loc[:,['ElapsedTime']].astype('float')
    
    camera_diff = camera['ElapsedTime'].diff()

    print('Max IFI: {} ms'.format(camera_diff.max()))
    
    # First estimate of the interframe interval, using the median
    ifi = camera_diff.median()
    # camera_diff.iloc[number_frames_discard_beg : ].median()
    print('First estimate of IFI: {} ms'.format(ifi))


    camera_diff_index_correct_IFI = np.where(abs(camera_diff - ifi) <= gen_var.max_interval_between_frames)[0]

    camera_diff_index_correct_IFI_diff = np.diff(camera_diff_index_correct_IFI)

    reference_frame_id = 0
    last_frame_id = 0

    #* Find a region at the beginning where the IFI from frame to frame does not vary significantly and is similar to the first estimate of the true IFI (ifi).
    for i in range(1, len(camera_diff_index_correct_IFI_diff)):

        if camera_diff_index_correct_IFI_diff[i-1] == 1 and camera_diff_index_correct_IFI_diff[i] == 1:

            reference_frame_id = camera['FrameID'].iloc[camera_diff_index_correct_IFI[i] - 1]


            # # first_frame_absolute_time is not None when there is absolute time in the cam file.
            # if first_frame_absolute_time is not None:
            #     reference_frame_time = first_frame_absolute_time + camera[ela_time].iloc[camera_diff_index_correct_IFI[i] - 1] - camera[ela_time].iloc[0]
            # else:
            #     reference_frame_time = None

            break

    #* Find a similar region but at the end of the experiment.
    for i in range(len(camera_diff_index_correct_IFI_diff)-1, 0, -1):

        if camera_diff_index_correct_IFI_diff[i-1] == 1 and camera_diff_index_correct_IFI_diff[i] == 1:
            
            last_frame_id = camera['FrameID'].iloc[camera_diff_index_correct_IFI[i] - 1]
            #last_frame_time = first_frame_absolute_time + camera[time].iloc[camera_diff_index_right_IFI[i] - 1] - camera[time].iloc[0]

            break


    #* Second estimate of the interframe interval, using the mean, and assuming there is no increasing accumulation of frames in the buffer during the experiment; Only the region between the two frames identified in the previous two for loops is considered.
    ifi = camera_diff.iloc[reference_frame_id - camera['FrameID'].iloc[0] : last_frame_id - camera['FrameID'].iloc[0]].mean()

    print('Second estimate of IFI: {} ms'.format(ifi))
    predicted_framerate = 1000 / ifi
    print('Estimated framerate: {} FPS'.format(predicted_framerate))


    def lost_frames(camera, camera_diff, ifi, fish_name, fig_camera_name):


        # Delay to capture frames by the computer
        delay = (camera_diff - ifi).cumsum().to_numpy()


        # Number of lost frames
        # More than one frame might be lost and number_frames_lost sometimes is not monotonically crescent (can go down when some ms are 'recovered').
        number_frames_lost = np.floor(delay / (ifi * gen_var.buffer_size))
        #TODO use this to speed up
        # number_frames_lost = np.max(number_frames_lost, 0, axis)
        number_frames_lost = np.where(number_frames_lost>=0, number_frames_lost, 0)

        number_frames_lost_diff = np.floor(np.diff(number_frames_lost))
        number_frames_lost_diff = np.where(number_frames_lost_diff>=0,number_frames_lost_diff,0)


        # Indices where frames were potentially lost
        where_frames_lost = np.where(number_frames_lost_diff > 0)[0]


        # Total number of missed frames
        if (Lost_frames := len(where_frames_lost) > 0):
            print('Total number of lost frames: ', len(where_frames_lost))
            print('Where: ', where_frames_lost)
            # save_info(protocol_info_path, fish_name, 'Lost frames.')
        else:
            print('No frames were lost')

        fig, axs = plt.subplots(5, 1, sharex=True, facecolor='white', figsize=(20, 40), constrained_layout=True)

        axs[0].plot(camera.iloc[:,1],'black')
        axs[0].set_ylabel('Elapsed time (ms)')
        axs[0].set_title('Estimated IFI: {} ms.    Estimated framerate: {} FPS'.format(round(ifi, 3), round(predicted_framerate, 3)))

        axs[1].plot(camera_diff,'black')
        axs[1].set_ylabel('IFI (ms)')

        axs[2].plot(delay,'black')
        axs[2].set_ylabel('Delay (ms)')

        axs[3].plot(number_frames_lost_diff.cumsum(),'black')
        axs[3].set_ylabel('Cumulative number of lost frames')

        axs[4].plot(number_frames_lost_diff,'black')
        # axs[4].set_xlabel('frame number')
        axs[4].set_ylabel('Lost frames')

        fig.suptitle('Frame number')
        plt.suptitle('Analysis of lost frames\n' + fish_name)

        fig.savefig(fig_camera_name, dpi=100, facecolor='white')
        plt.close(fig)


        # Correct frame IDs in camera dataframe.
        # correctedID = np.zeros(len(camera))

        # for i in tqdm(where_frames_lost):
        #     correctedID[i:number_frames_diff] += 1 # And not correctedID[i:] += 1 because, when the buffer is full, the Mako U29-B camera keeps what is already in the buffer and does not receive any new frames while the buffer is full.

        # del where_frames_lost, number_frames_lost_diff

        # camera['Corrected ID'] = camera['ID'] + correctedID
        # camera['Corrected ID'] = camera['Corrected ID'].astype('int')

        # # Second estimate of the interframe interval, using the median, and after estimating where there are missing frames 
        # camera_diff = camera.loc[:,'ElapsedTime'].diff()
        # ifi = camera_diff.iloc[number_frames_discard_beg : -number_frames_discard_beg].median()
        # print('\nFirst estimate of IFI: {} ms'.format(ifi))

        return Lost_frames

    Lost_frames = lost_frames(camera, camera_diff, ifi, stem_fish_path_orig, fig_camera_name)

    return predicted_framerate, reference_frame_id, Lost_frames

def read_protocol(protocol_path):
# , protocol_info_path, stem_fish_path_orig

    #* Read protocol file.
    if Path(protocol_path).exists():
        # Discarding the last column, which contains the cumulative number of bouts identified in C#.
        try:
            protocol = pd.read_csv(str(protocol_path), sep=' ', header=0, names=['Experiment type', 'beg (ms)', 'end (ms)'], usecols=[0, 1, 2], index_col=0)
        except:
            protocol = pd.read_csv(str(protocol_path), sep='\t', header=0, names=['Experiment type', 'beg (ms)', 'end (ms)'], usecols=[0, 1, 2], index_col=0)

    else:
        # save_info(protocol_info_path, stem_fish_path_orig, 'stim control file does not exist.')
        print('Problems in protocol file.')
        return None

    #* Were the stimuli timings not saved?
    if protocol.empty:
        print('stim control file is empty.')
        # save_info(protocol_info_path, stem_fish_path_orig, 'stim control file is empty.')
        return None


    if protocol.iloc[0,0] == 0:
        
        return None
    
    # #* Is the first stimulus fake? This happened at some point. There was sometimes a line in protocol file in excess.
    # if protocol.loc[:,'beg (ms)'].iloc[0] == 0 and len(protocol.loc[protocol.index.get_level_values('Experiment type') == 'Cycle&Bout']) == expected_number_cs+1:
    #     save_info(protocol_info_path, stem_fish_path_orig, 'Lost beginning of first cycle.')
        
    #     return None
    
    # protocol.rename(index={'Cycle&Bout': 'Cycle'}, inplace=True)
    protocol.sort_values(by='beg (ms)', inplace=True)

    # if reference_frame_time is not None:
        # Getting here means that there is absolute time in the cam file.
        # protocol = protocol - reference_frame_time

    return protocol

def protocol_info(protocol):

    #* Count the number of cycles, trials, blocks and bouts.
    # Using len() just in case these is a single element.
    number_cycles = len(protocol.loc['Cycle', 'beg (ms)'])

    if protocol.index.isin(['Session']).any():
        number_blocks = len(protocol.loc['Session', 'beg (ms)'])
    else:
        number_blocks = number_cycles

    if protocol.index.isin(['Trial']).any():
        trial_numbers = len(protocol.loc['Trial', 'beg (ms)'])
    else:
        trial_numbers = number_cycles

    if protocol.index.isin(['Bout']).any():
        number_bouts = len(protocol.loc['Bout', 'beg (ms)'])      
    else:
        number_bouts = 0
        
    if protocol.index.isin(['Reinforcer']).any():
        number_reinforcers = len(protocol.loc['Reinforcer', 'beg (ms)'])

        us_beg = protocol.loc['Reinforcer', 'beg (ms)']
        us_end = protocol.loc['Reinforcer', 'end (ms)']
        us_dur = (us_end - us_beg).to_numpy() # in ms
        us_isi = (us_beg[1:] - us_end[:-1]).to_numpy() / 1000 / 60 # min
    else:
        number_reinforcers = 0

        us_dur = None
        us_isi = None

    habituation_duration = protocol.iloc[0,0] / 1000 / 60 # min

    cs_beg = protocol.loc['Cycle', 'beg (ms)']
    cs_end = protocol.loc['Cycle', 'end (ms)']
    cs_dur = (cs_end - cs_beg).to_numpy() # in ms
    cs_isi = ('CS beg'[1:] - cs_end[:-1]).to_numpy() / 1000 / 60 # min


    return number_cycles, number_reinforcers, trial_numbers, number_blocks, number_bouts, habituation_duration, cs_dur, cs_isi, us_dur, us_isi

def map_abs_time_to_elapsed_time(camera, protocol):
    
    stimuli = protocol.index.unique()

    camera['AbsoluteTime'] = camera['AbsoluteTime'].astype('float')
    
    camera['ElapsedTime'] = camera['ElapsedTime'].astype('float')
    
    for beg_end in ['beg (ms)', 'end (ms)']:

        protocol_ = protocol.loc[:,beg_end].reset_index().rename(columns={beg_end : 'AbsoluteTime'})

        camera_protocol = pd.merge_ordered(camera, protocol_).set_index('AbsoluteTime').interpolate(kind='slinear').reset_index()

        #* Because here I am relying on "absolute time" (UNIX time, which has ms-resolution), some rows in the original camera dataframe may have the same value of absolute time.
        camera_protocol = camera_protocol.drop_duplicates('AbsoluteTime', keep='first')

        # protocol.loc['Cycle',beg_end] = camera_protocol[camera_protocol['Experiment type']=='Cycle'].set_index('Experiment type').loc[:,'ElapsedTime']
        # protocol.loc['Reinforcer',beg_end] = camera_protocol[camera_protocol['Experiment type']=='Reinforcer'].set_index('Experiment type').loc[:,'ElapsedTime']
        # protocol.loc[:,beg_end] = camera_protocol[camera_protocol['Experiment type'].notna()].set_index('Experiment type').loc[:,'ElapsedTime'].to_numpy()

        for stim in stimuli:

            if len(camera_protocol[camera_protocol['Experiment type']==stim].set_index('Experiment type').loc[:,'ElapsedTime']) == 1:
                
                protocol.loc[stim,beg_end] = camera_protocol[camera_protocol['Experiment type']==stim].set_index('Experiment type').loc[:,'ElapsedTime'].to_numpy()[0]
                
            else:
                
                protocol.loc[stim,beg_end] = camera_protocol[camera_protocol['Experiment type']==stim].set_index('Experiment type').loc[:,'ElapsedTime']

    return protocol[protocol.notna().all(axis=1)]


def lost_stim(number_cycles, number_reinforcers, min_number_cs_trials, min_number_us_trials, protocol_info_path, stem_fish_path_orig, id_debug):

    if number_cycles < min_number_cs_trials:

        save_info(protocol_info_path, stem_fish_path_orig, 'Not all CS! Stopped at CS {} ({}).'.format(number_cycles, id_debug))

        return True

    elif number_reinforcers < min_number_us_trials:
        
        save_info(protocol_info_path, stem_fish_path_orig, 'Not all US! Stopped at US {} ({}).'.format(number_reinforcers, id_debug))

        return True
    else:
        return False

def plot_protocol(cs_dur, cs_isi, us_dur, us_isi, stem_fish_path_orig, fig_protocol_name):

    set_plot_style()

    plt.figure(figsize=(14,14))
    plt.plot(np.arange(1, len(cs_isi) + 1), cs_isi, label='inter-CS interval\nmin int.=' + str(round(np.amin(cs_isi)*60,1)) + ' s\n' + 'CS min dur=' + str(round(np.amin(cs_dur)/1000,3)) + ' s\n' + 'CS max dur=' + str(round(np.amax(cs_dur)/1000,3)) + ' s')
    
    plt.plot(np.arange(5, 4+len(us_isi)+1), us_isi, label='inter-US interval\nmin int.=' + str(round(np.amin(us_isi)*60,1)) + ' s\n' + 'US min dur=' + str(round(np.amin(us_dur)/1000,3)) + 's\n' + 'US max dur='+ str(round(np.amax(us_dur)/1000,3)) + ' s')
    plt.xlabel('Trial number')
    plt.ylabel('ISI (min)')
    plt.ylim(0, 10)
    plt.legend(frameon=False, loc='upper center', ncol=2)
    plt.suptitle('Summary of protocol\n' + stem_fish_path_orig)
    plt.savefig(fig_protocol_name, dpi=100, bbox_inches='tight')
    plt.close()

def number_frames_discard(data_path, reference_frame_id):
    # Consider the 'Exp.' starts only whith the first frame whose ID is both in tail tracking and camera files.

    with open(data_path, 'r') as f:
        f.readline()
        tracking_frames_to_discard = 0
        while reference_frame_id != int(f.readline().split(' ')[0]):
            tracking_frames_to_discard += 1

    return tracking_frames_to_discard

#def readTailTracking(data_path, protocol_frame, tracking_frames_to_discard, time_bcf_window, time_max_window, time_min_window, time_bef_frame, time_aft_frame):
    #    # protocol in number of frames


    #    start = timer()
    #    extra_time_window = np.max([time_bcf_window, time_max_window, time_min_window])

    #    protocol_frame += tracking_frames_to_discard
    #    protocol_frame['beg (ms)'] = protocol_frame['beg (ms)'] + time_bef_frame- 2*extra_time_window
    #    protocol_frame['end (ms)'] = protocol_frame['end (ms)'] + time_aft_frame + 2*extra_time_window

    #    number_frames = protocol_frame['end (ms)'].max()
    #    rows_to_skip = []
    #    number_rows = None
    #    for i in range(len(protocol_frame)):
    #        if i == 0:
    #            # 1 is required to avoid removing the names of the columns
    #            rows_to_skip.extend(np.arange(1, protocol_frame.iat[i,0]))
    #        else:
    #            if (b:= protocol_frame['beg (ms)'].iat[i]) - (a:= protocol_frame['end (ms)'].iloc[:i].max()) > 0:
    #                rows_to_skip.extend(np.arange(a, b))

    #    # frames = np.arange(reference_frame_id - tracking_frames_to_discard, last_frame)
    #    # # frames = pd.read_csv(data, sep=' ', usecols=[0], decimal=',', dtype='int64', engine='c', squeeze=True)
    #    # # frames -= reference_frame_id

    #    # mask_frames = np.zeros(len(frames), dtype=bool)

    #    # for i in range(len(protocol)):
    #    #     mask_frames |= ((frames >= protocol.iat[i,0]) & (frames <= protocol.iat[i,1]))

    #    # rows_to_skip = np.arange(len(mask_frames))[~mask_frames]+1
    #    number_rows = number_frames - len(rows_to_skip)
        
    #    data = pd.read_csv(data_path, sep=' ', header=0, usecols=cols_to_use_orig, nrows=number_rows, skiprows=rows_to_skip, decimal=',')
        
    #    print(timer()-start)
        
    #    return data

def read_tail_tracking_data(data_path):
    """
    Read and preprocess tail tracking data from a file.
    Optimized version with reduced memory allocations and faster operations.
    """
    start = timer()

    try:
        # Use pyarrow engine for faster reading
        data = pd.read_csv(
            data_path, 
            sep=' ', 
            header=0, 
            usecols=gen_var.cols_to_use_orig, 
            decimal=',', 
            engine='pyarrow',
            dtype={gen_var.cols_to_use_orig[0]: 'int32'}  # Pre-specify FrameID dtype
        )
        
        # Discard the last row in one operation
        data = data.iloc[:-1]

    except Exception:
        try:
            data = pd.read_csv(
                data_path, 
                sep=' ', 
                header=0, 
                usecols=gen_var.cols_to_use_orig, 
                decimal=',', 
                engine='c'
            )
            data = data.iloc[:-1]
        except Exception:
            print('Tail tracking might be corrupted!')
            return None

    print(f'Time to read tail tracking .txt: {timer()-start:.3f} (s)')

    # Store original frame number (avoid .copy(deep=True) if possible)
    data['Original frame number'] = data['FrameID'].values

    # Convert to float32 in bulk (already done by pyarrow in many cases)
    # Only convert if needed
    if data.iloc[:, 1:].dtypes[0] != 'float32':
        data.iloc[:, 1:] = data.iloc[:, 1:].astype('float32', copy=False)

    # Vectorized conversion from radian to degree using numpy constant
    radian_cols = gen_var.cols_to_use_orig[1:]
    data.loc[:, radian_cols] = data.loc[:, radian_cols].values * np.float32(180/np.pi)
    
    # Rename columns using dict comprehension
    data.rename(
        columns={old: new for old, new in zip(radian_cols, gen_var.cols)}, 
        inplace=True
    )

    return data





def tracking_errors(data, single_point_tracking_error_thr):

    errors = False

    if ((a := data.loc[:,gen_var.cols].abs().max()) > single_point_tracking_error_thr).any():
        print('Possible tracking error! Max(abs(angle of individual point)):')
        print(a)

        errors = True

    if data.loc[:,gen_var.cols].isna().to_numpy().any():
        print('Possible tracking failures. There are NAs in data!')

        errors = True

    return errors


def merge_camera_with_data(data, camera):

    data = pd.merge_ordered(data, camera, on='FrameID', how='inner')

    data['FrameID'] -= data['FrameID'].iat[0]

    return data


def interpolate_data(data, expected_framerate, predicted_framerate):
    # expected_framerate is the framerate to which data is interpolated. So, output data is as if it had been acquired at the expected_framerate (700 FPS when I wrote this).

    data_ = data.copy(deep=True)

    #* Interpolate tail tracking data to the expected framerate.

    data_['FrameID'] *= expected_framerate/predicted_framerate

    data_.rename(columns={'FrameID' : gen_var.time_trial_frame}, inplace=True)

    interp_function = interpolate.interp1d(data_[gen_var.time_trial_frame], data_.drop(columns=gen_var.time_trial_frame), kind='slinear', axis=0, assume_sorted=True, bounds_error=False, fill_value="extrapolate")

    data = pd.DataFrame(np.arange(data_[gen_var.time_trial_frame].iat[0], data_[gen_var.time_trial_frame].iat[-1]), columns=[gen_var.time_trial_frame])

    data[data_.drop(columns=gen_var.time_trial_frame).columns] = interp_function(data[gen_var.time_trial_frame])


    return data

def rolling_window(a, window):

    #* Alexandre Laborde confirmed this.
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def filter_data(data, space_bcf_window, time_bcf_window):

    #* Select just the part of data to change.
    data_ = data.loc[:, [gen_var.time_trial_frame] + gen_var.cols]

    #* Filter with a rolling average in space
    #* Alexandre Laborde confirmed this.
    # Not using pandas rolling mean beacause over columns it takes a lot of time (confirmed that with this way the result is the same)
    # The fact that here we are using the cumsum means that when averaging more importance is given to the first points
    data_.iloc[:, 2:-1] = np.mean(rolling_window(data_.loc[:,gen_var.cols].to_numpy(), space_bcf_window), axis=2)
    data_.iloc[:, 1] = data_.iloc[:, 1:3].mean(axis=1)
    data_.iloc[:, -1] = data_.iloc[:, -2:].mean(axis=1)

    # This does not work with data_ in float32. Might be a bug in Pandas.
    # Too slow. Use alternative above.
    # data_.loc[:,gen_var.cols] = data_.astype('float').loc[:,gen_var.cols].rolling(window=space_bcf_window, center=True, axis=1).mean()

    #* Filter with a rolling average in time
    data_.loc[:,gen_var.cols] = data_.loc[:,gen_var.cols].rolling(window=time_bcf_window, center=True, axis=0).mean()

    #* Update data with the values changed in data_.
    data.loc[:, [gen_var.time_trial_frame] + gen_var.cols] = data_
    
    data = data.dropna()

    data[gen_var.cols] = data[gen_var.cols].astype('float32')

    return data

def vigor_for_bout_detection(data, chosen_tail_point, time_min_window, time_max_window):
    # Calculate ''Vigor for bout detection (deg/ms)'' (deg/ms)
    #! JUST TRY THIS
    #! data.loc[:, 'Vigor for bout detection (deg/ms)'] = (data.iloc[:,1:2+chosen_tail_point].diff(axis=1).diff().rolling(window=7, center=True, axis=0).mean() * expected_framerate / 1000).pow(2).sum(axis=1)

    # #* Calculate the cumulative sum of the angular velocity over space.
    # #* This allows to take into account movement in any segment with a single scalar value.
    # data.loc[:, 'Vigor for bout detection (deg/ms)'] = data.iloc[:,1:2+chosen_tail_point].diff().abs().sum(axis=1) * (expected_framerate / 1000) # deg/ms
    
    #* Calculate the abstract measure defined by me as ''Vigor for bout detection (deg/ms)''.
    data.loc[:, 'Vigor for bout detection (deg/ms)'] = data.loc[:, 'Vigor for bout detection (deg/ms)'].rolling(window=time_max_window, center=True, axis=0).max() - data.loc[:, 'Vigor for bout detection (deg/ms)'].rolling(window=time_min_window, center=True, axis=0).min()
    
    data.dropna(inplace=True)

    return data




def stim_in_data(data_, protocol):

    data = data_.copy(deep=True)

    data[['CS beg', 'CS end', 'US beg', 'US end']] = 0

    for cs_us in ['CS', 'US']:

        if cs_us=='CS':
            if not protocol.index.isin(['Cycle']).any():
                continue
            else:
                protocol_sub = protocol.loc['Cycle', ['beg (ms)', 'end (ms)']].to_numpy()

        else:
            if not protocol.index.isin(['Reinforcer']).any():
                continue
            else:
                protocol_sub = protocol.loc['Reinforcer', ['beg (ms)', 'end (ms)']].to_numpy()

        for i, p in enumerate(protocol_sub):

            ind_beg = data[data['AbsoluteTime'] > p[0]].index[0]

            # In case the first stimulus happen before the first timepoint in data, still consider it in the number of stimuli in data.
            try:
                ind_end = data[data['AbsoluteTime'] <= p[1]].index[-1]
                data.loc[ind_beg , cs_us + ' beg'] = i + 1
                data.loc[ind_end, cs_us + ' end'] = i + 1
            except:
                continue    

    # data = data.set_index('AbsoluteTime')
    # data.loc[:, data_cols] = data.loc[:, data_cols].interpolate(kind='slinear')
    # data = data.reset_index(drop=True).dropna()
    # data[time_experiment_f] = data[time_experiment_f].astype('int64')

    #* Fix dtypes.
    data['CS beg'] = data['CS beg'].astype(CategoricalDtype(categories=data['CS beg'].unique().sort(), ordered=True))        
    data['US beg'] = data['US beg'].astype(CategoricalDtype(categories=data['US beg'].unique().sort(), ordered=True))
    data['CS end'] = data['CS end'].astype(CategoricalDtype(categories=data['CS end'].unique().sort(), ordered=True))
    data['US end'] = data['US end'].astype(CategoricalDtype(categories=data['US end'].unique().sort(), ordered=True))

    data['AbsoluteTime'] -= data['AbsoluteTime'].iat[0]
    data['AbsoluteTime'] = data['AbsoluteTime'].astype('int32')

    return data



def plot_behavior_overview(data, stem_fish_path_orig, fig_behavior_name):
    # data containing gen_var.tail_angle.

    # mask_frames = np.ones(number_frames + round(60*framerate), dtype=bool)
    # mask_frames[:: round(framerate * 0.5)] = False
    # mask_frames[0] = False
    
    # rows_to_skip = np.arange(number_frames + round(60*framerate))
    # rows_to_skip = rows_to_skip[mask_frames]

    # start = timer()

    # overall_data = pd.read_csv(data, sep=' ', header=0, usecols=cols, skiprows=rows_to_skip, decimal=',')
    # overall_data = overall_data.astype('float32')

    # print(timer() - start)
    plt.figure(figsize=(28, 14))
    plt.plot(data.iloc[:,0]/gen_var.expected_framerate/60/60, data[gen_var.tail_angle], 'black')
    plt.xlabel('Time (h)')
    plt.ylabel('Tail end (ms) angle (deg)')
    plt.suptitle('Behavior overview\n' + stem_fish_path_orig)
    # plt.show()
    # plt.legend(frameon=False, loc='upper center', ncol=2)
    plt.savefig(fig_behavior_name, dpi=100, bbox_inches='tight')
    plt.close()

def extract_data_around_stimuli(data, protocol_frame, time_bef_frame, time_aft_frame, time_bcf_window, time_max_window, time_min_window):

    # protocol_frame is the protocol in number of frames.
    # protocol_frame sorted by 'beg (ms)' and in number of frames.
    # Only save data around the stimuli.


    extra_time_window = np.max([time_bcf_window, time_max_window, time_min_window])

    protocol_frame['beg (ms)'] = protocol_frame['beg (ms)'] + time_bef_frame - 2*extra_time_window
    protocol_frame['end (ms)'] = protocol_frame['end (ms)'] + time_aft_frame + 2*extra_time_window

    # rows_to_skip contains the line numbers with data belonging to frames between trials and not within each 'Trial' time span (-gen_var.time_bef to gen_var.time_aft referenced to stim).
    rows_to_skip = np.arange(protocol_frame.iat[0,0]).tolist()
    # For each stimulus, check if the beginning of the 'Trial' of stimulus 'i' happens after or before of the 'end (ms)' of the previous trials. Remember that protocol contains the stimuli order by their beginning.
    for i in range(1, len(protocol_frame)):
        if (b:= protocol_frame['beg (ms)'].iat[i]) - (a:= protocol_frame['end (ms)'].iloc[:i].max()) > 0:
            rows_to_skip.extend(np.arange(a, b).tolist())

    # errors='ignore' to ignore if rows_to_skip includes line numbers of lines that are already not present in data, instead of showing an error.
    data.drop(index=rows_to_skip, errors='ignore', inplace=True)

    return data

def find_beg_and_end_of_bouts(data, bout_detection_thr_1, min_bout_duration, min_interbout_time, bout_detection_thr_2):

    #* Use the derivative to find the beginning and end (ms) of bouts.
    def bouts_beg_and_end(bouts):
        bouts_beg = np.where(np.diff(bouts) > 0)[0] + 1
        bouts_end = np.where(np.diff(bouts) < 0)[0]
        return bouts_beg, bouts_end


    #* For each timepoint, bouts indicates whether it belongs to a 'Bout' or not.
    # It cannot be initialized to an array of nan because of the derivative calculated below.
    bouts = np.zeros(len(data.loc[:, 'Vigor for bout detection (deg/ms)']))


    # bouts[0] and bouts[-1] = 0 to account for cases when the period under analysis starts in the middle of a 'Bout' or finishes in the middle of a 'Bout'.
    bouts[1:-1][data['Vigor for bout detection (deg/ms)'].iloc[1:-1] >= bout_detection_thr_1] = 1


    bouts_beg, bouts_end = bouts_beg_and_end(bouts)


    # In principle, the line where we used bouts[1:-1] does not allow to enter in the else part.
    if len(bouts_beg) == len(bouts_end):
        bouts_interval = bouts_beg[1:] - bouts_end[:-1]
    else:
        bouts_interval = []
        print('bouts_beg and end (ms) have diff len')


    #* Join bouts close in time after finding the interbout intervals too short.
    for short_interval_bout in reversed(np.where(bouts_interval < min_interbout_time)[0]):
        # if short_interval_bout < len(bouts) - 1:
        bouts[bouts_end[short_interval_bout] + 1 : bouts_beg[short_interval_bout + 1]] = 1


    bouts_beg, bouts_end = bouts_beg_and_end(bouts)


    #* Find bouts too short and remove them.
    for short_bouts in np.where(bouts_end - bouts_beg < min_bout_duration)[0]:

        bouts[bouts_beg[short_bouts] : bouts_end[short_bouts] + 1] = 0


    bouts_beg, bouts_end = bouts_beg_and_end(bouts)


    #* Filter by maximum tail angle of each tail movement.
    # bouts_max = np.zeros_like(bouts_beg)

    for bout_b, bout_e in zip(bouts_beg, bouts_end):
    
        # Angular velocity is converted to deg/ms.
        if data.iloc[bout_b : bout_e + 1, data.columns.get_loc(gen_var.tail_angle)].diff().abs().max() * (gen_var.expected_framerate / 1000) < bout_detection_thr_2:
            bouts[bout_b : bout_e + 1] = 0


    # Previous version
            # for 'Bout' in range(len(bouts_beg)):
            
            #     # Find the maximum of each 'Bout'.        
            #     bouts_max['Bout'] = data.iloc[bouts_beg['Bout'] : bouts_end['Bout'] + 1, data.columns.get_loc(gen_var.tail_angle)].diff().abs().max()

            # too_weak_bouts = np.where(bouts_max < bout_detection_thr_2)[0]
            
            # for weak_bouts in too_weak_bouts:

            #     bouts[bouts_beg[weak_bouts] : bouts_end[weak_bouts] + 1] = 0

    data['Bout'] = bouts
    data['Bout beg'] = data['Bout'].diff() > 0
    data['Bout end'] = data['Bout'].diff() < 0

    data[['Bout', 'Bout beg', 'Bout end']] = data[['Bout', 'Bout beg', 'Bout end']].astype('bool')

    # # Create a column in data with the beginning and end (ms) of bouts.
    # bouts_beg, bouts_end = bouts_beg_and_end(bouts)
    
    # data.iloc[bouts_beg, data.columns.get_loc('Bout beg')] = True
    # data.iloc[bouts_end, data.columns.get_loc('Bout end')] = True
    
    # bouts_beg = data.iloc[bouts_beg,0].to_numpy()
    # bouts_end = data.iloc[:,0].iloc[bouts_end].to_numpy()
    # bouts_beg = data.iloc[:,0].iloc[np.where(data['Bout beg'])[0]].to_numpy()
    # bouts_end = data.iloc[:,0].iloc[np.where(data['Bout end'])[0]].to_numpy()

    return data
    # , bouts_beg, bouts_end



def findStim(data, time_var):

    def correct_stim_array(stim_beg, stim_end):

        if len(stim_beg) > 0 or len(stim_end) > 0:

            if (len(stim_beg) > 0 and len(stim_end) == 0):
                stim_end = np.append(stim_end, data.loc[data.index[-1], time_var])

            if (len(stim_beg) == 0 and len(stim_end) > 0):
                stim_beg = np.append(data.loc[data.index[0], time_var], stim_beg)

            if (stim_end[0] < stim_beg[0]):
                stim_beg = np.append(data.loc[data.index[0], time_var], stim_beg)
                            
            if (stim_end[-1] < stim_beg[-1]):
                stim_end = np.append(stim_end, data.loc[data.index[-1], time_var])

        return stim_beg, stim_end


    # Time needs to be in data's first column.

    cs_beg_array = data.loc[data['CS beg'] != 0,time_var].to_numpy()
    cs_end_array = data.loc[data['CS end'] != 0,time_var].to_numpy()

    us_beg_array = data.loc[data['US beg'] != 0,time_var].to_numpy()
    us_end_array = data.loc[data['US end'] != 0,time_var].to_numpy()



    #* Correct when the beg or end of a block happens while there is a stim going on.
    result = [correct_stim_array(stim_beg, stim_end) for stim_beg, stim_end in [(cs_beg_array, cs_end_array), (us_beg_array, us_end_array)]]

    return result[0][0], result[0][1], result[1][0], result[1][1]


def findEvents(data, event_beg, event_end, time_var):

    data = data.reset_index(drop=True)

    def correctEventArray(e_beg, e_end):

        if len(e_beg) > 0 or len(e_end) > 0:

            if (len(e_beg) > 0 and len(e_end) == 0):
                e_end = np.append(e_end, data.loc[data.index[-1], time_var])

            if (len(e_beg) == 0 and len(e_end) > 0):
                e_beg = np.append(data.loc[data.index[0], time_var], e_beg)

            if (e_end[0] < e_beg[0]):
                e_beg = np.append(data.loc[data.index[0], time_var], e_beg)
                            
            if (e_end[-1] < e_beg[-1]):
                e_end = np.append(e_end, data.loc[data.index[-1], time_var])

        return e_beg, e_end

    e_beg_array = data.loc[data[event_beg] > 0, time_var].to_numpy()
    e_end_array = data.loc[data[event_end] > 0, time_var].to_numpy()

    return correctEventArray(e_beg_array, e_end_array)




def plot_cropped_experiment(data, expected_framerate, bout_detection_thr_1, bout_detection_thr_2, downsampling_step, stem_fish_path_orig, fig_cropped_exp_with_bout_detection_name):

    set_plot_style()

    data = data.copy(deep=True)

    # Convert time to s.
    # data[gen_var.time_trial_frame] = data[gen_var.time_trial_frame] / expected_framerate


    # # Stimuli 'beg' and end (ms) need to be read from data as there were a few changes to data after applying stim_in_data function.
    # cs_beg_array = data.loc[data['CS beg'] != 0, data.columns[0]].to_numpy()
    # cs_end_array = data.loc[data['CS end'] != 0, data.columns[0]].to_numpy()

    # us_beg_array = data.loc[data['US beg'] != 0, data.columns[0]].to_numpy()
    # us_end_array = data.loc[data['US end'] != 0, data.columns[0]].to_numpy()

    # if len(cs_end_array) < len(cs_beg_array):
    #     cs_end_array.extend([data.iat[-1,0]])

    # if len(us_end_array) < len(us_beg_array):
    #     us_end_array.extend([data.iat[-1,0]])

    # try:
    #     cs_beg_array = data.loc[data['CS beg']>0, 'AbsoluteTime'].to_numpy()
    #     cs_end_array = data.loc[data['CS end']>0, 'AbsoluteTime'].to_numpy()
    # except:
    #     pass
    # try:
    #     us_beg_array = data.loc[data['US beg']>0, 'AbsoluteTime'].to_numpy()
    #     us_end_array = data.loc[data['US end']>0, 'AbsoluteTime'].to_numpy()
    # except:
    #     pass
    cs_beg_array, cs_end_array, us_beg_array, us_end_array = findStim(data, 'AbsoluteTime')

    bouts_beg_array = data.loc[data['Bout beg'], 'AbsoluteTime'].to_numpy()
    bouts_end_array = data.loc[data['Bout end'], 'AbsoluteTime'].to_numpy()

    # print(zip(bouts_beg_array, bouts_end_array))

    trial_transition = np.where(np.diff(data.index) > 1)[0]
    # np.where(data.iloc[:,0].diff() > 1)[0]
    trial_transition = data.iloc[trial_transition-1,0].to_numpy() # / expected_framerate

    # data_plot = deepcopy(data.iloc[::gen_var.downsampling_step, :])
    data = data.iloc[::downsampling_step, :]


    x = data['AbsoluteTime'] #/ expected_framerate

    fig = make_subplots(specs=[[{'secondary_y': True}]])

    fig.add_scatter(x=x, y=data.loc[:, gen_var.tail_angle], name='bcf-time bcf-space cumsum angle [point {}]'.format(gen_var.chosen_tail_point), mode='lines', line_color='black', opacity=0.7, secondary_y=True,)

    fig.add_scatter(x=x, y=data.loc[:, gen_var.tail_angle].diff().abs() * expected_framerate/1000, name='Abs velocity [point {}]'.format(gen_var.chosen_tail_point), mode='lines', line_color='rgb'+str(tuple(gen_var.us_color)), opacity=0.7, visible='legendonly')

    fig.add_scatter(x=x, y=data.loc[:, 'Vigor for bout detection (deg/ms)'], name='Vigor', mode='lines', line_color='blue', opacity=0.7, visible='legendonly', legendgroup='Vigor')

    # if camera_value in data.columns:

    #     fig.add_scatter(x=x, y=data.loc[:, camera_value], name='Camera', mode='lines', line_color='red', opacity=0.7, visible='legendonly')
    
    # if galvo_value in data.columns:

    #     fig.add_scatter(x=x, y=data.loc[:, galvo_value], name='Galvo', mode='lines', line_color='purple', opacity=0.7, visible='legendonly')

    # if photodiode_value in data.columns:

    #     fig.add_scatter(x=x, y=data.loc[:, photodiode_value]*150, name='Photodiode', mode='lines', line_color='brown', opacity=0.7, visible='legendonly')

    # if arduino_value in data.columns:

    #     fig.add_scatter(x=x, y=data.loc[:, arduino_value].diff(), name='Arduino', mode='lines', line_color='darkyellow', opacity=0.7, visible='legendonly')



    # Make shapes for the plots
    shapes = [go.layout.Shape(type='line', xref='x', x0=trial, x1=trial, yref='paper', y0=0, y1=1, opacity=0.7, line_width=1, fillcolor='gray') for trial in trial_transition]


    # if not data[data['Bout beg']].empty:
    # for 'Bout' in range(len(bouts_beg)):
    for bout_b, bout_e in zip(bouts_beg_array, bouts_end_array):

        shapes.append(go.layout.Shape(type='rect', xref='x', x0=bout_b, x1=bout_e, yref='paper', y0=0, y1=1, opacity=0.3, line_width=0, fillcolor='lightgray'))
        # fig.add_vrect(x0=bouts_beg['Bout'], x1=bouts_end['Bout'], opacity=0.3, line_width=0, fillcolor='lightgray')

    shapes.append(go.layout.Shape(type='line', xref='paper', x0=0, x1=1, yref='y', y0=bout_detection_thr_1, y1=bout_detection_thr_1,
    opacity=0.5, line_width=1, fillcolor='gray', line_dash='dash'))

    shapes.append(go.layout.Shape(type='line', xref='paper', x0=0, x1=1, yref='y', y0=bout_detection_thr_2, y1=bout_detection_thr_2,
    opacity=0.5, line_width=1, fillcolor='gray', line_dash='dot'))

    # fig.add_hline(y=bout_detection_thr_1, opacity=0.5, line_width=1, fillcolor='gray', line_dash='dash')
    # fig.add_hline(y=bout_detection_thr_2, opacity=0.5, line_width=1, fillcolor='gray', line_dash='dot')

    for cs_b, cs_e in zip(cs_beg_array, cs_end_array):

        shapes.append(go.layout.Shape(type='rect', xref='x', x0=cs_b, x1=cs_e, yref='paper', y0=0, y1=1, opacity=0.4, line_width=0, fillcolor='rgb' + str(tuple(gen_var.cs_color))))
        # fig.add_vrect(x0=cs_beg_, x1=cs_end_, opacity=0.4, line_width=0, fillcolor=cs_color)

    # for stim in range(len(cs_beg_array)):

        # cs_beg_ = cs_beg_array[stim]
        # # cs_end_array = data.loc[data['CS'] == t, data.columns[0]].iloc[-1]
        # cs_end_ = cs_end_array[stim]

        # shapes.append(go.layout.Shape(type='rect', xref='x', x0=cs_beg_, x1=cs_end_, yref='paper', y0=0, y1=1, opacity=0.4, line_width=0, fillcolor=cs_color))
        # # fig.add_vrect(x0=cs_beg_, x1=cs_end_, opacity=0.4, line_width=0, fillcolor=cs_color)

    for us_b, us_e in zip(us_beg_array, us_end_array):

        shapes.append(go.layout.Shape(type='rect', xref='x', x0=us_b, x1=us_e, yref='paper', y0=0, y1=1, opacity=0.4, line_width=0, fillcolor='rgb' + str(tuple(gen_var.us_color))))
        # fig.add_vrect(x0=us_beg_, x1=us_end_, opacity=0.4, line_width=0, fillcolor=us_color)    

    # for stim in range(len(us_beg_array)):

        # us_beg_ = us_beg_array[stim]
        # # us_end_array = data.loc[data['US'] == t, data.columns[0]].iloc[-1]
        # us_end_ = us_end_array[stim]

        # shapes.append(go.layout.Shape(type='rect', xref='x', x0=us_beg_, x1=us_end_, yref='paper', y0=0, y1=1, opacity=0.4, line_width=0, fillcolor=us_color))
        # # fig.add_vrect(x0=us_beg_, x1=us_end_, opacity=0.4, line_width=0, fillcolor=us_color)

    fig.update_layout(height=1000, width=2000, showlegend=True, plot_bgcolor='rgba(0,0,0,0)', title_text='Behavior before cleaning data, downsampled 5X        ' + stem_fish_path_orig, legend=dict(yanchor='top',y=1,xanchor='left',x=0, bgcolor='white'), shapes=shapes)

    # paper_bgcolor='rgba(0,0,0,0)',

    fig.update_xaxes(title='t (s)', showgrid=False, automargin=True,)

    fig.update_yaxes(title='Velocity or vigor (deg/ms)', showgrid=False, zeroline=False, zerolinecolor='black', automargin=False, range=[0, 20], secondary_y=False,)    

    fig.update_yaxes(title='Angle (deg)', showgrid=False, zeroline=False, zerolinecolor='black', automargin=False, range=[-200, 200], secondary_y=True,)

    py.io.write_html(fig=fig, file=fig_cropped_exp_with_bout_detection_name, auto_open=False)

def clean_data(data):

    # We should not set the angles to 0 deg because of subsequent steps.
    data.loc[~data['Bout'], data.columns[1:2+gen_var.chosen_tail_point]] = np.nan

    # Previous version

            # mask_with_lines_to_keep_bout = np.array([False] * len(data))

            # bouts_beg = data.iloc[:,0].iloc[np.where(data['Bout beg'])[0]].to_numpy()
            # bouts_end = data.iloc[:,0].iloc[np.where(data['Bout end'])[0]].to_numpy()


            # if not data[data['Bout beg']].empty:
            #     for 'Bout' in range(len(bouts_beg)):
                    
            #         mask_with_lines_to_keep_bout += ((data.iloc[:,0] >= bouts_beg['Bout']) & (data.iloc[:,0] <= bouts_end['Bout'])).to_numpy()
    

            # # We should not set the angles to 0 deg...
            # data.loc[~mask_with_lines_to_keep_bout, gen_var.cols] = np.nan

    data.drop(columns='Vigor for bout detection (deg/ms)', inplace=True)

    # data[cols] = data[cols].astype(pd.SparseDtype('float32', np.nan))
    # # Need to do this again as the previous operation seems to change the dtype to int32.
    # data[['CS beg', 'CS end', 'US beg', 'US end', 'Trial type', 'Trial number', 'Block name']] = data[['CS beg', 'CS end', 'US beg', 'US end', 'Trial type', 'Trial number', 'Block name']].astype(pd.SparseDtype('int8', 0))

    return data

# def calculate_tail_vigor(data, gen_var.cols, chosen_tail_point, expected_framerate):
    
    # data[vigor] = data[cols[::1+chosen_tail_point]].diff().abs().sum(axis=1) * (expected_framerate / 1000) # deg/ms
    # # data[vigor] = data[vigor].astype(pd.SparseDtype('float32', 0))
    
    # # Discard the columns with the angle data used to calculate the vigor.
    # # data.drop(cols, axis=1, inplace=True)

    # # data['Bout'] = data[vigor] > 0
    # # data.loc[data[vigor] > 0, 'Bout'] = True

    # return data

def identify_trials(data, time_bef_frame, time_aft_frame):

    trials_list = []

    for cs_us in ['CS', 'US']:

        cs_us_beg = cs_us + ' beg'

        trials_csus = data.loc[data[cs_us_beg] != 0, cs_us_beg].unique()
        # trials_csus = data.loc[data[cs_us_beg] > 0, cs_us_beg].unique()

        for t in trials_csus:

            # trial_beg = trial_reference + time_bef_frame# time_bef_ms / 1000
            # # trial_end in relation to cs_us_beg also because stimuli duration may slightly differ from 'Trial number' to 'Trial number'.
            # trial_end = trial_reference + time_aft_frame # time_aft_ms / 1000 
            trial_reference = data.loc[data[cs_us_beg] == t, data.columns[0]].to_numpy()[0]
            
            trial = data.loc[(data.iloc[:,0] >= trial_reference + time_bef_frame) & (data.iloc[:,0] <= trial_reference + time_aft_frame), :]
            
            # trial[gen_var.time_trial_frame] is not given by np.arange(time_bef_frame, time_aft_frame + 1) because there may be "incomplete' trials at the end (ms) (stopped before trial_reference + time_aft_frame).
            # trial[['Trial type', 'Trial number', gen_var.time_trial_frame]] = cs_us, str(t), np.arange(time_bef_frame, len(trial) + time_bef_frame)    
            trial[['Trial type', 'Trial number']] = cs_us, str(t)
            trial[gen_var.time_trial_frame] = np.arange(time_bef_frame, len(trial) + time_bef_frame)
            # 1000/expected_framerate

            trials_list.append(trial)
        
    data = pd.concat(trials_list)

    # data[vigor] = data[vigor].astype(pd.SparseDtype('float32', 0))
    data['Trial type'] = data['Trial type'].astype('category')
    data['Trial number'] = data['Trial number'].astype('int32')
    # data['Trial number'] = data['Trial number'].astype(CategoricalDtype(categories=data['Trial number'].unique().sort(), ordered=True))
    # data.loc[ : , gen_var.time_trial_frame] = data.loc[ : , gen_var.time_trial_frame].astype('float32')

    # data.drop(data.columns[0], axis=1, inplace=True)

        # To discard automatically 'Fish'.
            #zero_bouts_trials = 0
            
            # 'Trial' = data.loc[data['Trial number'] == t, :]

            # Check that 'Fish' beats the tail before the 'US' at least every few trials.
            # if csus == 'US':
            #     if 'Trial'.loc[('Trial'[gen_var.time_trial_frame] > -numb_seconds_before_us*expected_framerate) & ('Trial'[gen_var.time_trial_frame] < numb_seconds_after_us*expected_framerate) & ('Trial'[vigor] > 0),:].empty:
            #         zero_bouts_trials += 1
            #         if zero_bouts_trials == max_numb_trials_no_bout_bef:
            #             print('!!! Quiet 'Fish' before and after 'US' !!!  'Trial': ', t)
            #             lines.append(stem_fish_path + '\n\t' ' Quiet 'Fish' before and after 'CS' ({} consecutive trials). last 'Trial': {}\n'.format(max_numb_trials_no_bout_bef, t))

            #             skip = True
            #             break
            #     else:
            #         if zero_bouts_trials > 0:
            #             zero_bouts_trials = 0

            # Check that 'Fish' always beats the tail after the 'US'.
            # else:
            #     if 'Trial'.loc[('Trial'[gen_var.time_trial_frame] > 0) & ('Trial'[gen_var.time_trial_frame] < numb_seconds_after_us*expected_framerate) & ('Trial'[vigor] > 0),:].empty:
            #         print('!!! Fish inactive after 'US' !!!  'Trial': ', t)
            #         lines.append(stem_fish_path + '\n\t' ' Fish inactive after 'US'. 'Trial': {}\n'.format(t))

            #         skip = True
            #         break


    return data



def identify_blocks_trials(data_, blocks_dict):

    data = data_[['Trial type', 'Trial number']].copy(deep=True)

    data['Block name'] = ''

    for csus in ['CS','US']:

        blocks_csus = blocks_dict['blocks 10 trials'][csus]['trials in each block']

        for s_i, trials_in_s in enumerate(blocks_csus):

            # if type(trials_in_s) is list:
            data.loc[(data['Trial type']==csus) & (data['Trial number'].astype('int').isin([t for t in trials_in_s])), 'Block name'] = blocks_dict['blocks 10 trials'][csus]['names of blocks'][s_i]
            # s_i + 1

            # In case of single trials and blocks_csus entries being scalars and not lists with a single entry.
            # else:

            #     data.loc[data['Trial number'] == str(trials_in_s), name_block] = s_i + 1

    if blocks_dict['blocks 10 trials']['CS']['trials in each block']:
        data['Block name'] = data['Block name'].astype(CategoricalDtype(categories=blocks_dict['blocks 10 trials']['CS']['names of blocks'], ordered=True))
    else:
        data['Block name'] = data['Block name'].astype(CategoricalDtype(categories=blocks_dict['blocks 10 trials']['US']['names of blocks'], ordered=True))

    data_['Block name'] = data['Block name']

    return data_



def convert_time_from_frame_to_s(data):

    data[gen_var.time_trial_frame] = data[gen_var.time_trial_frame] / gen_var.expected_framerate # s
    
    return data.rename(columns={gen_var.time_trial_frame : 'Trial time (s)'})


def convert_time_from_s_to_frame(data):

    data['Trial time (s)'] = data['Trial time (s)'] * gen_var.expected_framerate # frame
    
    data['Trial time (s)'] = data['Trial time (s)'].astype('int')
    
    return data.rename(columns={'Trial time (s)' : gen_var.time_trial_frame})








def normalized_pooled(data, metric, baseline_window, cr_window, segments_analysis, groups, csus):

    #!!!!!!! Now only implemented for csus=='CS'.


#!!!!!!!!!

    # data[metric].fillna(0, inplace=True)
 

    if csus == 'CS':

        trials_bef_onset = data.loc[data['Trial time (s)'].between(-baseline_window, 0), :].groupby(groups, observed=True)[metric].agg('mean')
        # trials_bef_and_aft_onset = data.loc[data['Trial time (s)'].between(-baseline_window, cr_window), :].groupby(groups, observed=True)[metric].agg('mean')

        trials_aft_onset = data.loc[data['Trial time (s)'].between(0, cr_window), :].groupby(groups, observed=True)[metric].agg('mean')
        
    
    else:
        pass

        trials_bef_onset = data.loc[data['Trial time (s)'].between(-baseline_window-cr_window, -cr_window), :].groupby(groups, observed=True)[metric].agg('mean')

        trials_aft_onset = data.loc[data['Trial time (s)'].between(-cr_window, 0), :].groupby(groups, observed=True)[metric].agg('mean')
         




#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # trials_bef_and_aft_onset.fillna(0, inplace=True)
    # trials_aft_onset.fillna(0, inplace=True)



    data_division_at_onset = pd.concat([trials_bef_onset, trials_aft_onset], axis=1)
    # data_division_at_onset = pd.concat([trials_bef_and_aft_onset, trials_aft_onset], axis=1)
    # data_division_at_onset = trials_aft_onset / trials_bef_and_aft_onset

#!
    # data_division_at_onset.fillna(0, inplace=True)




    # data_division_at_onset[segments_analysis[2]] = data_division_at_onset.iloc[:,1] / (data_division_at_onset.iloc[:,0] + data_division_at_onset.iloc[:,1])
    data_division_at_onset[segments_analysis[2]] = data_division_at_onset.iloc[:,1] / data_division_at_onset.iloc[:,0]
    
    # data_division_at_onset[segments_analysis[2]] = (data_division_at_onset.iloc[:,1] - data_division_at_onset.iloc[:,0]) / data_division_at_onset.iloc[:,0]
    


#!
    # data_division_at_onset[segments_analysis[2]] = data_division_at_onset[segments_analysis[2]].fillna(0.5)
    



    data_division_at_onset.columns = segments_analysis

    # data_division_at_onset = pd.concat([data_division_at_onset, data], axis=1).reset_index('Trial number')

    return data_division_at_onset.reset_index(groups).sort_index()





# def plotVigorHeatmap(data_heatmap, gen_var.downsampling_step, csus, stim_dur, window_data_plot, interval_between_xticks):


#     if csus == 'CS':

#         'color' = cs_color

#         # fig, axs = plt.subplots(1, 1, facecolor='white')

#         # sns.heatmap(data_heatmap, cbar=False, robust=True, xticklabels=int(15*expected_framerate/gen_var.downsampling_step), yticklabels=False, ax=axs, clip_on=False)


#     elif csus == 'US':

#         'color' = us_color

#     fig, axs = plt.subplots(1, 1, facecolor='white')

#     sns.heatmap(data_heatmap, cbar=False, robust=True, xticklabels=int(interval_between_xticks/gen_var.downsampling_step), yticklabels=False, ax=axs, clip_on=False)

#     xlims = axs.get_xlim()
#     middle = np.mean(xlims)
#     factor = (xlims[-1] - xlims[0]) / (2*window_data_plot)

#     axs.axvline(middle, color='color', alpha=0.95, lw=1, linestyle='-')
#     axs.axvline(middle + stim_dur * factor, color='color', alpha=0.95, lw=1, linestyle='-')
    
#     # axs.set_xbound(-40,40)
#     # axs.set_xticks(ticks=axs.get_xticks(), labels=np.arange(-baseline_window, baseline_window+1, interval_between_xticks))

#     axs.set_xlabel('Time relative to {} onset (s)'.format(csus))
#     axs.tick_params(axis='both', which='both', bottom=True, top=False, right=False, direction='out')
#     # axs.set_title(csus, color='color', fontsize=14)

#     return fig, axs



def setDtypesAndSortIndex(data):


    #* Set the columns' dtypes.
    data = data.astype({
        gen_var.time_trial_frame:'int32',
        'CS beg':    CategoricalDtype(categories=np.sort(data['CS beg'].unique()).astype('int64'), ordered=True),
        'CS end':    CategoricalDtype(categories=np.sort(data['CS end'].unique()).astype('int64'), ordered=True),
        'US beg':    CategoricalDtype(categories=np.sort(data['US beg'].unique()).astype('int64'), ordered=True),
        'US end':    CategoricalDtype(categories=np.sort(data['US end'].unique()).astype('int64'), ordered=True),
        'Trial number':    CategoricalDtype(categories=np.sort(data['Trial number'].unique()).astype('int64'), ordered=True),
        # gen_var.tail_angle:'float32',
        'Vigor (deg/ms)':'float32',
        # vigor_digested:'float32',
        'Bout':'bool',
        # 'Bout beg':'bool',
        # 'Bout end':'bool'
        }, copy=False)
    

    ind_list = []

    for ind in data.index.names:
        
        ind_list.append(data.index.get_level_values(ind).astype('category'))

    data.index = ind_list

        # astype(CategoricalDtype(categories=np.sort(data[col_s].unique()), ordered=True))


    data.sort_index(inplace=True)

    return data



def firstPrep(data):

    data['Exp.'] = data['Exp.'].astype(CategoricalDtype(categories=data['Exp.'].unique(), ordered=True))
    data['Fish'] = data['Fish'].astype(CategoricalDtype(categories=data['Fish'].unique(), ordered=True))

    return data


def prepareData(data):

    data['Fish'] = ['_'.join(str(i))  for i in data.index]
    data.reset_index('Exp.', inplace=True)
    # data.reset_index(drop=True, inplace=True)
    # data.loc[:, 'Exp.'] = data.loc[:, 'Exp.']
    
    data = setDtypesAndSortIndex(data)

    return data



def change_block_names (data, blocks, block_names):

    data.drop(columns='Block name',inplace=True)

    data['Block name'] = ''

    for s_i, trials_in_s in enumerate(blocks):

        data.loc[(data['Trial number'].astype('int').isin(trials_in_s)), 'Block name'] = block_names[s_i]

    data['Block name'] = data['Block name'].astype(CategoricalDtype(categories=block_names, ordered=True))

    return data