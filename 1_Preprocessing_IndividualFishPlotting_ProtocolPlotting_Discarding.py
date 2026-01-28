"""
Preprocessing and Individual Fish Analysis Pipeline
===================================================

Consolidated data processing and quality control workflow:
- Preprocessing: Read, synchronize, and process raw tracking data
- Individual Plotting: Generate per-fish diagnostic plots
- Protocol Visualization: Verify stimulus timing
- Quality Control: Apply exclusion criteria and manage discarded fish
"""
# %%
# region Imports
import gc
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Add the repository root containing shared modules to the Python path.
if "__file__" in globals():
    module_root = Path(__file__).resolve()
else:
    module_root = Path.cwd()

import analysis_utils
import data_io
import file_utils
import plotting_style
from experiment_configuration import ExperimentType, get_experiment_config
from general_configuration import config as gen_config
from plotting_style import configure_axes_for_pipeline

plotting_style.set_plot_style(use_constrained_layout=False)
# endregion Imports


# region Parameters
# ------------------------------------------------------------------------------
# Pipeline Control Flags
# ------------------------------------------------------------------------------
RUN_PREPROCESS = False
RUN_PLOT_INDIVIDUALS = False
RUN_PLOT_PROTOCOLS = False
RUN_DISCARD = False

FILTER_FISH_ID = None
EXPERIMENT_TYPE = ExperimentType.FIRST_DELAY.value

# ------------------------------------------------------------------------------
# Preprocess Parameters
# ------------------------------------------------------------------------------
PREPROCESS_OVERWRITE = True

# ------------------------------------------------------------------------------
# Plot Individual Trials Parameters
# ------------------------------------------------------------------------------
PLOT_INDIVIDUALS_OVERWRITE = True
RAW_TAIL_ANGLE = True
RAW_VIGOR = True
SCALED_VIGOR = True
NORMALIZED_VIGOR_TRIAL = True
METRIC_SINGLE_TRIALS = gen_config.tail_angle_label
WINDOW_DATA_PLOT_S = 40
INTERVAL_BETWEEN_XTICKS_S = 20
FIG_FORMAT = 'png'

# ------------------------------------------------------------------------------
# Plot Protocols Parameters
# ------------------------------------------------------------------------------
PLOT_PROTOCOLS_OVERWRITE = True
PLOT_FROM_STIM_CONTROL = True
PLOT_FROM_PROCESSED_DATA = True
PROTOCOLS_FIG_FORMAT = 'png'
# endregion Parameters


# region Pipeline Functions

# %%
def run_preprocess(params: dict = None):
    """
    Executes the raw data preprocessing pipeline.

    This function iterates through all raw fish data files ('*mp tail tracking.txt'),
    performs cleaning, interpolation, and bout detection, and saves the resulting
    dataframe as a pickle file (`.pkl`) in the processed data directory.

    Steps:
    1.  Load camera and tracking data.
    2.  Check for tracking errors.
    3.  Synchronize camera frames with tracking data.
    4.  Interpolate missing data.
    5.  Compute tail angle and vigor.
    6.  Detect behavioral bouts.
    7.  Save processed data for downstream analysis.

    Args:
        params (dict, optional): Dictionary of parameters to override defaults. Defaults to None.
    """
    print("\n" + "="*80)
    print("RUNNING PREPROCESS")
    print("="*80)

    config = get_experiment_config(EXPERIMENT_TYPE)

    def angle_columns(data: pd.DataFrame) -> list:
        """Identifies and sorts columns representing tail point angles."""
        cols = [c for c in data.columns if c.startswith('Angle of point')]

        def key(c):
            try:
                return int(c.split('Angle of point ')[1].split(' ')[0])
            except Exception:
                print(f"Failed to parse angle point number from column name: {c}")
                return c
        return sorted(cols, key=key)

    def plot_behavior_overview(data, fish_name, save_path):
        """Generates a long-timescale plot of tail activity for validity checking."""
        angle_col = gen_config.tail_angle_label
        if angle_col not in data.columns:
            cols = [c for c in data.columns if c.startswith('Angle of point')]
            if cols:
                angle_col = cols[-1]
            else:
                print(f"No angle columns found for behavior overview: {fish_name}")
                return
        if 'AbsoluteTime' in data.columns:
            time_h = data['AbsoluteTime'] / 1000 / 3600
        else:
            time_h = np.arange(len(data)) / gen_config.expected_framerate / 3600
            
        plt.figure(figsize=(28, 14))
        plt.plot(time_h, data[angle_col], 'k', lw=0, markersize=0.1, marker='.')
        plt.xlabel('Time (h)')
        plt.ylabel('Tail end angle (deg)')
        plt.suptitle('Behavior overview\n' + fish_name)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

    if not config.path_home:
        raise ValueError('config.path_home is empty; set a valid experiment path before running')

    (
        path_lost_frames, path_summary_exp, path_summary_beh, path_processed_data,
        path_cropped_exp_with_bout_detection, _, _, _, _, _, _, _, _, _, _,
        path_orig_pkl, _, _
    ) = file_utils.create_folders(config.path_save)

    # Collect raw fish files (optionally filtered by ID).
    all_fish_raw_data_paths = list(Path(config.path_home).glob('*mp tail tracking.txt'))
    if FILTER_FISH_ID:
        all_fish_raw_data_paths = [p for p in all_fish_raw_data_paths if FILTER_FISH_ID in p.name]
    print('Found %d fish files' % len(all_fish_raw_data_paths))

    for fish_path in tqdm(all_fish_raw_data_paths, desc='Preprocessing fish'):
        gc.collect()
        stem_fish_path_orig = fish_path.stem.replace('_mp tail tracking', '').lower()
        pkl_path = path_orig_pkl / f'{stem_fish_path_orig}.pkl'

        if not PREPROCESS_OVERWRITE and pkl_path.exists():
            print('Skipping existing: %s' % stem_fish_path_orig)
            continue

        print('Processing %s' % stem_fish_path_orig)
        
        # Define related file paths
        protocol_path = str(fish_path).replace('mp tail tracking', 'stim control')
        camera_path = str(fish_path).replace('mp tail tracking', 'cam')
        fig_camera_name = str(path_lost_frames / f'{stem_fish_path_orig}_camera.png')
        fig_behavior_name = str(path_summary_beh / f'{stem_fish_path_orig}_behavior.png')

        # Parse Fish ID metadata
        day, strain, age, cond_type, rig, fish_number = file_utils.fish_id(stem_fish_path_orig)

        # 1. Load Camera Data to check for dropped frames
        camera = data_io.read_camera(camera_path)
        if camera is None:
            print(f"Camera data not found or failed to read: {camera_path}")
            continue

        predicted_framerate, reference_frame_id, lost_f = analysis_utils.framerate_and_reference_frame(
            camera, stem_fish_path_orig, fig_camera_name
        )
        if lost_f:
            print(f"Skipping {stem_fish_path_orig} due to high frame loss or framing issues.")
            continue
        camera = camera[camera['FrameID'] >= reference_frame_id]

        # 2. Load Tracking Data
        data = data_io.read_tail_tracking_data(Path(fish_path))
        if data is None:
            print(f"Tail tracking data not found or failed to read: {fish_path}")
            continue
        data = data[data['FrameID'] >= reference_frame_id]

        if analysis_utils.tracking_errors(data, gen_config.validation.single_point_tracking_error_thr):
            print('Tracking errors for %s' % stem_fish_path_orig)
            continue

        # 3. Synchronize and Interpolate
        data = analysis_utils.merge_camera_with_data(data, camera)
        data = analysis_utils.interpolate_data(data, gen_config.expected_framerate, predicted_framerate)

        # 4. Integrate Protocol Data
        protocol = data_io.read_protocol(Path(protocol_path))
        if protocol is None:
            print(f"Protocol data not found or failed to read: {protocol_path}")
            continue

        if data['AbsoluteTime'].max() - protocol['beg (ms)'].max() < 0:
            print('Data not acquired for the entire experiment: %s' % stem_fish_path_orig)
            continue

        data = analysis_utils.stim_in_data(data, protocol)
        time_step = 1000 / predicted_framerate
        data['AbsoluteTime'] = data['AbsoluteTime'].iat[0] + np.arange(len(data), dtype='float64') * time_step

        # 5. Process Tail Angles
        angle_cols = angle_columns(data)
        if not angle_cols:
            print('No angle columns found for %s' % stem_fish_path_orig)
            continue

        # Cumulative sum of angles to handle wrapping/unwrapping if necessary (depends on raw data format)
        data.loc[:, angle_cols] = data.loc[:, angle_cols].cumsum(axis=1)
        plot_behavior_overview(data, stem_fish_path_orig, fig_behavior_name)

        # 6. Filter Data and Compute Vigor
        data = analysis_utils.filter_data(data, gen_config.filtering.space_bcf_window, gen_config.filtering.time_bcf_window)
        tail_angle_col = gen_config.tail_angle_label
        if tail_angle_col not in data.columns:
            if not angle_cols:
                print(f"No angle columns for vigor calculation: {stem_fish_path_orig}")
                continue
            tail_angle_col = angle_cols[-1]
        
        data['Vigor (deg/ms)'] = analysis_utils.calculate_vigor_fast(data[tail_angle_col].values, gen_config.expected_framerate)
        
        # Calculate a rolling window vigor metric specifically for bout detection stability
        data['Vigor for bout detection (deg/ms)'] = (
            data['Vigor (deg/ms)'].rolling(window=gen_config.filtering.time_max_window, center=True).max() -
            data['Vigor (deg/ms)'].rolling(window=gen_config.filtering.time_min_window, center=True).min()
        )
        data.dropna(inplace=True)

        # Normalize time to start at 0
        time_col = gen_config.time_trial_frame_label
        data[time_col] = data[time_col] - data[time_col].iat[0]

        # 7. Bout Detection
        data = analysis_utils.find_beg_and_end_of_bouts(
            data, gen_config.bout_detection.bout_detection_thr_1, gen_config.bout_detection.min_bout_duration,
            gen_config.bout_detection.min_interbout_time, gen_config.bout_detection.bout_detection_thr_2
        )

        # 8. Identify Trials and Blocks
        data = analysis_utils.identify_trials(data, gen_config.time_bef_frame, gen_config.time_aft_frame)
        data.drop(columns='Vigor for bout detection (deg/ms)', inplace=True)
        data = analysis_utils.identify_blocks_trials(data, config.blocks_dict)

        # 9. Compute Scaled Vigor (Normalized to Baseline)
        data['Scaled vigor (AU)'] = data['Vigor (deg/ms)']
        baseline_window_frames = int(round(gen_config.baseline_window * gen_config.expected_framerate))

        for t in data['Trial number'].unique():
            mask_trial = data['Trial number'] == t
            mask_baseline = mask_trial & (data[time_col] < -baseline_window_frames)
            baseline_vigor = data.loc[mask_baseline, 'Vigor (deg/ms)'].dropna().values
            
            if baseline_vigor.size == 0:
                print(f"Skipping scaled vigor for trial {t}: no baseline bouts.")
                data.loc[mask_trial, 'Scaled vigor (AU)'] = np.nan
                continue
            
            # Use 10th and 90th percentiles for robust scaling
            min_vigor_pre_stim, max_vigor_pre_stim = np.quantile(baseline_vigor, [0.1, 0.9])
            
            if np.isnan(max_vigor_pre_stim) or min_vigor_pre_stim == max_vigor_pre_stim:
                print(f"Skipping scaled vigor for trial {t}: baseline has no variance or is NaN.")
                data.loc[mask_trial, 'Scaled vigor (AU)'] = np.nan
                continue
                
            data.loc[mask_trial, 'Scaled vigor (AU)'] = (
                (data.loc[mask_trial, 'Vigor (deg/ms)'] - min_vigor_pre_stim) /
                (max_vigor_pre_stim - min_vigor_pre_stim)
            )

        # 10. Organize Columns and Finalize
        ordered_cols = [
            'Original frame number', time_col, 'CS beg', 'CS end', 'US beg', 'US end',
            'Trial type', 'Trial number', 'Block name', 'Vigor (deg/ms)', 'Scaled vigor (AU)',
            'Bout beg', 'Bout end', 'Bout'
        ] + angle_cols
        ordered_cols = [c for c in ordered_cols if c in data.columns]
        data = data.loc[:, ordered_cols]
        
        # Optimize dtypes to save memory
        for col in angle_cols: 
            if col in data.columns: data[col] = data[col].astype('float32')
        data['Vigor (deg/ms)'] = data['Vigor (deg/ms)'].astype('float32')
        data['Scaled vigor (AU)'] = data['Scaled vigor (AU)'].astype('float32')
        data[time_col] = data[time_col].astype('int32')
        for col in ['Bout beg', 'Bout end', 'Bout']: 
            if col in data.columns: data[col] = data[col].astype('bool')
        for col in ['CS beg', 'CS end', 'US beg', 'US end', 'Trial number']:
            if col in data.columns:
                data[col] = pd.Categorical(data[col], categories=np.sort(pd.unique(data[col])), ordered=True)

        data['Exp.'] = cond_type
        data['ProtocolRig'] = rig
        data['Age (dpf)'] = age
        data['Day'] = day
        data['Fish no.'] = fish_number
        data['Strain'] = strain
        data[['Exp.', 'ProtocolRig', 'Age (dpf)', 'Day', 'Fish no.', 'Strain']] = (
            data[['Exp.', 'ProtocolRig', 'Age (dpf)', 'Day', 'Fish no.', 'Strain']].astype('category')
        )
        data.set_index(keys=['Strain', 'Age (dpf)', 'Exp.', 'ProtocolRig', 'Day', 'Fish no.'], inplace=True)
        
        # Save to pickle
        data.to_pickle(pkl_path, compression='gzip')

    print('PREPROCESS FINISHED')


# %%
def run_plot_individual_trials():
    """
    Generates single-fish diagnostic plots to visualize behavior during trials.

    Produces three types of plots for each fish, aligned to CS and US:
    1.  **Tail Angle Traces**: Raw tail angle over time for every trial.
    2.  **Raw Vigor Heatmaps**: Vigor intensity across trials (using heatmaps).
    3.  **Scaled Vigor Heatmaps**: Vigor normalized to baseline activity.
    4.  **Normalized Vigor Summary**: Average vigor change (Post/Pre) across trials.

    Figures are saved in the configured output directory grouped by fish.
    """
    print("\n" + "="*80)
    print("RUNNING PLOT INDIVIDUAL TRIALS")
    print("="*80)

    config = get_experiment_config(EXPERIMENT_TYPE)

    cr_window = config.cr_window
    if isinstance(cr_window, (int, float, np.integer, np.floating)): cr_window = [0, cr_window]
    
    # Logging structure for skipped files
    process_log = {
        'processed': {'CS': [], 'US': []},
        'skipped_existing': {'CS': [], 'US': []},
        'skipped_read': {'CS': [], 'US': []},
        'skipped_missing_tail': {'CS': [], 'US': []},
        'skipped_missing_time': {'CS': [], 'US': []},
        'skipped_missing_cols': {'CS': [], 'US': []},
        'skipped_empty': {'CS': [], 'US': []},
    }

    interval_between_xticks_frames = int(INTERVAL_BETWEEN_XTICKS_S * gen_config.expected_framerate)
    xtick_step_raw = max(1, int(interval_between_xticks_frames / gen_config.plotting.downsampling_step))
    xtick_step_scaled = max(1, int(interval_between_xticks_frames))

    if not config.path_save: raise ValueError('config.path_save is empty')
    (_, _, _, _, _, path_tail_angle_fig_cs, path_tail_angle_fig_us, path_raw_vigor_fig_cs, path_raw_vigor_fig_us,
     path_scaled_vigor_fig_cs, path_scaled_vigor_fig_us, path_normalized_fig_cs, path_normalized_fig_us,
     _, _, path_orig_pkl, _, _) = file_utils.create_folders(config.path_save)

    for csus in ['CS', 'US']:
        print(f"Processing {csus} trials...")
        try:
            trials_blocks = config.blocks_dict['blocks 10 trials'][csus]['trials in each block']
            if not trials_blocks:
                print(f"No trial blocks defined for {csus} in config.")
                continue
            trial_numbers = trials_blocks[-1][-1] - trials_blocks[0][0] + 1
        except Exception as e:
            print(f"Failed to extract trial block info for {csus}: {e}")
            continue

        if csus == 'CS':
            path_tail_fig, path_raw_fig = path_tail_angle_fig_cs, path_raw_vigor_fig_cs
            path_sc_fig, path_norm_fig = path_scaled_vigor_fig_cs, path_normalized_fig_cs
            stim_duration = config.cs_duration
        else:
            path_tail_fig, path_raw_fig = path_tail_angle_fig_us, path_raw_vigor_fig_us
            path_sc_fig, path_norm_fig = path_scaled_vigor_fig_us, path_normalized_fig_us
            stim_duration = gen_config.us_duration

        all_fish_data_paths = list(Path(path_orig_pkl).glob('*.pkl'))
        if FILTER_FISH_ID:
            all_fish_data_paths = [p for p in all_fish_data_paths if FILTER_FISH_ID in p.name]

        for fish_path in reversed(all_fish_data_paths):
            stem_fish_path_orig = fish_path.stem.lower()
            stem_split = stem_fish_path_orig.split('_')
            fish_id = '_'.join(stem_split[:2]) if len(stem_split) >= 2 else stem_fish_path_orig
            
            # Note: Temporary filter removed. Iterates all fish.
            
            fig_path_tail = Path(str(path_tail_fig / stem_fish_path_orig) + f'_tail angle aligned to {csus}.{FIG_FORMAT}')
            fig_path_raw = Path(str(path_raw_fig / stem_fish_path_orig) + f'_raw vigor heatmap aligned to {csus}.{FIG_FORMAT}')
            fig_path_sc = Path(str(path_sc_fig / stem_fish_path_orig) + f'_scaled vigor heatmap aligned to {csus}.{FIG_FORMAT}')
            fig_path_norm = Path(str(path_norm_fig / stem_fish_path_orig) + f'_normalized vigor trial aligned to {csus}.{FIG_FORMAT}')

            do_tail = RAW_TAIL_ANGLE and (PLOT_INDIVIDUALS_OVERWRITE or not fig_path_tail.exists())
            do_raw = RAW_VIGOR and (PLOT_INDIVIDUALS_OVERWRITE or not fig_path_raw.exists())
            do_sc = SCALED_VIGOR and (PLOT_INDIVIDUALS_OVERWRITE or not fig_path_sc.exists())
            do_norm = NORMALIZED_VIGOR_TRIAL and (PLOT_INDIVIDUALS_OVERWRITE or not fig_path_norm.exists())

            if not any([do_tail, do_raw, do_sc, do_norm]):
                process_log['skipped_existing'][csus].append(fish_id)
                print(f"Skip {csus} {fish_id}: all figures exist")
                continue
            
            # Load per-fish processed data.
            try:
                data = pd.read_pickle(str(fish_path), compression='gzip')
            except Exception:
                process_log['skipped_read'][csus].append(fish_id)
                print(f"Skip {csus} {fish_id}: read failed")
                continue
            data.reset_index(drop=True, inplace=True)

            tail_col = analysis_utils.get_tail_angle_col(data)
            if tail_col is None:
                process_log['skipped_missing_tail'][csus].append(fish_id)
                print(f"Skip {csus} {fish_id}: missing tail angle col")
                continue
            
            # Ensure time is in frame units for plotting.
            time_col = gen_config.time_trial_frame_label
            if time_col not in data.columns and 'Trial time (s)' in data.columns:
                data = analysis_utils.convert_time_from_s_to_frame(data)
            if time_col not in data.columns:
                process_log['skipped_missing_time'][csus].append(fish_id)
                print(f"Skip {csus} {fish_id}: missing time col")
                continue

            metric = METRIC_SINGLE_TRIALS
            if metric == gen_config.tail_angle_label:
                metric = tail_col

            needed_cols = [
                time_col,
                'CS beg',
                'CS end',
                'US beg',
                'US end',
                'Trial type',
                'Trial number',
                'Block name',
                tail_col,
                'Vigor (deg/ms)',
                'Bout beg',
                'Bout end',
                'Bout',
            ]
            if metric not in needed_cols:
                needed_cols.append(metric)

            # Subset to columns required for the plots.
            try:
                data = data.loc[:, needed_cols].copy()
            except Exception:
                process_log['skipped_missing_cols'][csus].append(fish_id)
                print(f"Skip {csus} {fish_id}: missing required columns")
                continue

            data = data.loc[data['Trial type'] == csus, :]
            if data.empty:
                process_log['skipped_empty'][csus].append(fish_id)
                print(f"Skip {csus} {fish_id}: no trials for csus")
                continue

            process_log['processed'][csus].append(fish_id)
            print(f"Analyze {csus} {fish_id}")

            # Convert time to seconds for plotting
            data = analysis_utils.convert_time_from_frame_to_s(data)
            data = data.loc[data['Trial time (s)'].between(-WINDOW_DATA_PLOT_S, WINDOW_DATA_PLOT_S)]

            # Densify sparse columns if needed
            for col in ['Vigor (deg/ms)', 'Bout beg', 'Bout end', 'Bout']:
                if col in data.columns:
                    try:
                        data[col] = data[col].sparse.to_dense()
                    except Exception as e:
                        print(f"Failed to densify {col} for {fish_id}: {e}")
                        pass

            # Plot 1: Tail Angle Traces
            if do_tail:
                data_plot = data[['Trial time (s)', 'Trial number', metric]].copy()
                trials = data_plot['Trial number'].unique().astype('int')

                fig, axs = plt.subplots(len(trials), 1, sharex=False, sharey=False, facecolor='white')
                if len(trials) == 1:
                    axs = [axs]

                for t_i, t in enumerate(trials):
                    data_trial = data[data['Trial number'] == t]

                    # Extract CS/US event times
                    try:
                        data_trial_events = data_trial[['Trial time (s)', 'CS beg', 'CS end']].copy()
                        cs_b, cs_e = analysis_utils.find_events(data_trial_events, 'CS beg', 'CS end', 'Trial time (s)')
                    except Exception as e:
                        print(f"Failed to find CS events for trial {t} in {fish_id}: {e}")
                        cs_b, cs_e = [], []

                    try:
                        data_trial_events = data_trial[['Trial time (s)', 'US beg', 'US end']].copy()
                        us_b, us_e = analysis_utils.find_events(data_trial_events, 'US beg', 'US end', 'Trial time (s)')
                    except Exception as e:
                        print(f"Failed to find US events for trial {t} in {fish_id}: {e}")
                        us_b, us_e = [], []

                    # Plot traces or events
                    if metric in [tail_col, 'Vigor (deg/ms)']:
                        axs[t_i].plot(
                            data_trial['Trial time (s)'],
                            data_trial[metric],
                            'k',
                            alpha=1,
                            lw=0.3,
                            clip_on=False,
                        )
                    else:
                        axs[t_i].eventplot(
                            data_trial.loc[data_trial[metric], 'Trial time (s)'].to_list(),
                            color='k',
                            alpha=1,
                            lineoffsets=1,
                            linelengths=1,
                        )

                    # Add stimulus markers
                    for cs_beg_, cs_end_ in zip(cs_b, cs_e):
                        axs[t_i].axvline(cs_beg_, color=gen_config.plotting.cs_color, alpha=0.95, lw=2, linestyle='-')
                        axs[t_i].axvline(cs_end_, color=gen_config.plotting.cs_color, alpha=0.95, lw=2, linestyle='--')

                    for us_beg_, us_end_ in zip(us_b, us_e):
                        axs[t_i].axvline(us_beg_, color=gen_config.plotting.us_color, alpha=0.95, lw=2, linestyle='-')
                        axs[t_i].axvline(us_end_, color=gen_config.plotting.us_color, alpha=0.95, lw=2, linestyle='--')

                    axs[t_i].set_title(t, fontsize='small', loc='left')
                    configure_axes_for_pipeline(
                        axs[t_i],
                        xlim=(gen_config.time_bef_s, gen_config.time_aft_s),
                        show_xticks=False,
                        show_yticks=False,
                        hide_spines=('top', 'right', 'bottom', 'left'),
                    )
                    if metric == tail_col:
                        axs[t_i].set_ylim((-50, 50))

                configure_axes_for_pipeline(
                    axs[-1],
                    xlim=(gen_config.time_bef_s, gen_config.time_aft_s),
                    x_ticks=[gen_config.time_bef_s, 0, 10, gen_config.time_aft_s],
                    show_xticks=True,
                    show_yticks=True,
                )

                if metric == tail_col:
                    axs[-1].set_ylim((-50, 50))
                    axs[-1].set_yticks([-50, 50])
                elif metric == 'Vigor (deg/ms)':
                    baseline_mask = data['Trial time (s)'].between(-gen_config.baseline_window, 0)
                    baseline_q95 = data_trial.loc[baseline_mask, metric].quantile(0.95)
                    axs[-1].set_ylim((0, baseline_q95))
                elif metric in ['Bout', 'Bout beg', 'Bout end']:
                    axs[-1].set_ylim((0.5, 1))

                axs[-1].set_xlabel('Time relative to CS onset (s)')
                fig.subplots_adjust(hspace=1.5)
                fig.supylabel('Tail angle (deg)')
                fig.set_size_inches(fig.get_size_inches()[0], 20)
                fig.savefig(str(fig_path_tail), format=FIG_FORMAT, dpi=300, transparent=False, bbox_inches="tight")
                plt.close(fig)

            # Plot 2: Raw Vigor Heatmap
            if do_raw:
                data_plot = data.copy(deep=True)
                data_plot = data_plot[data_plot['Trial time (s)'].between(-WINDOW_DATA_PLOT_S, WINDOW_DATA_PLOT_S)]

                v_min, v_max = np.quantile(data_plot['Vigor (deg/ms)'], [0.1, 0.9])

                data_plot = (
                    data_plot[['Trial time (s)', 'Trial number', 'Vigor (deg/ms)']]
                    .pivot(index='Trial time (s)', columns='Trial number')
                    .reset_index()
                    .iloc[::gen_config.plotting.downsampling_step]
                    .set_index('Trial time (s)')
                    .droplevel(0, axis=1)
                    .T
                )

                phases_trials = config.blocks_dict['blocks phases'][csus]['trials in each block']
                phases_names = config.blocks_dict['blocks phases'][csus]['names of blocks']

                fig, axs = plt.subplots(
                    len(phases_trials),
                    1,
                    facecolor='white',
                    gridspec_kw={'height_ratios': [len(x) for x in phases_trials], 'hspace': 0},
                    sharex=True,
                    squeeze=False,
                )

                for b_i, b in enumerate(phases_names):
                    sns.heatmap(
                        data_plot[data_plot.index.isin(phases_trials[b_i])],
                        cbar=False,
                        robust=False,
                        xticklabels=xtick_step_raw,
                        yticklabels=False,
                        ax=axs[b_i][0],
                        clip_on=False,
                        vmin=v_min,
                        vmax=v_max,
                    )

                    xlims = axs[b_i][0].get_xlim()
                    middle = np.mean(xlims)
                    factor = (xlims[-1] - xlims[0]) / (2 * WINDOW_DATA_PLOT_S)

                    axs[b_i][0].set_ylabel(b, color='k', rotation=90, loc='center')
                    axs[b_i][0].axvline(middle, color='white', alpha=0.95, lw=1, linestyle='-')

                    if csus == 'CS':
                        axs[b_i][0].axvline(
                            middle + stim_duration * factor, color='white', alpha=0.95, lw=1, linestyle='-'
                        )

                    axs[b_i][0].axhline(axs[b_i][0].get_ylim()[0], color='white', alpha=0.95, lw=2, linestyle='-')

                configure_axes_for_pipeline(
                    axs[-1][0],
                    show_xticks=True,
                    show_yticks=True,
                    hide_spines=('top', 'right'),
                )
                axs[-1][0].set_xlabel(f'Time relative to \n{csus} onset (s)')

                fig.set_size_inches(fig.get_size_inches()[0], len(phases_trials) * 4)
                fig.savefig(str(fig_path_raw), format=FIG_FORMAT, dpi=300, transparent=False, bbox_inches='tight')
                plt.close(fig)

            # Plot 3: Scaled Vigor Heatmap
            if do_sc:
                data_plot = data.copy(deep=True)
                data_plot = data_plot[data_plot['Block name'] != '']
                data_plot = data_plot[data_plot['Trial time (s)'].between(-WINDOW_DATA_PLOT_S, WINDOW_DATA_PLOT_S)]
                data_plot.loc[~data_plot['Bout'], 'Vigor (deg/ms)'] = np.nan

                for t in data_plot['Trial number'].unique():
                    mask_trial = data_plot['Trial number'] == t
                    data_trial = data_plot.loc[mask_trial].copy(deep=True)

                    beg_bouts_trial, end_bouts_trial = analysis_utils.find_events(
                        data_trial, 'Bout beg', 'Bout end', 'Trial time (s)'
                    )

                    # Replace samples within each bout by the bout-mean vigor
                    for bout_b, bout_e in zip(beg_bouts_trial, end_bouts_trial):
                        mask_bout = data_trial['Trial time (s)'].between(bout_b, bout_e)
                        mean_vigor = data_trial.loc[mask_bout, 'Vigor (deg/ms)'].mean()
                        data_trial.loc[mask_bout, 'Vigor (deg/ms)'] = mean_vigor

                    # Use baseline quantiles for min/max scaling
                    mask_baseline = data_trial['Trial time (s)'] < -gen_config.baseline_window
                    baseline_vigor = data_trial.loc[mask_baseline, 'Vigor (deg/ms)'].dropna().values

                    if baseline_vigor.size == 0:
                        continue

                    min_vigor_pre_stim, max_vigor_pre_stim = np.quantile(baseline_vigor, [0.1, 0.9])

                    if np.isnan(max_vigor_pre_stim) or min_vigor_pre_stim == max_vigor_pre_stim:
                        continue

                    data_trial.loc[mask_trial, 'Vigor (deg/ms)'] = (
                        (data_trial.loc[mask_trial, 'Vigor (deg/ms)'] - min_vigor_pre_stim) /
                        (max_vigor_pre_stim - min_vigor_pre_stim)
                    )

                    data_trial['Vigor (deg/ms)'] = data_trial['Vigor (deg/ms)'].clip(0, 1)
                    data_plot.loc[mask_trial] = data_trial

                data_plot.loc[~data_plot['Bout'], 'Vigor (deg/ms)'] = np.nan
                data_plot = (
                    data_plot[['Trial time (s)', 'Trial number', 'Vigor (deg/ms)']]
                    .pivot(index='Trial time (s)', columns='Trial number')
                    .reset_index()
                    .set_index('Trial time (s)')
                    .droplevel(0, axis=1)
                    .T
                )

                data_plot.columns = data_plot.columns.astype('int')

                if csus == 'CS':
                    phases_trial_numbers = config.trials_cs_blocks_phases
                    phases_block_names = config.names_cs_blocks_phases
                else:
                    phases_trial_numbers = config.trials_us_blocks_phases
                    phases_block_names = config.names_us_blocks_phases

                fig, axs = plt.subplots(
                    len(phases_trial_numbers),
                    1,
                    facecolor='white',
                    gridspec_kw={
                        'height_ratios': [len(b) for b in phases_trial_numbers],
                        'hspace': 0.025,
                    },
                    squeeze=False,
                    constrained_layout=False,
                    figsize=(5 / 2.54, 6 / 2.54),
                )

                if csus == 'CS':
                    for b_i, b in enumerate(phases_block_names):
                        show_xticks = (b_i == len(phases_block_names) - 1)
                        sns.heatmap(
                            data_plot[data_plot.index.isin(phases_trial_numbers[b_i])],
                            cbar=False,
                            robust=False,
                            xticklabels=xtick_step_scaled if show_xticks else False,
                            yticklabels=False,
                            ax=axs[b_i][0],
                            clip_on=False,
                            vmin=0,
                            vmax=1,
                            rasterized=True,
                        )

                        if not show_xticks:
                            axs[b_i][0].set_xlabel('')

                        axs[b_i][0].set_ylabel(b, va='center', ha='center')

                        xlims = axs[b_i][0].get_xlim()
                        middle = np.mean(xlims)
                        factor = (xlims[-1] - xlims[0]) / (WINDOW_DATA_PLOT_S * 2)

                        axs[b_i][0].axvline(
                            middle, color=gen_config.plotting.cs_color, alpha=0.7, lw=1.5, linestyle='-'
                        )
                        axs[b_i][0].axvline(
                            middle + config.cs_duration * factor,
                            color=gen_config.plotting.cs_color,
                            alpha=0.7,
                            lw=1.5,
                            linestyle='-',
                        )

                        axs[b_i][0].set_facecolor('k')
                        axs[b_i][0].set_rasterization_zorder(0)

                    configure_axes_for_pipeline(
                        axs[-1][0],
                        show_xticks=True,
                        show_yticks=True,
                        hide_spines=('top', 'right'),
                    )
                    axs[-1][0].set_xlabel(f'Time relative to \n{csus} onset (s)')
                    axs[0][0].set_title(fish_id, loc='left')

                else:
                    # For US, show only the train block
                    train_idx = 1 if len(phases_trial_numbers) > 1 else 0
                    sns.heatmap(
                        data_plot[data_plot.index.isin(phases_trial_numbers[train_idx])],
                        cbar=True,
                        robust=False,
                        xticklabels=xtick_step_scaled,
                        yticklabels=False,
                        ax=axs[train_idx][0],
                        clip_on=False,
                        vmin=0,
                        vmax=1,
                        rasterized=True,
                    )

                    axs[train_idx][0].set_xlabel('')
                    xlims = axs[train_idx][0].get_xlim()
                    middle = np.mean(xlims)

                    axs[train_idx][0].set_facecolor('k')
                    axs[train_idx][0].set_rasterization_zorder(0)
                    axs[train_idx][0].set_ylabel('Train', va='center', ha='center')

                    for i in range(len(phases_trial_numbers)):
                        if i != train_idx:
                            axs[i][0].set_visible(False)

                    axs[train_idx][0].axvline(
                        middle, color=gen_config.plotting.us_color, alpha=0.7, lw=1.5, linestyle='-'
                    )

                    configure_axes_for_pipeline(
                        axs[train_idx][0],
                        show_xticks=True,
                        show_yticks=True,
                        hide_spines=('top', 'right'),
                    )
                    axs[train_idx][0].set_xlabel(f'Time relative to \n{csus} onset (s)')
                    axs[train_idx][0].set_title(fish_id, loc='left')

                fig.suptitle('A', fontsize=11, fontweight='bold', x=0, y=1, va='bottom', ha='left')
                fig.savefig(str(fig_path_sc), format=FIG_FORMAT, dpi=1000, transparent=False, bbox_inches='tight')
                plt.close(fig)

            # Plot 4: Normalized Vigor
            if do_norm:
                data_plot = data.copy(deep=True)
                data_plot = data_plot[data_plot['Block name'] != '']
                data_plot.loc[~data_plot['Bout'], 'Vigor (deg/ms)'] = np.nan

                trials_blocks = config.blocks_dict['blocks 10 trials'][csus]['trials in each block']
                trial_numbers = trials_blocks[-1][-1] - trials_blocks[0][0] + 1
                
                data_trial_bef = np.ones(trial_numbers)
                data_trial_dur = np.ones(trial_numbers)
                data_trial_nv = np.ones(trial_numbers)

                trials_range = np.arange(trials_blocks[0][0], trials_blocks[-1][-1] + 1)

                for t_i, t in enumerate(trials_range):
                    mask_trial = data_plot['Trial number'] == t
                    data_trial = data_plot.loc[mask_trial]

                    if csus == 'CS':
                        data_trial_bef[t_i] = data_trial.loc[
                            data_trial['Trial time (s)'].between(-gen_config.baseline_window, 0),
                            'Vigor (deg/ms)'
                        ].mean()
                        data_trial_dur[t_i] = data_trial.loc[
                            data_trial['Trial time (s)'].between(cr_window[0], cr_window[1]),
                            'Vigor (deg/ms)'
                        ].mean()
                    else:
                        data_trial_bef[t_i] = data_trial.loc[
                            data_trial['Trial time (s)'].between(-gen_config.baseline_window - cr_window[1], -cr_window[1]),
                            'Vigor (deg/ms)'
                        ].mean()
                        data_trial_dur[t_i] = data_trial.loc[
                            data_trial['Trial time (s)'].between(cr_window[0] - cr_window[1], 0),
                            'Vigor (deg/ms)'
                        ].mean()

                    data_trial_nv[t_i] = data_trial_dur[t_i] / data_trial_bef[t_i]

                fig, axs = plt.subplots(2, 1, sharex=True, sharey=False, facecolor='white')

                for t_i in range(trial_numbers):
                    axs[1].plot(t_i + 1, data_trial_nv[t_i], '.', color='k', alpha=0.8, lw=0, markersize=10, clip_on=False)

                axs[0].legend(['Before CS onset', 'After CS onset'], loc='upper right', frameon=False)

                for ax_i, ax in enumerate(axs):
                    ax.locator_params(axis='y', tight=False, nbins=4)
                    configure_axes_for_pipeline(
                        ax,
                        show_xticks=(ax_i == len(axs) - 1),
                        show_yticks=True,
                        hide_spines=('top', 'right'),
                    )

                max_ylim = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])
                axs[0].set_ylim((None, max_ylim))

                axs[1].set_yticks([0.75, 1, 1.25])
                axs[1].set_yticklabels(['0.75', '1.0', '1.25'])
                axs[1].set_ylim(0.75, 1.25)

                axs[1].axhline(1, color='k', alpha=0.8, lw=1)
                axs[1].axhline(0, color='k', alpha=0.8, lw=1)

                axs[0].set_ylabel('Average vigor (deg/ms)')
                axs[1].set_ylabel('Normalized vigor (AU)')
                axs[1].set_xlabel('Trial number')

                fig.set_size_inches(fig.get_size_inches()[0], 7)
                fig.savefig(str(fig_path_norm), format=FIG_FORMAT, dpi=300, transparent=False)
                plt.close(fig)
            
    print("PLOT INDIVIDUAL TRIALS FINISHED")


# %%
def run_plot_protocols():
    """
    Plots the stimulus protocol as experienced by the fish.
    
    Verifies the timing of CS and US delivery by reading either:
    1. 'Stim control' text files (raw logs).
    2. Processed dataframe event columns.
    
    This is useful for quality control to ensure stimuli were delivered as programmed.
    """
    print("\n" + "="*80)
    print("RUNNING PLOT PROTOCOLS")
    print("="*80)

    config = get_experiment_config(EXPERIMENT_TYPE)
    
    (
     _, _, _, _, _, _, _, _, _, _, _, _, _, _,
     path_analysis_protocols, path_orig_pkl, _, _
    ) = file_utils.create_folders(config.path_save)

    # Prepare output folders for protocol figures.
    path_fish = path_analysis_protocols / 'Single fish' / 'From processed data'
    path_fish.mkdir(parents=True, exist_ok=True)
    (path_fish / 'Individual trials').mkdir(exist_ok=True)
    (path_fish / 'Blocks of trials').mkdir(exist_ok=True)
    path_sc = path_analysis_protocols / 'Single fish' / 'From stim control files'
    path_sc.mkdir(parents=True, exist_ok=True)

    def plot_protocol_from_stimcontrol(protocol: pd.DataFrame, fig_path: Path, time_bef_first_stim_ms: int) -> None:
        """Plot protocol timeline from stim control file."""
        protocol = protocol.copy()
        protocol['ISI'] = list(np.diff(protocol['beg (ms)'].to_numpy())) + [0]
        protocol['Time'] = [time_bef_first_stim_ms] + (
            protocol['ISI'][:-1].cumsum() + time_bef_first_stim_ms
        ).to_list()
        protocol.loc[protocol.index[1:], 'Time'] /= (1000 * 60)

        def to_event_list(values) -> list:
            """Convert protocol times into event list."""
            if isinstance(values, pd.Series):
                return values.to_list()
            if np.isscalar(values):
                return [values]
            return list(values)

        cs_beg = []
        us_beg = []
        if protocol.index.isin(['Cycle']).any():
            cs_beg = to_event_list(protocol.loc['Cycle', 'Time'])
        if protocol.index.isin(['Reinforcer']).any():
            us_beg = to_event_list(protocol.loc['Reinforcer', 'Time'])

        fig, axs = plt.subplots(2, 1, sharex=True, constrained_layout=True, figsize=(20, 5))
        axs[0].eventplot(cs_beg, color=gen_config.plotting.cs_color, label='CS')
        axs[0].eventplot(
            us_beg,
            color=gen_config.plotting.us_color,
            lineoffsets=1.7,
            linelengths=0.8,
        )
        axs[0].axes.get_yaxis().set_visible(False)
        axs[0].spines[:].set_visible(False)

        axs[-1].set_xlim((0, None))
        axs[-1].set_xlabel('Time (min)')

        fig.legend()
        fig.suptitle('Total duration: ' + str(round(protocol.loc[:, 'Time'].iloc[-1] / 60, 2)) + ' h')
        fig.savefig(str(fig_path), format=PROTOCOLS_FIG_FORMAT, dpi=100, facecolor='white')
        plt.close(fig)

    # Constants for protocol plotting
    time_bef_first_stim_ms = 10000
    t_axis = (-40, 47)
    bins = np.arange(t_axis[0], t_axis[1], 4)
    time_col = gen_config.time_trial_frame_label

    # Iterate stim control paths
    if PLOT_FROM_STIM_CONTROL and config.path_home and Path(config.path_home).exists():
        all_stimcontrol_paths = list(Path(config.path_home).glob('*stim control.txt'))
        if FILTER_FISH_ID:
            all_stimcontrol_paths = [p for p in all_stimcontrol_paths if FILTER_FISH_ID in p.name]
        
        for protocol_path in all_stimcontrol_paths:
            stem_fish_path_orig = protocol_path.stem.replace('_stim control', '').lower()
            fig_protocol_name = path_sc / stem_fish_path_orig

            protocol = data_io.read_protocol(protocol_path)
            if protocol is None:
                print(f'Problems in protocol file: {stem_fish_path_orig}')
                continue

            plot_protocol_from_stimcontrol(protocol, fig_protocol_name.with_suffix(f'.{PROTOCOLS_FIG_FORMAT}'), time_bef_first_stim_ms)
            print(f'Plotted protocol from stim control: {stem_fish_path_orig}')

    # Iterate processed data
    if PLOT_FROM_PROCESSED_DATA:
        all_fish_data_paths = list(Path(path_orig_pkl).glob('*.pkl'))
        if FILTER_FISH_ID:
            all_fish_data_paths = [p for p in all_fish_data_paths if FILTER_FISH_ID in p.name]
        
        for fish_path in all_fish_data_paths:
            stem_fish_path_orig = fish_path.stem.lower()
            
            fig_protocol_individual = path_fish / 'Individual trials' / (stem_fish_path_orig + '_protocol from processed data')
            fig_protocol_blocks = path_fish / 'Blocks of trials' / (stem_fish_path_orig + '_protocol from processed data')

            # Skip if both figures already exist
            if (not PLOT_PROTOCOLS_OVERWRITE and 
                fig_protocol_individual.with_suffix(f'.{PROTOCOLS_FIG_FORMAT}').exists() and
                fig_protocol_blocks.with_suffix(f'.{PROTOCOLS_FIG_FORMAT}').exists()):
                print(f"Skipping existing protocol plots for {stem_fish_path_orig}")
                continue

            print(f'Processing protocol plots for: {stem_fish_path_orig}')

            try:
                data = pd.read_pickle(fish_path, compression='gzip')
            except Exception as e:
                print(f'Failed to read {stem_fish_path_orig}: {e}')
                continue

            data = analysis_utils.standardize_stim_cols(data)
            data = analysis_utils.identify_blocks_trials(data, config.blocks_dict)

            # Select columns needed for protocol plotting
            cols = [
                'Original frame number',
                time_col,
                'CS beg',
                'CS end',
                'US beg',
                'US end',
                'Trial number',
                'Block name',
                'Trial type',
            ]
            cols = [c for c in cols if c in data.columns]
            data = data.loc[:, cols].reset_index(drop=True)

            # Determine CS or US trials
            if (data_filtered := data[data['Trial type'] == 'CS']).empty:
                data_filtered = data[data['Trial type'] == 'US']
                csus = 'US'
            else:
                csus = 'CS'

            data_filtered = data_filtered.drop(columns='Trial type')

            # Extract event times
            data_cs_beg = data_filtered.loc[data_filtered['CS beg'] != 0, [time_col, 'Trial number', 'Block name']].copy()
            data_cs_beg['Stim'] = 'CS beg'

            data_cs_end = data_filtered.loc[data_filtered['CS end'] != 0, [time_col, 'Trial number', 'Block name']].copy()
            data_cs_end['Stim'] = 'CS end'

            data_us_beg = data_filtered.loc[data_filtered['US beg'] != 0, [time_col, 'Trial number', 'Block name']].copy()
            data_us_beg['Stim'] = 'US beg'

            data_us_end = data_filtered.loc[data_filtered['US end'] != 0, [time_col, 'Trial number', 'Block name']].copy()
            data_us_end['Stim'] = 'US end'

            data_events = pd.concat([data_cs_beg, data_cs_end, data_us_beg, data_us_end])
            data_events['Stim'] = data_events['Stim'].astype('category')
            data_events['Trial number'] = data_events['Trial number'].astype('category')
            data_events['Trial number'] = data_events['Trial number'].cat.remove_unused_categories()

            data_events = analysis_utils.convert_time_from_frame_to_s(data_events)
            
            if 'Block name' in data_events.columns:
                data_events['Block name'] = data_events['Block name'].astype('category')
                if config.blocks_dict['blocks 10 trials'][csus]['names of blocks']:
                    data_events['Block name'] = data_events['Block name'].cat.set_categories(
                        config.blocks_dict['blocks 10 trials'][csus]['names of blocks'],
                        ordered=True,
                    ).cat.remove_unused_categories()

            # Plot 1: Individual trials
            if PLOT_PROTOCOLS_OVERWRITE or not fig_protocol_individual.with_suffix(f'.{PROTOCOLS_FIG_FORMAT}').exists():
                g = sns.FacetGrid(data_events, row='Trial number', height=0.8, aspect=3.8, despine=True, sharex=True, sharey=True)
                fig, axs = g.figure, g.axes

                for t_i, t in enumerate(data_events['Trial number'].cat.categories):
                    data_plot = data_events.loc[data_events['Trial number'] == t, :]

                    axs[t_i][0].eventplot(
                        data_plot.loc[data_plot['Stim'] == 'CS beg', 'Trial time (s)'].to_list(),
                        color=gen_config.plotting.cs_color,
                        alpha=1,
                        lineoffsets=0,
                        linelengths=1,
                        label='CS beg',
                    )
                    axs[t_i][0].eventplot(
                        data_plot.loc[data_plot['Stim'] == 'CS end', 'Trial time (s)'].to_list(),
                        color=gen_config.plotting.cs_color,
                        alpha=1,
                        lineoffsets=0.3,
                        linelengths=0.5,
                        label='CS end',
                    )
                    axs[t_i][0].eventplot(
                        data_plot.loc[data_plot['Stim'] == 'US beg', 'Trial time (s)'].to_list(),
                        color=gen_config.plotting.us_color,
                        alpha=1,
                        lineoffsets=0,
                        linelengths=1,
                        label='US beg',
                    )
                    axs[t_i][0].eventplot(
                        data_plot.loc[data_plot['Stim'] == 'US end', 'Trial time (s)'].to_list(),
                        color=gen_config.plotting.us_color,
                        alpha=1,
                        lineoffsets=0.3,
                        linelengths=0.5,
                        label='US end',
                    )

                    configure_axes_for_pipeline(
                        axs[t_i][0],
                        xlim=t_axis,
                        x_ticks=np.arange(t_axis[0], t_axis[1], 10),
                        show_xticks=True,
                        show_yticks=False,
                        hide_spines=('top', 'right', 'bottom', 'left'),
                    )

                    axs[t_i, 0].set_title('')
                    axs[t_i, 0].set_title(t, loc='left')

                axs[-1][0].set_xlim(t_axis)
                axs[-1][0].set_xlabel('t (s)')
                axs[0][0].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
                fig.set_facecolor('white')

                fig.savefig(str(fig_protocol_individual.with_suffix(f'.{PROTOCOLS_FIG_FORMAT}')), 
                           format=PROTOCOLS_FIG_FORMAT, dpi=150, transparent=False, bbox_inches="tight")
                plt.close(fig)

            # Plot 2: Blocks of trials
            if PLOT_PROTOCOLS_OVERWRITE or not fig_protocol_blocks.with_suffix(f'.{PROTOCOLS_FIG_FORMAT}').exists():
                data_events_blocks = data_events.dropna()

                g = sns.FacetGrid(data_events_blocks, row='Block name', height=1.2, aspect=3.8, despine=True, sharex=True, sharey=True)
                fig, axs = g.figure, g.axes

                for t_i, t in enumerate(data_events_blocks['Block name'].cat.categories):
                    data_plot = data_events_blocks.loc[data_events_blocks['Block name'] == t, ['Trial time (s)', 'Stim']]

                    axs[t_i][0].hist(
                        data_plot.loc[data_plot['Stim'] == 'CS beg', 'Trial time (s)'],
                        bins=bins,
                        range=t_axis,
                        density=True,
                        histtype='stepfilled',
                        align='left',
                        lw=0,
                        color=gen_config.plotting.cs_color,
                        alpha=0.6,
                        label='CS beg',
                    )
                    axs[t_i][0].hist(
                        data_plot.loc[data_plot['Stim'] == 'US beg', 'Trial time (s)'],
                        bins=bins,
                        range=t_axis,
                        density=True,
                        histtype='stepfilled',
                        align='left',
                        lw=0,
                        color=gen_config.plotting.us_color,
                        alpha=0.6,
                        label='US beg',
                    )

                    configure_axes_for_pipeline(
                        axs[t_i][0],
                        xlim=t_axis,
                        x_ticks=np.arange(t_axis[0], t_axis[1], 10),
                        show_xticks=True,
                        show_yticks=False,
                        hide_spines=('top', 'right', 'bottom', 'left'),
                    )

                    axs[t_i, 0].set_title('')
                    axs[t_i, 0].set_title(t, loc='left')

                    axs[t_i][0].set_xlim(t_axis)
                    axs[t_i][0].set_xlabel('t (s)')

                axs[0][0].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
                fig.set_facecolor('white')

                fig.savefig(str(fig_protocol_blocks.with_suffix(f'.{PROTOCOLS_FIG_FORMAT}')), 
                           format=PROTOCOLS_FIG_FORMAT, dpi=150, transparent=False, bbox_inches="tight")
                plt.close(fig)

    print("PLOT PROTOCOLS FINISHED")


# %%
def run_discard():
    """
    Canonical discard flow (aligned to `Discard/Discard fish.py`).
    """
    config = get_experiment_config(EXPERIMENT_TYPE)

    def fish_id_from_path(fish_path: Path) -> str:
        """Extract fish ID from path."""
        return file_utils.fish_id_from_path(fish_path)

    def write_to_txt(fish_name, reason):
        """Log discard reasons to text files."""
        print(fish_name)
        with open(path_processed_data / 'Fish to discard.txt', 'a') as file:
            file.write(f'{fish_name}  {reason}\n')
        with open(path_processed_data / 'Discarded_fish_IDs.txt', 'a') as file:
            file.write(f"'{fish_name}', ")

    def record_excluded_fish(fish_name):
        """Track excluded fish IDs."""
        if fish_name in excluded_fish_ids:
            return
        excluded_fish_ids.add(fish_name)
        with open(excluded_ids_path, 'a') as file:
            file.write(f"{fish_name}\n")

    def move_raw_data_files(fish_name, raw_excluded_dir, path_home):
        """Move matching raw files to Excluded directory."""
        if raw_excluded_dir is None or not path_home or not Path(path_home).exists():
            return
        raw_excluded_dir.mkdir(exist_ok=True)
        matches = [p for p in Path(path_home).rglob(f"{fish_name}*") if p.is_file()]
        for raw_path in matches:
            if raw_excluded_dir in raw_path.parents:
                continue
            target = raw_excluded_dir / raw_path.name
            if target.exists():
                stem = raw_path.stem
                suffix = raw_path.suffix
                idx = 1
                while (raw_excluded_dir / f"{stem}_{idx}{suffix}").exists():
                    idx += 1
                target = raw_excluded_dir / f"{stem}_{idx}{suffix}"
            raw_path.replace(target)

    def exclude(fish_path, reason, cleanup_refs=None):
        """Apply exclusion: log reason, move pkl, and move raw files."""
        fish_name = fish_id_from_path(fish_path)
        write_to_txt(fish_name, reason)
        record_excluded_fish(fish_name)
        try:
            fish_path.replace(excluded_dir / fish_path.name)
        except Exception as e:
            write_to_txt(fish_name, f'failed to move file: {e}')
        move_raw_data_files(fish_name, raw_excluded_dir, config.path_home)
        if cleanup_refs:
            for ref in cleanup_refs:
                if hasattr(ref, "clear"):
                    ref.clear()

    def valid_block_mask(df):
        """Filter rows with valid block names."""
        return df['Block name'].notna() & df['Block name'].ne('')

    def window_mask(df, start_s, end_s):
        """Select trial-time window."""
        return df['Trial time (s)'].ge(start_s) & df['Trial time (s)'].le(end_s)

    def select_block_window(df, block_name, start_s, end_s):
        """Filter a block and time window."""
        return df[
            valid_block_mask(df)
            & df['Block name'].str.contains(block_name)
            & window_mask(df, start_s, end_s)
        ]

    def trials_with_no_bout(df):
        """Find trials without bouts."""
        trials_sum = df.groupby('Trial number', observed=True)['Bout'].sum()
        return trials_sum[trials_sum == 0].index.tolist()

    def check_viability(data_us, us_window):
        """Check viability in last US trial."""
        if data_us.empty:
            return False, 'no US trials found'
        max_trial = int(data_us['Trial number'].max())
        last_us_end = data_us.loc[
            (data_us['Trial number'] == max_trial) & (data_us['US end'] > 0),
            'Trial time (s)',
        ]
        if last_us_end.empty:
            return False, 'missing US end in last trial'
        if last_us_end.iloc[0] < 0.4:
            return False, 'US end too short in last trial'
        has_bout_last_trial = data_us.loc[
            (data_us['Trial number'] == max_trial) & window_mask(data_us, 0, us_window),
            'Bout',
        ].any()
        if not has_bout_last_trial:
            return False, 'failed viability test'
        return True, None

    def check_train(data_us, us_window):
        """Check Train block bouts."""
        data_us_train = select_block_window(data_us, 'Train', 0, us_window)
        if data_us_train.empty:
            return False, 'no Train block found'
        trials_no_bout = trials_with_no_bout(data_us_train)
        if trials_no_bout:
            return False, f'no bouts in Train for trial(s): {trials_no_bout}'
        return True, None

    def check_retrain(data_us, us_window):
        """Check Re-Train block bouts."""
        data_us_retrain = select_block_window(data_us, 'Re-Train', 0, us_window)
        if data_us_retrain.empty:
            return True, None
        trials_no_bout = trials_with_no_bout(data_us_retrain)
        if trials_no_bout:
            return False, f'no bouts in Retrain for trial(s): {trials_no_bout}'
        return True, None

    def check_baseline(data_cs, blocks_trials_sets, blocks_chosen, min_trials, baseline_window):
        """Check baseline bouts per block."""
        data_cs_pre = data_cs[window_mask(data_cs, -baseline_window, 0) & data_cs['Bout']]
        for block_name, trials_block_set in zip(blocks_chosen, blocks_trials_sets):
            trials_with_bouts = data_cs_pre.loc[
                data_cs_pre['Trial number'].isin(trials_block_set), 'Trial number'
            ].nunique()
            if trials_with_bouts < min_trials:
                return (
                    False,
                    f'less than {min_trials} trials with bouts in pre-CS period in {block_name}',
                )
        return True, None

    def check_cr(data_cs, blocks_trials_sets, blocks_chosen, min_trials, cr_window):
        """Check CR window bouts per block."""
        data_cs_cr = data_cs[window_mask(data_cs, cr_window[0], cr_window[1]) & data_cs['Bout']]
        for block_name, trials_block_set in zip(blocks_chosen, blocks_trials_sets):
            trials_with_bouts = data_cs_cr.loc[
                data_cs_cr['Trial number'].isin(trials_block_set), 'Trial number'
            ].nunique()
            if trials_with_bouts < min_trials:
                return (
                    False,
                    f'less than {min_trials} trials with bouts in CR period in {block_name}',
                )
        return True, None

    cr_window = config.cr_window
    print(config)
    print('cr window: ', cr_window)

    us_window = config.us_window_qc
    number_trials_block = config.number_trials_block
    min_number_trials_with_bouts_per_block = config.min_number_trials_with_bouts_per_block
    baseline_window = gen_config.baseline_window

    # Build contiguous blocks using the old convention (based on trial range).
    if config.trials_cs_blocks_10:
        blocks = [
            range(x, x + number_trials_block)
            for x in range(
                config.trials_cs_blocks_10[0][0],
                config.trials_cs_blocks_10[-1][-1] + 1,
                number_trials_block,
            )
        ]
    else:
        blocks = []

    base_block_names = []
    if config.names_cs_blocks_10 and len(config.names_cs_blocks_10) == len(config.trials_cs_blocks_10):
        base_block_names = list(config.names_cs_blocks_10)

    if base_block_names and len(blocks) == len(base_block_names):
        block_names = list(base_block_names)
    elif config.block_names and len(config.block_names) == len(blocks):
        block_names = list(config.block_names)
    elif base_block_names and len(base_block_names) and len(blocks) % len(base_block_names) == 0:
        blocks_per_base = len(blocks) // len(base_block_names)
        if blocks_per_base == 2:
            block_names = []
            for name in base_block_names:
                block_names.extend([f'Early {name}', f'Late {name}'])
        else:
            block_names = []
            for name in base_block_names:
                block_names.extend([f'{name} part {i + 1}' for i in range(blocks_per_base)])
    else:
        block_names = [f'Block {i + 1}' for i in range(len(blocks))]

    blocks_chosen = [name for name in config.blocks_chosen if name in block_names]
    if (
        not blocks_chosen
        and base_block_names
        and config.blocks_chosen
        and len(blocks) % len(base_block_names) == 0
    ):
        blocks_per_base = len(blocks) // len(base_block_names)
        base_index = {name: idx for idx, name in enumerate(base_block_names)}
        expanded = []
        for base_name in config.blocks_chosen:
            idx = base_index.get(base_name)
            if idx is None:
                continue
            start = idx * blocks_per_base
            expanded.extend(block_names[start:start + blocks_per_base])
        blocks_chosen = list(dict.fromkeys(expanded))
    if not blocks_chosen and block_names:
        blocks_chosen = list(block_names)

    blocks_chosen_set = set(blocks_chosen)
    name_to_trials = {name: set(trials) for name, trials in zip(block_names, blocks)}
    blocks_trials_sets = [name_to_trials[name] for name in blocks_chosen]

    (
        _,
        _,
        _,
        path_processed_data,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        path_orig_pkl,
        _,
        _,
    ) = file_utils.create_folders(config.path_save)
    all_fish_data_paths = list(path_orig_pkl.glob('*.pkl'))
    if FILTER_FISH_ID:
        all_fish_data_paths = [p for p in all_fish_data_paths if FILTER_FISH_ID in p.name]

    excluded_dir = path_orig_pkl / 'Excluded'
    excluded_dir.mkdir(exist_ok=True)
    raw_excluded_dir = None
    if config.path_home:
        raw_home = Path(config.path_home)
        if raw_home.exists() and str(raw_home) not in {".", ""}:
            raw_excluded_dir = raw_home / 'Excluded'
            raw_excluded_dir.mkdir(exist_ok=True)

    excluded_ids_path = excluded_dir / 'excluded_fish_ids.txt'
    excluded_ids_path.write_text('', encoding='utf-8')
    excluded_fish_ids = set()

    # Write analysis parameters as header to discard file.
    with open(path_processed_data / 'Fish to discard.txt', 'a') as file:
        file.write("Analysis Parameters:\n")
        file.write(f"  Experiment type: {EXPERIMENT_TYPE}\n")
        file.write(f"  Filter fish ID: {FILTER_FISH_ID}\n")
        file.write(f"  Path home: {config.path_home}\n")
        file.write(f"  Path save: {config.path_save}\n")
        file.write(f"  US window (s): {us_window}\n")
        file.write(f"  Baseline window (s): {baseline_window}\n")
        file.write(f"  CR window (s): {cr_window}\n")
        file.write(f"  Number trials per block: {number_trials_block}\n")
        file.write(f"  Minimum trials with bouts per block: {min_number_trials_with_bouts_per_block}\n")
        file.write(f"  Blocks chosen for QC: {blocks_chosen}\n")
        file.write(f"  Block names: {block_names}\n")
        file.write(f"  Contiguous blocks: {blocks}\n")
        file.write(f"  CS 10-trial blocks (config): {config.trials_cs_blocks_10}\n\n")

    # --- Main discard/QC loop ---
    for f_i, fish_path in tqdm(
        enumerate(all_fish_data_paths),
        total=len(all_fish_data_paths),
        desc="Processing fish",
    ):
        fish_name = fish_id_from_path(fish_path)
        print(f'\nProcessing fish {f_i + 1}/{len(all_fish_data_paths)}: {fish_name}')

        base_cols = ['Trial number', 'Bout', 'Trial type', 'Block name', 'CS beg', 'US beg', 'CS end', 'US end']
        time_cols = [gen_config.time_trial_frame_label, 'Trial time (s)']
        data = None
        last_error = None
        for time_col in time_cols:
            try:
                data = pd.read_pickle(fish_path, compression='gzip').loc[:, [time_col] + base_cols]
                break
            except Exception as e:
                last_error = e
                data = None
        if data is None:
            exclude(fish_path, f'failed to open file: {last_error}')
            continue

        data = analysis_utils.convert_time_from_frame_to_s(data)
        trial_type = data['Trial type']
        data_us = data[trial_type.eq('US')].copy()

        ok, reason = check_viability(data_us, us_window)
        if not ok:
            exclude(fish_path, reason)
            continue

        ok, reason = check_train(data_us, us_window)
        if not ok:
            exclude(fish_path, reason)
            continue

        ok, reason = check_retrain(data_us, us_window)
        if not ok:
            exclude(fish_path, reason)
            continue

        if blocks:
            data_cs = data[trial_type.eq('CS') & valid_block_mask(data)].copy()
            data_cs = analysis_utils.change_block_names(data_cs, blocks, block_names).copy()
            data_cs = data_cs[data_cs['Block name'].isin(blocks_chosen_set)].copy()
            data_cs.loc[:, 'Trial number'] = data_cs['Trial number'].cat.remove_unused_categories()
        else:
            data_cs = pd.DataFrame()

        if blocks and not data_cs.empty:
            ok, reason = check_baseline(
                data_cs,
                blocks_trials_sets,
                blocks_chosen,
                min_number_trials_with_bouts_per_block,
                baseline_window,
            )
            if not ok:
                exclude(fish_path, reason)
                continue

        if blocks and not data_cs.empty:
            ok, reason = check_cr(
                data_cs,
                blocks_trials_sets,
                blocks_chosen,
                min_number_trials_with_bouts_per_block,
                cr_window,
            )
            if not ok:
                exclude(fish_path, reason)
                continue

    missing = [
        fish_id
        for fish_id in excluded_fish_ids
        if not any(p.stem.startswith(fish_id) for p in excluded_dir.glob('*.pkl'))
    ]
    if missing:
        with open(path_processed_data / 'Fish to discard.txt', 'a') as file:
            file.write(f"Missing excluded PKL files for: {missing}\n")
        print(f"Warning: missing excluded PKL files for {len(missing)} fish.")

    print('Done')
# endregion Pipeline Functions


# region Main
if __name__ == "__main__":
    if RUN_PREPROCESS:
        run_preprocess()

    if RUN_PLOT_INDIVIDUALS:
        run_plot_individual_trials()

    if RUN_PLOT_PROTOCOLS:
        run_plot_protocols()

    if RUN_DISCARD:
        run_discard()
# endregion Main
