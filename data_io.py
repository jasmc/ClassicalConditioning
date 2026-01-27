"""I/O helpers for camera, protocol, and tail-tracking files."""
import logging
from pathlib import Path
from timeit import default_timer as timer
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from general_configuration import config
import file_utils

logger = logging.getLogger(__name__)

def read_initial_abs_time(camera_path: Path) -> Optional[int]:
    """Read the absolute time at the beginning of the experiment from camera file."""
    try:
        with open(camera_path, 'r') as f:
            f.readline()
            return int(float(f.readline().strip('\n').split('\t')[2]))
    except Exception as e:
        logger.warning(f'No absolute time in cam file {camera_path}: {e}')
        return None

def read_camera(camera_path: str) -> Optional[pd.DataFrame]:
    """Reads camera timing data."""
    try:
        start = timer()
        
        # Determine separator/decimal format (space or tab, dot or comma).
        # Trying space first
        try:
             camera = pd.read_csv(str(camera_path), sep=' ', decimal='.', header=0, 
                                 skiprows=[*range(1, config.validation.number_frames_discard_beg)])
        except Exception:
             camera = pd.read_csv(str(camera_path), sep='\t', decimal=',', header=0, 
                                 skiprows=[*range(1, config.validation.number_frames_discard_beg)])

        if len(camera.columns) == 1:
             # Fallback if first read failed to separate columns correctly
             camera = pd.read_csv(str(camera_path), sep='\t', decimal=',', header=0, 
                                 skiprows=[*range(1, config.validation.number_frames_discard_beg)])

        camera.rename(columns={'TotalTime' : 'ElapsedTime', 'ID' : 'FrameID'}, inplace=True)
            
        camera = camera.astype({'FrameID' : 'int', 'ElapsedTime' : 'float', 'AbsoluteTime' : 'int64'}, errors='ignore')
        
        logger.info(f'Time to read cam.txt: {timer()-start} (s)')
        return camera

    except Exception as e:
        logger.error(f'Cannot read camera file {camera_path}: {e}')
        return None

def read_sync_reader(sync_reader_path: Path) -> Optional[pd.DataFrame]:
    """Reads sync reader data."""
    try:
        start = timer()
        sync_reader = pd.read_csv(str(sync_reader_path), sep=' ', header=0, decimal='.')
        logger.info(f'Time to read scape sync reader.txt: {timer()-start} (s)')
        return sync_reader
    except Exception:
        return None

def read_protocol(protocol_path: Path) -> Optional[pd.DataFrame]:
    """Reads protocol file."""
    if not Path(protocol_path).exists():
        logger.warning('stim control file does not exist.')
        return None

    try:
        protocol = pd.read_csv(str(protocol_path), sep=' ', header=0, 
                               names=['Experiment type', 'beg (ms)', 'end (ms)'], 
                               usecols=[0, 1, 2], index_col=0)
    except:
        protocol = pd.read_csv(str(protocol_path), sep='\t', header=0, 
                               names=['Experiment type', 'beg (ms)', 'end (ms)'], 
                               usecols=[0, 1, 2], index_col=0)

    if protocol.empty or (protocol.iloc[0,0] == 0 if not protocol.empty else True):
        return None
    
    protocol.sort_values(by='beg (ms)', inplace=True)
    return protocol

def map_abs_time_to_elapsed_time(camera: pd.DataFrame, protocol: pd.DataFrame) -> pd.DataFrame:
    """Maps absolute timestamps to elapsed time using camera data interpolation."""
    
    if 'AbsoluteTime' not in camera.columns:
        return protocol

    stimuli = protocol.index.unique()
    camera['AbsoluteTime'] = camera['AbsoluteTime'].astype('float')
    camera['ElapsedTime'] = camera['ElapsedTime'].astype('float')
    
    for beg_end in ['beg (ms)', 'end (ms)']:
        protocol_ = protocol.loc[:,beg_end].reset_index().rename(columns={beg_end : 'AbsoluteTime'})
        
        # Merge and interpolate
        camera_protocol = pd.merge_ordered(camera, protocol_).set_index('AbsoluteTime').interpolate(kind='slinear').reset_index()
        camera_protocol = camera_protocol.drop_duplicates('AbsoluteTime', keep='first')

        for stim in stimuli:
            subset = camera_protocol[camera_protocol['Experiment type']==stim].set_index('Experiment type').loc[:,'ElapsedTime']
            
            if len(subset) == 1:
                protocol.loc[stim,beg_end] = subset.to_numpy()[0]
            else:
                protocol.loc[stim,beg_end] = subset

    return protocol[protocol.notna().all(axis=1)]


def read_tail_tracking_data(data_path: Path) -> Optional[pd.DataFrame]:
    """
    Read and preprocess tail tracking data from a file.
    Optimized version with pyarrow/c engine.
    """
    start = timer()

    try:
        # Use pyarrow engine for faster reading when available.
        data = pd.read_csv(
            data_path, 
            sep=' ', 
            header=0, 
            usecols=config.cols_to_use_orig, 
            decimal=',', 
            engine='pyarrow',
            dtype={config.cols_to_use_orig[0]: 'int32'}
        )
        # Drop the trailing summary row emitted by the acquisition software.
        data = data.iloc[:-1]

    except Exception:
        try:
            data = pd.read_csv(
                data_path, 
                sep=' ', 
                header=0, 
                usecols=config.cols_to_use_orig, 
                decimal=',', 
                engine='c'
            )
            # Drop the trailing summary row emitted by the acquisition software.
            data = data.iloc[:-1]
        except Exception:
            logger.error('Tail tracking might be corrupted!')
            return None

    logger.info(f'Time to read tail tracking .txt: {timer()-start:.3f} (s)')

    # Store original frame number in an integer dtype for downstream joins.
    data['Original frame number'] = data['FrameID'].astype('int32', copy=False)

    # Optimize dtypes for angle columns only (avoid casting non-angle columns).
    angle_cols = [c for c in config.cols_to_use_orig[1:] if c in data.columns]
    if angle_cols:
         data = data.astype({col: 'float32' for col in angle_cols}, copy=False)

    # Convert radian to degree
    # Assuming columns other than FrameID are angles
    radian_cols = config.cols_to_use_orig[1:]
    # Check if columns exist before operation to be safe.
    existing_cols = [c for c in radian_cols if c in data.columns]
    if existing_cols:
        data.loc[:, existing_cols] = data.loc[:, existing_cols].values * np.float32(180/np.pi)
        
        # Rename columns
        # Generating target names: 'Angle of point 0 (deg)'
        target_names = [f'Angle of point {i} (deg)' for i in range(len(existing_cols))]
        data.rename(columns={old: new for old, new in zip(existing_cols, target_names)}, inplace=True)

    return data
