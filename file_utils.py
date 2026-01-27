"""File and path helpers shared across preprocessing and plotting."""
from pathlib import Path
from typing import Any, List, Tuple, Union

def create_folders(path_home: Path) -> Tuple[Path, ...]:
    """
    Creates necessary directories for the experiment analysis.
    """
    # Keep the return tuple order aligned with existing callers.
    path_lost_frames = path_home / 'Lost frames'
    path_lost_frames.mkdir(parents=True, exist_ok=True)

    path_summary_exp = path_home / 'Summary of protocol actually run'
    path_summary_exp.mkdir(parents=True, exist_ok=True)

    path_summary_beh = path_home / 'Summary of behavior'
    path_summary_beh.mkdir(parents=True, exist_ok=True)

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

    path_pooled_data = path_pkl / '3. Pooled data'
    path_pooled_data.mkdir(parents=True, exist_ok=True)

    return (path_lost_frames, path_summary_exp, path_summary_beh, path_processed_data, 
            path_cropped_exp_with_bout_detection, path_tail_angle_fig_cs, path_tail_angle_fig_us, 
            path_raw_vigor_fig_cs, path_raw_vigor_fig_us, path_scaled_vigor_fig_cs, 
            path_scaled_vigor_fig_us, path_normalized_fig_cs, path_normalized_fig_us, 
            path_pooled_vigor_fig, path_analysis_protocols, path_orig_pkl, 
            path_all_fish, path_pooled_data)

def msg(stem_fish_path_orig: Union[str, Path], message: Union[str, List[Any]]) -> List[str]:
    """Formats a message for logging."""
    if isinstance(message, list):
        message = '\t'.join([str(i) for i in message])
    
    return [str(stem_fish_path_orig)] + ['\t' + message + '\n']

def save_info(protocol_info_path: Path, stem_fish_path_orig: Union[str, Path], message: Union[str, List[Any]]):
    """Logs information to a file and prints it."""
    formatted_message = msg(stem_fish_path_orig, message)
    print(formatted_message)

    with open(protocol_info_path, 'a') as file:
        file.writelines(formatted_message)

def fish_id(stem_path: str) -> Tuple[str, str, str, str, str, str]:
    """Parses fish ID from filename."""
    # Info about a specific 'Fish'.
    
    stem_fish_path = stem_path.lower()
    parts = stem_fish_path.split('_')
    
    if len(parts) < 6:
        # Fallback or error handling if needed
        # Assuming format: day_fish#_cond_rig_strain_age
        # But code below uses specific indices
        pass

    day = parts[0]
    fish_number = parts[1]
    cond_type = parts[2]
    rig = parts[3]
    strain = parts[4]
    age = parts[5].replace('dpf', '')

    return day, strain, age, cond_type, rig, fish_number

def fish_id_from_path(fish_path: Path) -> str:
    """Extracts basic Fish ID (Day_Fish#) from path."""
    return '_'.join(fish_path.stem.split('_')[:2])


def load_excluded_fish_ids(excluded_dir: Path, filename: str = "excluded_fish_ids.txt") -> List[str]:
    """Load excluded fish IDs from the Excluded new folder."""
    excluded_path = excluded_dir / filename
    if not excluded_path.exists():
        return []
    lines = excluded_path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]
