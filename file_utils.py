"""File and path helpers shared across preprocessing and plotting."""

import re
from pathlib import Path
from typing import Any, Iterable, List, NamedTuple, Tuple, Union


class PipelinePaths(NamedTuple):
    """Named container for every directory created by ``create_folders``.

    Attribute names match the variable names formerly unpacked from the raw
    tuple, so callers can migrate at their own pace:
    ``paths = create_folders(home); paths.orig_pkl`` instead of index [15].
    """
    lost_frames: Path
    summary_exp: Path
    summary_beh: Path
    processed_data: Path
    cropped_exp_with_bout_detection: Path
    tail_angle_fig_cs: Path
    tail_angle_fig_us: Path
    raw_vigor_fig_cs: Path
    raw_vigor_fig_us: Path
    scaled_vigor_fig_cs: Path
    scaled_vigor_fig_us: Path
    normalized_fig_cs: Path
    normalized_fig_us: Path
    pooled_vigor_fig: Path
    analysis_protocols: Path
    orig_pkl: Path
    all_fish: Path
    pooled_data: Path


def create_folders(path_home: Path) -> PipelinePaths:
    """Create all pipeline directories under *path_home*.

    Returns a :class:`PipelinePaths` named-tuple so callers can access
    directories by name instead of positional index.
    """
    path_lost_frames = path_home / 'Lost frames'
    path_summary_exp = path_home / 'Summary of protocol actually run'
    path_summary_beh = path_home / 'Summary of behavior'
    path_processed_data = path_home / 'Processed data'
    path_cropped_exp_with_bout_detection = path_processed_data / '1. summary of exp.'
    path_tail_angle_fig_cs = path_processed_data / '2. single fish_tail angle' / 'aligned to CS'
    path_tail_angle_fig_us = path_processed_data / '2. single fish_tail angle' / 'aligned to US'
    path_raw_vigor_fig_cs = path_processed_data / '3. single fish_raw vigor heatmap' / 'aligned to CS'
    path_raw_vigor_fig_us = path_processed_data / '3. single fish_raw vigor heatmap' / 'aligned to US'
    path_scaled_vigor_fig_cs = path_processed_data / '4. single fish_scaled vigor heatmap' / 'aligned to CS'
    path_scaled_vigor_fig_us = path_processed_data / '4. single fish_scaled vigor heatmap' / 'aligned to US'
    path_normalized_fig_cs = path_processed_data / '5. single fish_suppression ratio vigor trial' / 'aligned to CS'
    path_normalized_fig_us = path_processed_data / '5. single fish_suppression ratio vigor trial' / 'aligned to US'
    path_pooled_vigor_fig = path_processed_data / 'All fish'
    path_analysis_protocols = path_processed_data / 'Analysis of protocols'
    path_pkl = path_processed_data / 'pkl files'
    path_orig_pkl = path_pkl / '1. Original'
    path_all_fish = path_pkl / '2. All fish by condition'
    path_pooled_data = path_pkl / '3. Pooled data'

    # Create all directories in one pass.
    for p in [
        path_lost_frames, path_summary_exp, path_summary_beh,
        path_processed_data, path_cropped_exp_with_bout_detection,
        path_tail_angle_fig_cs, path_tail_angle_fig_us,
        path_raw_vigor_fig_cs, path_raw_vigor_fig_us,
        path_scaled_vigor_fig_cs, path_scaled_vigor_fig_us,
        path_normalized_fig_cs, path_normalized_fig_us,
        path_pooled_vigor_fig, path_analysis_protocols,
        path_pkl, path_orig_pkl, path_all_fish, path_pooled_data,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    return PipelinePaths(
        lost_frames=path_lost_frames,
        summary_exp=path_summary_exp,
        summary_beh=path_summary_beh,
        processed_data=path_processed_data,
        cropped_exp_with_bout_detection=path_cropped_exp_with_bout_detection,
        tail_angle_fig_cs=path_tail_angle_fig_cs,
        tail_angle_fig_us=path_tail_angle_fig_us,
        raw_vigor_fig_cs=path_raw_vigor_fig_cs,
        raw_vigor_fig_us=path_raw_vigor_fig_us,
        scaled_vigor_fig_cs=path_scaled_vigor_fig_cs,
        scaled_vigor_fig_us=path_scaled_vigor_fig_us,
        normalized_fig_cs=path_normalized_fig_cs,
        normalized_fig_us=path_normalized_fig_us,
        pooled_vigor_fig=path_pooled_vigor_fig,
        analysis_protocols=path_analysis_protocols,
        orig_pkl=path_orig_pkl,
        all_fish=path_all_fish,
        pooled_data=path_pooled_data,
    )

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


def _unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def load_fish_ids_from_text_file(path: Path) -> List[str]:
    """Load fish IDs from a plain-text file.

    Supports:
    - one ID per line
    - comma/semicolon/tab/space separated IDs
    - comments starting with '#'
    - optional quotes around tokens

    Returns IDs in first-seen order with duplicates removed.
    """
    if path is None or not Path(path).exists():
        return []

    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    tokens: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip inline comments.
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if not line:
            continue
        # Normalize separators to whitespace.
        line = re.sub(r"[;,\t]", " ", line)
        for tok in line.split():
            tok = tok.strip().strip('"').strip("'")
            if not tok:
                continue
            # Skip common header-like tokens.
            if tok.lower() in {"fish", "fish_id", "fishid", "id", "ids"}:
                continue
            tokens.append(tok)
    return _unique_preserve_order(tokens)


def load_discarded_fish_ids(discarded_file: Path) -> List[str]:
    """Load discarded fish IDs from a dedicated text file.

    The repository historically stored excluded IDs under "Excluded new".
    Newer pipelines store discarded IDs under
    "Processed data/pkl files/1. Original/Excluded/excluded_fish_ids.txt".
    """
    return load_fish_ids_from_text_file(discarded_file)


def load_excluded_fish_ids(excluded_dir: Path, filename: str = "excluded_fish_ids.txt") -> List[str]:
    """Load excluded fish IDs from the legacy "Excluded new" folder."""
    excluded_path = excluded_dir / filename
    return load_fish_ids_from_text_file(excluded_path)
