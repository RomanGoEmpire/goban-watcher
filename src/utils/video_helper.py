import glob
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike


def get_video_files(videos_path: Path, formats: list[str] = None) -> list[Path]:
    """
    Get all video files from the videos directory.

    Args:
        videos_path: Path to videos directory
        formats: List of video formats to search for (default: ['.mp4', '.avi', '.mov'])

    Returns:
        Sorted list of video file paths
    """
    if formats is None:
        formats = [".mp4", ".avi", ".mov"]

    video_files = []
    for fmt in formats:
        video_files.extend(videos_path.glob(f"*{fmt}"))
        video_files.extend(videos_path.glob(f"*{fmt.upper()}"))

    # Sort by filename (numeric if possible)
    def sort_key(path):
        try:
            return int(path.stem)
        except ValueError:
            return path.stem

    return sorted(video_files, key=sort_key)


def detect_board_position_change(
    frame1: MatLike,
    frame2: MatLike,
    corners1: list[list[int]],
    corners2: list[list[int]],
    threshold: float = 50.0,
) -> bool:
    """
    Detect if the board position has changed significantly between two frames.

    Args:
        frame1: First frame
        frame2: Second frame
        corners1: Corner positions for first frame
        corners2: Corner positions for second frame (can be same as corners1)
        threshold: Pixel distance threshold for detecting change

    Returns:
        True if board position changed significantly, False otherwise
    """
    # Calculate corner movement
    corners1_array = np.array(corners1, dtype=np.float32)
    corners2_array = np.array(corners2, dtype=np.float32)

    # Calculate Euclidean distance for each corner
    distances = np.linalg.norm(corners1_array - corners2_array, axis=1)
    max_distance = np.max(distances)

    if max_distance > threshold:
        return True

    # Additional check: compare image similarity in board region
    # Extract regions around corners and compare
    try:
        # Simple approach: compare histograms of the two frames
        hist1 = cv2.calcHist(
            [frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
        )
        hist2 = cv2.calcHist(
            [frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
        )

        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        # Calculate correlation
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # If correlation is too low, board position likely changed
        if correlation < 0.85:
            return True
    except Exception:
        pass

    return False


def create_video_capture(video_path: Path) -> cv2.VideoCapture:
    """
    Create a VideoCapture object for the given video file.

    Args:
        video_path: Path to video file

    Returns:
        cv2.VideoCapture object

    Raises:
        ValueError: If video cannot be opened
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    return cap


def get_video_info(cap: cv2.VideoCapture) -> dict:
    """
    Get information about the video.

    Args:
        cap: VideoCapture object

    Returns:
        Dictionary with video information (fps, frame_count, width, height)
    """
    return {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }


def seek_video(cap: cv2.VideoCapture, frame_number: int) -> bool:
    """
    Seek to a specific frame in the video.

    Args:
        cap: VideoCapture object
        frame_number: Frame number to seek to

    Returns:
        True if successful, False otherwise
    """
    return cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)


def get_current_frame_number(cap: cv2.VideoCapture) -> int:
    """
    Get the current frame number.

    Args:
        cap: VideoCapture object

    Returns:
        Current frame number
    """
    return int(cap.get(cv2.CAP_PROP_POS_FRAMES))


# Made with Bob
