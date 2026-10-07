from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2


def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    duration = (frame_count / fps) if fps > 0 else 0.0
    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
    }


def extract_clip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_dir: str | None = None,
) -> str:
    """
    Extract a short MP4 clip using OpenCV.

    This avoids requiring the system `ffmpeg` command for the first prototype.
    The output codec is mp4v, which Qwen/torchvision/decord can normally decode.
    """
    info = get_video_info(video_path)
    duration = info["duration"]

    start_sec = max(0.0, float(start_sec))
    end_sec = min(max(start_sec + 0.1, float(end_sec)), duration)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = info["fps"] if info["fps"] > 0 else 25.0
    width = info["width"]
    height = info["height"]

    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)

    if output_dir is None:
        output_dir = tempfile.gettempdir()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fd, output_path = tempfile.mkstemp(
        prefix="driving_vlm_clip_",
        suffix=".mp4",
        dir=output_dir,
    )
    os.close(fd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    target_frames = max(1, int(round((end_sec - start_sec) * fps)))
    written = 0

    while written < target_frames:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()

    if written == 0:
        try:
            os.remove(output_path)
        except OSError:
            pass
        raise ValueError(
            f"No frames could be extracted from {start_sec:.1f}s to {end_sec:.1f}s."
        )

    return output_path
