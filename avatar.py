import cv2
import numpy as np

from config import (
    AVATAR_COLOR,
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    JOINT_COLOR,
    JOINT_RADIUS,
    LINE_THICKNESS,
    NOSE,
    POSE_CONNECTIONS,
)

ACTIVE_COLOR = (60, 100, 255)
MAX_AVERAGE_DISPLACEMENT = 40.0

_previous_landmarks = None


def reset_avatar_motion():
    global _previous_landmarks
    _previous_landmarks = None


def _calculate_avatar_color(mapped_landmarks):
    global _previous_landmarks

    average_displacement = 0.0
    if _previous_landmarks:
        shared_indices = sorted(set(mapped_landmarks).intersection(_previous_landmarks))
        if shared_indices:
            displacements = [
                np.linalg.norm(np.subtract(mapped_landmarks[index], _previous_landmarks[index]))
                for index in shared_indices
            ]
            average_displacement = float(np.mean(displacements))

    motion_ratio = min(average_displacement / MAX_AVERAGE_DISPLACEMENT, 1.0)
    still_color = np.array(AVATAR_COLOR, dtype=np.float32)
    active_color = np.array(ACTIVE_COLOR, dtype=np.float32)
    blended_color = still_color + (motion_ratio * (active_color - still_color))
    _previous_landmarks = dict(mapped_landmarks)
    return tuple(int(round(channel)) for channel in blended_color)


def draw_avatar(mapped_landmarks):
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
    line_color = _calculate_avatar_color(mapped_landmarks)

    if NOSE in mapped_landmarks:
        cv2.circle(canvas, mapped_landmarks[NOSE], 20, AVATAR_COLOR, -1)

    for start_index, end_index in POSE_CONNECTIONS:
        if start_index in mapped_landmarks and end_index in mapped_landmarks:
            cv2.line(
                canvas,
                mapped_landmarks[start_index],
                mapped_landmarks[end_index],
                line_color,
                LINE_THICKNESS,
            )

    for point in mapped_landmarks.values():
        cv2.circle(canvas, point, JOINT_RADIUS, JOINT_COLOR, -1)

    return canvas
