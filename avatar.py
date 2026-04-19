import cv2
import numpy as np

from config import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)

BODY_COLOR = (220, 210, 180)
GLOW_COLOR = (255, 140, 60)
EYE_COLOR = (255, 255, 255)
EYE_CORE_COLOR = (200, 220, 255)
RIB_COLOR = tuple(int(channel * 0.55) for channel in BODY_COLOR)

HEAD_AXES = (35, 42)
HEAD_GLOW_PADDING = 7
LIMB_GLOW_PADDING = 6

UPPER_ARM_WIDTH = 14
FOREARM_WIDTH = 10
THIGH_WIDTH = 18
SHIN_WIDTH = 12

_previous_landmarks = None


def reset_avatar_motion():
    global _previous_landmarks
    _previous_landmarks = None


def _to_point(mapped_landmarks, landmark_index):
    point = mapped_landmarks.get(landmark_index)
    if point is None:
        return None
    return np.array(point, dtype=np.float32)


def _draw_filled_polygon(canvas, points, color):
    polygon = np.round(points).astype(np.int32)
    cv2.fillPoly(canvas, [polygon], color)


def _expand_polygon(points, padding):
    center = np.mean(points, axis=0)
    vectors = points - center
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_lengths = np.maximum(lengths, 1.0)
    return points + (vectors / safe_lengths) * padding


def draw_thick_limb(canvas, pt1, pt2, width, color):
    p1 = np.array(pt1, dtype=np.float32)
    p2 = np.array(pt2, dtype=np.float32)
    direction = p2 - p1
    length = np.linalg.norm(direction)

    if length < 1.0:
        cv2.circle(canvas, tuple(np.round(p1).astype(int)), max(int(width // 2), 1), color, -1)
        return

    direction /= length
    perpendicular = np.array([-direction[1], direction[0]], dtype=np.float32) * (width / 2.0)
    quad = np.array(
        [
            p1 + perpendicular,
            p1 - perpendicular,
            p2 - perpendicular,
            p2 + perpendicular,
        ],
        dtype=np.float32,
    )
    _draw_filled_polygon(canvas, quad, color)


def _draw_glowing_limb(canvas, pt1, pt2, width):
    draw_thick_limb(canvas, pt1, pt2, width + LIMB_GLOW_PADDING, GLOW_COLOR)
    draw_thick_limb(canvas, pt1, pt2, width, BODY_COLOR)


def _draw_head(canvas, nose_point):
    head_center = tuple(np.round(nose_point + np.array([0.0, -8.0], dtype=np.float32)).astype(int))
    glow_axes = (HEAD_AXES[0] + HEAD_GLOW_PADDING, HEAD_AXES[1] + HEAD_GLOW_PADDING)

    cv2.ellipse(canvas, head_center, glow_axes, 0, 0, 360, GLOW_COLOR, -1)
    cv2.ellipse(canvas, head_center, HEAD_AXES, 0, 0, 360, BODY_COLOR, -1)

    eye_y = head_center[1] - 4
    left_eye_center = (head_center[0] - 12, eye_y)
    right_eye_center = (head_center[0] + 12, eye_y)

    for eye_center in (left_eye_center, right_eye_center):
        cv2.ellipse(canvas, eye_center, (6, 3), 0, 0, 360, EYE_COLOR, -1)
        cv2.ellipse(canvas, eye_center, (2, 2), 0, 0, 360, EYE_CORE_COLOR, -1)


def _draw_torso(canvas, left_shoulder, right_shoulder, left_hip, right_hip):
    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    hip_center = (left_hip + right_hip) / 2.0

    upper_left = shoulder_center + (left_shoulder - shoulder_center) * 1.2 + np.array([-6.0, -6.0], dtype=np.float32)
    upper_right = shoulder_center + (right_shoulder - shoulder_center) * 1.2 + np.array([6.0, -6.0], dtype=np.float32)
    lower_left = hip_center + (left_hip - hip_center) * 0.72 + np.array([-2.0, 8.0], dtype=np.float32)
    lower_right = hip_center + (right_hip - hip_center) * 0.72 + np.array([2.0, 8.0], dtype=np.float32)

    torso = np.array([upper_left, upper_right, lower_right, lower_left], dtype=np.float32)
    glow_torso = _expand_polygon(torso, LIMB_GLOW_PADDING)

    _draw_filled_polygon(canvas, glow_torso, GLOW_COLOR)
    _draw_filled_polygon(canvas, torso, BODY_COLOR)

    left_edge_start, left_edge_end = upper_left, lower_left
    right_edge_start, right_edge_end = upper_right, lower_right
    for ratio in (0.28, 0.46, 0.64, 0.8):
        rib_left = left_edge_start + (left_edge_end - left_edge_start) * ratio
        rib_right = right_edge_start + (right_edge_end - right_edge_start) * ratio
        rib_left = rib_left + np.array([10.0, 0.0], dtype=np.float32)
        rib_right = rib_right - np.array([10.0, 0.0], dtype=np.float32)
        cv2.line(
            canvas,
            tuple(np.round(rib_left).astype(int)),
            tuple(np.round(rib_right).astype(int)),
            RIB_COLOR,
            2,
            cv2.LINE_AA,
        )


def draw_avatar(mapped_landmarks):
    global _previous_landmarks

    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)

    left_shoulder = _to_point(mapped_landmarks, LEFT_SHOULDER)
    right_shoulder = _to_point(mapped_landmarks, RIGHT_SHOULDER)
    left_elbow = _to_point(mapped_landmarks, LEFT_ELBOW)
    right_elbow = _to_point(mapped_landmarks, RIGHT_ELBOW)
    left_wrist = _to_point(mapped_landmarks, LEFT_WRIST)
    right_wrist = _to_point(mapped_landmarks, RIGHT_WRIST)
    left_hip = _to_point(mapped_landmarks, LEFT_HIP)
    right_hip = _to_point(mapped_landmarks, RIGHT_HIP)
    left_knee = _to_point(mapped_landmarks, LEFT_KNEE)
    right_knee = _to_point(mapped_landmarks, RIGHT_KNEE)
    left_ankle = _to_point(mapped_landmarks, LEFT_ANKLE)
    right_ankle = _to_point(mapped_landmarks, RIGHT_ANKLE)
    nose_point = _to_point(mapped_landmarks, NOSE)

    for start, end, width in (
        (left_hip, left_knee, THIGH_WIDTH),
        (right_hip, right_knee, THIGH_WIDTH),
        (left_knee, left_ankle, SHIN_WIDTH),
        (right_knee, right_ankle, SHIN_WIDTH),
    ):
        if start is not None and end is not None:
            _draw_glowing_limb(canvas, start, end, width)

    if all(point is not None for point in (left_shoulder, right_shoulder, left_hip, right_hip)):
        _draw_torso(canvas, left_shoulder, right_shoulder, left_hip, right_hip)

    for start, end, width in (
        (left_shoulder, left_elbow, UPPER_ARM_WIDTH),
        (right_shoulder, right_elbow, UPPER_ARM_WIDTH),
        (left_elbow, left_wrist, FOREARM_WIDTH),
        (right_elbow, right_wrist, FOREARM_WIDTH),
    ):
        if start is not None and end is not None:
            _draw_glowing_limb(canvas, start, end, width)

    if nose_point is not None:
        _draw_head(canvas, nose_point)

    _previous_landmarks = dict(mapped_landmarks)
    return canvas
