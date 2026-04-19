from config import AVATAR_LANDMARKS, CANVAS_HEIGHT, CANVAS_WIDTH, FLIP_AVATAR_X

SMOOTHING_FACTOR = 0.4

_previous_landmarks = {}


def reset_mirror_smoothing():
    _previous_landmarks.clear()


def mirror_landmarks(landmarks):
    if not landmarks:
        return {}

    mirrored_points = {}

    for landmark_index in AVATAR_LANDMARKS:
        if landmark_index >= len(landmarks):
            continue

        x, y, visibility = landmarks[landmark_index]
        if visibility < 0.5:
            continue

        mapped_x = 1.0 - x if FLIP_AVATAR_X else x
        pixel_x = max(0, min(CANVAS_WIDTH - 1, int(round(mapped_x * CANVAS_WIDTH))))
        pixel_y = max(0, min(CANVAS_HEIGHT - 1, int(round(y * CANVAS_HEIGHT))))

        previous_point = _previous_landmarks.get(landmark_index)
        if previous_point is not None:
            previous_x, previous_y = previous_point
            pixel_x = int(round((SMOOTHING_FACTOR * pixel_x) + ((1.0 - SMOOTHING_FACTOR) * previous_x)))
            pixel_y = int(round((SMOOTHING_FACTOR * pixel_y) + ((1.0 - SMOOTHING_FACTOR) * previous_y)))

        mirrored_points[landmark_index] = (pixel_x, pixel_y)

    _previous_landmarks.clear()
    _previous_landmarks.update(mirrored_points)

    return mirrored_points
