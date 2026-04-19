import os
import time

import cv2
import numpy as np

from avatar import draw_avatar, reset_avatar_motion
from config import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    DEFAULT_CAMERA_INDICES,
    FLIP_CAMERA_VIEW,
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
from mirror import mirror_landmarks, reset_mirror_smoothing
from pose_detector import PoseDetector


def get_camera_indices():
    camera_index = os.getenv("MIRROR_TRACKER_CAMERA_INDEX")
    if camera_index is None:
        return DEFAULT_CAMERA_INDICES

    try:
        return (int(camera_index),)
    except ValueError as exc:
        raise RuntimeError(
            "MIRROR_TRACKER_CAMERA_INDEX must be an integer, "
            f"got {camera_index!r}."
        ) from exc


def open_webcam():
    for camera_index in get_camera_indices():
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"Using camera index {camera_index}")
            return cap
        cap.release()

    raise RuntimeError(
        "Unable to open webcam. On macOS, grant camera access to your terminal "
        "app in System Settings > Privacy & Security > Camera, then fully quit "
        "and reopen the terminal before running again. If Continuity Camera is "
        "taking over, try MIRROR_TRACKER_CAMERA_INDEX=1 python main.py or turn "
        "off Continuity Camera on your iPhone."
    )


def create_idle_pose_canvas():
    center_x = CANVAS_WIDTH // 2
    idle_landmarks = {
        NOSE: (center_x, 85),
        LEFT_SHOULDER: (center_x - 55, 155),
        RIGHT_SHOULDER: (center_x + 55, 155),
        LEFT_ELBOW: (center_x - 75, 220),
        RIGHT_ELBOW: (center_x + 75, 220),
        LEFT_WRIST: (center_x - 70, 290),
        RIGHT_WRIST: (center_x + 70, 290),
        LEFT_HIP: (center_x - 35, 270),
        RIGHT_HIP: (center_x + 35, 270),
        LEFT_KNEE: (center_x - 30, 365),
        RIGHT_KNEE: (center_x + 30, 365),
        LEFT_ANKLE: (center_x - 25, 450),
        RIGHT_ANKLE: (center_x + 25, 450),
    }
    return draw_avatar(idle_landmarks)


def main():
    cap = open_webcam()
    pose_detector = None
    previous_time = time.perf_counter()

    try:
        pose_detector = PoseDetector()

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.resize(frame, (CANVAS_WIDTH, CANVAS_HEIGHT))
            landmarks = pose_detector.detect(frame)
            display_frame = cv2.flip(frame, 1) if FLIP_CAMERA_VIEW else frame.copy()

            if landmarks is not None:
                mapped_landmarks = mirror_landmarks(landmarks)
                if mapped_landmarks:
                    avatar_canvas = draw_avatar(mapped_landmarks)
                else:
                    reset_mirror_smoothing()
                    reset_avatar_motion()
                    avatar_canvas = create_idle_pose_canvas()
            else:
                reset_mirror_smoothing()
                reset_avatar_motion()
                avatar_canvas = create_idle_pose_canvas()

            current_time = time.perf_counter()
            fps = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time

            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            combined_view = np.hstack((display_frame, avatar_canvas))
            cv2.imshow("Mirror Tracker", combined_view)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if pose_detector is not None:
            pose_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
