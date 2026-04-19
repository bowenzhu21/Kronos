import os
import time

import cv2
import numpy as np

from config import CANVAS_HEIGHT, CANVAS_WIDTH, SHOW_WEBCAM_UNDERLAY, UNDERLAY_ALPHA
from hand_detector import HandDetector
from particle_system import ParticleSystem
from silhouette import SilhouetteDetector

DEFAULT_CAMERA_INDICES = (0, 1, 2)
BACKGROUND_COLOR = (15, 14, 12)
FPS_COLOR = (80, 80, 75)
WINDOW_NAME = "Dust"


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
        "and reopen the terminal before running again."
    )


def main():
    cap = open_webcam()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CANVAS_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CANVAS_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    hand_detector = None
    silhouette_detector = None
    particle_system = None
    previous_time = time.perf_counter()
    canvas = np.empty((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        hand_detector = HandDetector()
        silhouette_detector = SilhouetteDetector()
        particle_system = ParticleSystem()

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            if frame.shape[1] != CANVAS_WIDTH or frame.shape[0] != CANVAS_HEIGHT:
                frame = cv2.resize(frame, (CANVAS_WIDTH, CANVAS_HEIGHT), interpolation=cv2.INTER_LINEAR)
            mask = silhouette_detector.get_mask(frame)
            hands = hand_detector.get_hands(frame)
            particle_system.update(mask, hands=hands)

            canvas[:] = BACKGROUND_COLOR

            if SHOW_WEBCAM_UNDERLAY:
                cv2.addWeighted(
                    frame,
                    UNDERLAY_ALPHA,
                    canvas,
                    1.0 - UNDERLAY_ALPHA,
                    0.0,
                    dst=canvas,
                )

            particle_system.draw(canvas)

            current_time = time.perf_counter()
            fps = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time

            cv2.putText(
                canvas,
                f"FPS: {fps:.1f}",
                (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                FPS_COLOR,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, canvas)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if hand_detector is not None:
            hand_detector.close()
        if silhouette_detector is not None:
            silhouette_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
