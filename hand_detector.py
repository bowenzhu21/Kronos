import cv2
import mediapipe as mp
import numpy as np

from config import CANVAS_HEIGHT, CANVAS_WIDTH


class HandDetector:
    def __init__(self):
        hands_module = self._get_hands_module()
        self.detector = hands_module.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._previous_closed = {}
        self._closed_threshold = 1.55
        self._open_threshold = 1.9
        self._process_width = max(320, CANVAS_WIDTH // 2)
        self._process_height = max(180, CANVAS_HEIGHT // 2)
        self._process_interval = 2
        self._frame_index = 0
        self._cached_hands = None

    @staticmethod
    def _get_hands_module():
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            return mp.solutions.hands

        raise ImportError(
            "The installed mediapipe package does not expose the legacy "
            "'mp.solutions.hands' API required by this project. "
            "Use a Python environment with a full MediaPipe Solutions wheel "
            "and reinstall mediapipe."
        )

    def _is_closed_fist(self, landmarks, label):
        points = np.array([(landmark.x, landmark.y) for landmark in landmarks], dtype=np.float32)
        wrist = points[0]
        palm_anchors = points[[5, 9, 13, 17]]
        palm_size = float(np.mean(np.hypot(palm_anchors[:, 0] - wrist[0], palm_anchors[:, 1] - wrist[1])))
        palm_size = max(palm_size, 1e-6)

        tip_points = points[[8, 12, 16, 20]]
        extension = float(np.mean(np.hypot(tip_points[:, 0] - wrist[0], tip_points[:, 1] - wrist[1]) / palm_size))
        thumb_delta = points[4] - points[2]
        thumb_extension = float(np.hypot(thumb_delta[0], thumb_delta[1]) / palm_size)

        openness = extension + (thumb_extension * 0.35)
        previous_closed = self._previous_closed.get(label, False)
        threshold = self._open_threshold if previous_closed else self._closed_threshold
        return openness < threshold, openness

    def get_hands(self, frame):
        self._frame_index += 1
        if self._cached_hands is not None and (self._frame_index % self._process_interval) != 0:
            return self._cached_hands

        resized_frame = cv2.resize(
            frame,
            (self._process_width, self._process_height),
            interpolation=cv2.INTER_AREA,
        )
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)

        if not results.multi_hand_landmarks or not results.multi_handedness:
            self._previous_closed.clear()
            self._cached_hands = []
            return []

        hands = []
        seen_labels = set()

        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            seen_labels.add(label)

            landmarks = hand_landmarks.landmark
            palm_points = np.array(
                [(landmarks[index].x, landmarks[index].y) for index in (0, 5, 9, 13, 17)],
                dtype=np.float32,
            )
            center_x = int(round(float(np.mean(palm_points[:, 0])) * CANVAS_WIDTH))
            center_y = int(round(float(np.mean(palm_points[:, 1])) * CANVAS_HEIGHT))
            center_x = int(np.clip(center_x, 0, CANVAS_WIDTH - 1))
            center_y = int(np.clip(center_y, 0, CANVAS_HEIGHT - 1))

            is_closed, openness = self._is_closed_fist(landmarks, label)
            previous_closed = self._previous_closed.get(label, False)

            hands.append(
                {
                    "label": label,
                    "x": center_x,
                    "y": center_y,
                    "is_closed": is_closed,
                    "just_opened": previous_closed and not is_closed,
                    "just_closed": (not previous_closed) and is_closed,
                    "openness": openness,
                }
            )

            self._previous_closed[label] = is_closed

        for label in list(self._previous_closed):
            if label not in seen_labels:
                del self._previous_closed[label]

        self._cached_hands = [
            {
                **hand,
                "just_opened": False,
                "just_closed": False,
            }
            for hand in hands
        ]
        return hands

    def close(self):
        self.detector.close()
