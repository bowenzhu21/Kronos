import cv2
import mediapipe as mp


class PoseDetector:
    def __init__(self):
        pose_module = self._get_pose_module()
        self.pose = pose_module.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def _get_pose_module():
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            return mp.solutions.pose

        raise ImportError(
            "The installed mediapipe package does not expose the legacy "
            "'mp.solutions.pose' API required by this project. "
            "Use a Python environment with a full MediaPipe Solutions wheel "
            "(for example Python 3.11 on this machine) and reinstall mediapipe."
        )

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None

        return [
            (landmark.x, landmark.y, landmark.visibility)
            for landmark in results.pose_landmarks.landmark
        ]

    def close(self):
        self.pose.close()
