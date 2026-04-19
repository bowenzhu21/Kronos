import cv2
import mediapipe as mp

from config import CANVAS_HEIGHT, CANVAS_WIDTH


class SilhouetteDetector:
    def __init__(self):
        selfie_segmentation_module = self._get_selfie_segmentation_module()
        self.segmenter = selfie_segmentation_module.SelfieSegmentation(model_selection=1)

    @staticmethod
    def _get_selfie_segmentation_module():
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "selfie_segmentation"):
            return mp.solutions.selfie_segmentation

        raise ImportError(
            "The installed mediapipe package does not expose the legacy "
            "'mp.solutions.selfie_segmentation' API required by this project. "
            "Use a Python environment with a full MediaPipe Solutions wheel "
            "and reinstall mediapipe."
        )

    def get_mask(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.segmenter.process(rgb_frame)

        if results.segmentation_mask is None:
            mask = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                (CANVAS_WIDTH, CANVAS_HEIGHT),
                interpolation=cv2.INTER_NEAREST,
            )
            mask[:] = 0
            return mask

        segmentation_mask = results.segmentation_mask.astype("float32")
        segmentation_mask = cv2.GaussianBlur(segmentation_mask, (11, 11), 0)
        _, binary_mask = cv2.threshold(segmentation_mask, 0.5, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype("uint8")

        return cv2.resize(binary_mask, (CANVAS_WIDTH, CANVAS_HEIGHT), interpolation=cv2.INTER_NEAREST)

    def close(self):
        self.segmenter.close()
