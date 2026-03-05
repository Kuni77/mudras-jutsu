import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    HandLandmarksConnections,
    RunningMode,
)
import cv2
import os


class HandTracker:
    def __init__(self, max_hands=2, detection_confidence=0.7, tracking_confidence=0.6):
        model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

        self._latest_result = None

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.LIVE_STREAM,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            result_callback=self._result_callback,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def _result_callback(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self._latest_result = result

    def process(self, frame):
        """Process a BGR frame and return hand landmarks."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 33  # ~30fps
        self.landmarker.detect_async(mp_image, self._timestamp_ms)

        if self._latest_result and self._latest_result.hand_landmarks:
            return self._latest_result.hand_landmarks, self._latest_result.handedness
        return None, None

    def draw_landmarks(self, frame, hand_landmarks_list):
        """Draw hand landmarks on the frame."""
        if not hand_landmarks_list:
            return frame

        h, w = frame.shape[:2]
        connections = HandLandmarksConnections.HAND_CONNECTIONS

        for hand_landmarks in hand_landmarks_list:
            # Draw connections
            for connection in connections:
                start = hand_landmarks[connection.start]
                end = hand_landmarks[connection.end]
                start_px = (int(start.x * w), int(start.y * h))
                end_px = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_px, end_px, (0, 255, 0), 2)

            # Draw landmarks
            for lm in hand_landmarks:
                px = (int(lm.x * w), int(lm.y * h))
                cv2.circle(frame, px, 4, (0, 0, 255), -1)

        return frame

    def release(self):
        self.landmarker.close()
