import math


# MediaPipe hand landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


def _landmark_to_point(landmark):
    return (landmark.x, landmark.y, landmark.z)


def _distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def _is_finger_extended(landmarks, tip, dip, pip, mcp):
    """Check if a finger is extended by comparing tip-to-mcp vs pip-to-mcp distance."""
    tip_pt = _landmark_to_point(landmarks[tip])
    pip_pt = _landmark_to_point(landmarks[pip])
    mcp_pt = _landmark_to_point(landmarks[mcp])

    tip_to_mcp = _distance(tip_pt, mcp_pt)
    pip_to_mcp = _distance(pip_pt, mcp_pt)

    return tip_to_mcp > pip_to_mcp * 1.1


def _is_thumb_extended(landmarks):
    """Check if thumb is extended."""
    thumb_tip = _landmark_to_point(landmarks[THUMB_TIP])
    thumb_ip = _landmark_to_point(landmarks[THUMB_IP])
    thumb_mcp = _landmark_to_point(landmarks[THUMB_MCP])

    tip_to_mcp = _distance(thumb_tip, thumb_mcp)
    ip_to_mcp = _distance(thumb_ip, thumb_mcp)

    return tip_to_mcp > ip_to_mcp * 1.2


def _get_finger_states(landmarks):
    """Return dict of which fingers are extended."""
    lm = landmarks
    return {
        "thumb": _is_thumb_extended(lm),
        "index": _is_finger_extended(lm, INDEX_TIP, INDEX_DIP, INDEX_PIP, INDEX_MCP),
        "middle": _is_finger_extended(lm, MIDDLE_TIP, MIDDLE_DIP, MIDDLE_PIP, MIDDLE_MCP),
        "ring": _is_finger_extended(lm, RING_TIP, RING_DIP, RING_PIP, RING_MCP),
        "pinky": _is_finger_extended(lm, PINKY_TIP, PINKY_DIP, PINKY_PIP, PINKY_MCP),
    }


class MudraClassifier:
    """Classifies hand poses into Naruto mudras (simplified signs)."""

    MUDRA_NAMES = {
        "fist": "Poing",
        "open": "Main ouverte",
        "peace": "Peace (V)",
    }

    def classify(self, hand_landmarks_list, handedness_list=None):
        """Classify the current hand pose. Uses first detected hand."""
        if not hand_landmarks_list:
            return None

        # Use the first hand detected
        landmarks = hand_landmarks_list[0]
        fingers = _get_finger_states(landmarks)
        extended = sum(1 for k, v in fingers.items() if v and k != "thumb")

        # Poing: no fingers extended
        if extended == 0:
            return "fist"

        # Main ouverte: all 4 fingers extended
        if extended >= 4:
            return "open"

        # Peace (V): index + middle extended, ring + pinky folded
        if (fingers["index"] and fingers["middle"]
                and not fingers["ring"] and not fingers["pinky"]):
            return "peace"

        return None

    def get_display_name(self, mudra_key):
        return self.MUDRA_NAMES.get(mudra_key, mudra_key)
