import cv2
import time
from hand_tracker import HandTracker
from mudra_classifier import MudraClassifier
from jutsu_engine import JutsuEngine


def draw_hud(frame, mudra, classifier, engine, fps):
    """Draw the heads-up display overlay."""
    h, w = frame.shape[:2]

    # Background bar at top
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Current mudra detected
    if mudra:
        mudra_name = classifier.get_display_name(mudra)
        cv2.putText(frame, f"Mudra: {mudra_name}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Sequence progress
    step, total, jutsu_name = engine.get_progress()
    if total > 0:
        progress_text = f"Sequence: {step}/{total}"
        cv2.putText(frame, progress_text, (w - 250, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        # Progress bar
        bar_w = 200
        bar_x = w - 250
        bar_y = 35
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 15), (50, 50, 50), -1)
        fill_w = int(bar_w * step / total)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + 15), (0, 165, 255), -1)

    # Instructions at bottom
    cv2.putText(frame, "Bunshin no Jutsu: Poing -> Main ouverte -> Peace (V)", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "Press 'q' to quit", (w - 180, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


def apply_bunshin_effect(frame):
    """Create the clone mirror effect - show original and flipped side by side."""
    h, w = frame.shape[:2]

    # Resize each half to fit side by side
    half_w = w // 2
    left = cv2.resize(frame, (half_w, h))
    right = cv2.flip(left, 1)  # Mirror flip

    # Combine side by side
    combined = cv2.hconcat([left, right])

    # Add jutsu activation text
    cv2.rectangle(combined, (0, h // 2 - 40), (w, h // 2 + 40), (0, 0, 0), -1)
    text = "BUNSHIN NO JUTSU!"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(combined, text, (text_x, h // 2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)

    return combined


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur: impossible d'ouvrir la webcam!")
        return

    tracker = HandTracker()
    classifier = MudraClassifier()
    engine = JutsuEngine()

    prev_time = time.time()
    fps = 0

    print("Mudra Recognition - Bunshin no Jutsu")
    print("=====================================")
    print("Sequence: Poing -> Main ouverte -> Peace (V)")
    print("Maintenez chaque mudra ~0.5 seconde")
    print("Appuyez sur 'q' pour quitter")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for natural mirror view
        frame = cv2.flip(frame, 1)

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # Detect hands
        hand_landmarks, handedness = tracker.process(frame)

        # Classify mudra
        mudra = classifier.classify(hand_landmarks, handedness)

        # Update jutsu engine
        triggered = engine.update(mudra)
        if triggered:
            print(f"JUTSU ACTIVE: {triggered.display_name}")

        # Draw hand landmarks
        tracker.draw_landmarks(frame, hand_landmarks)

        # Check if jutsu effect should be shown
        active_jutsu = engine.get_active_jutsu()
        if active_jutsu and active_jutsu.name == "bunshin":
            frame = apply_bunshin_effect(frame)
        else:
            frame = draw_hud(frame, mudra, classifier, engine, fps)

        cv2.imshow("Mudra Recognition - Naruto Jutsu", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tracker.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
