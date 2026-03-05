import time


class Jutsu:
    def __init__(self, name, sequence, display_name=None):
        self.name = name
        self.sequence = sequence  # List of mudra keys in order
        self.display_name = display_name or name


# Bunshin no Jutsu: Poing -> Main ouverte -> Peace
BUNSHIN_NO_JUTSU = Jutsu(
    name="bunshin",
    sequence=["fist", "open", "peace"],
    display_name="Bunshin no Jutsu!",
)

ALL_JUTSUS = [BUNSHIN_NO_JUTSU]


class JutsuEngine:
    def __init__(self, timeout=3.0, hold_time=0.5, grace_period=0.3):
        """
        timeout: max seconds between mudras in a sequence
        hold_time: seconds a mudra must be held to register
        grace_period: seconds to tolerate None detections without resetting hold
        """
        self.timeout = timeout
        self.hold_time = hold_time
        self.grace_period = grace_period
        self.jutsus = ALL_JUTSUS

        # Tracking state
        self._current_sequence = []
        self._last_mudra_time = 0
        self._current_hold_mudra = None
        self._hold_start_time = 0
        self._last_detected_time = 0
        self._triggered_jutsu = None
        self._triggered_time = 0

    @property
    def triggered_jutsu(self):
        return self._triggered_jutsu

    @property
    def current_sequence(self):
        return list(self._current_sequence)

    def update(self, detected_mudra):
        """
        Update with the currently detected mudra.
        Returns a Jutsu if one was triggered, None otherwise.
        """
        now = time.time()

        # Check if sequence timed out
        if self._current_sequence and (now - self._last_mudra_time > self.timeout):
            self._reset()

        # Handle hold detection — tolerate brief None frames
        if detected_mudra is None:
            if self._current_hold_mudra and (now - self._last_detected_time < self.grace_period):
                return None  # Keep holding, just a flicker
            self._current_hold_mudra = None
            return None

        self._last_detected_time = now

        if detected_mudra != self._current_hold_mudra:
            self._current_hold_mudra = detected_mudra
            self._hold_start_time = now
            return None

        # Check if held long enough
        if now - self._hold_start_time < self.hold_time:
            return None

        # Mudra confirmed - check if it's a new one (not the last in sequence)
        if self._current_sequence and self._current_sequence[-1] == detected_mudra:
            return None  # Already registered this mudra

        # Add to sequence
        self._current_sequence.append(detected_mudra)
        self._last_mudra_time = now

        # Check if any jutsu matches
        for jutsu in self.jutsus:
            if self._current_sequence == jutsu.sequence:
                self._triggered_jutsu = jutsu
                self._triggered_time = now
                self._current_sequence = []
                return jutsu

        # Check if sequence is still a valid prefix
        is_prefix = False
        for jutsu in self.jutsus:
            seq = jutsu.sequence
            if seq[:len(self._current_sequence)] == self._current_sequence:
                is_prefix = True
                break

        if not is_prefix:
            self._reset()

        return None

    def get_progress(self):
        """Return (current_step, total_steps, jutsu_name) for the best matching jutsu."""
        if not self._current_sequence:
            return 0, 0, None

        for jutsu in self.jutsus:
            seq = jutsu.sequence
            if seq[:len(self._current_sequence)] == self._current_sequence:
                return len(self._current_sequence), len(seq), jutsu.display_name

        return 0, 0, None

    def is_jutsu_active(self, duration=3.0):
        """Check if a jutsu effect should still be displayed."""
        if self._triggered_jutsu is None:
            return False
        return time.time() - self._triggered_time < duration

    def get_active_jutsu(self):
        """Return the currently active jutsu or None."""
        if self.is_jutsu_active():
            return self._triggered_jutsu
        if self._triggered_jutsu and not self.is_jutsu_active():
            self._triggered_jutsu = None
        return None

    def _reset(self):
        self._current_sequence = []
        self._current_hold_mudra = None
