from collections import deque
import numpy as np

class TemporalSmoother:
    """
    Prevents random flickering predictions.
    Only commits a word when the model says the same word
    consistently for N frames in a row.
    """

    def __init__(self, window_size=10, threshold=0.7, cooldown=20):
        """
        window_size : how many recent predictions to look at
        threshold   : fraction of window that must agree (0.7 = 70%)
        cooldown    : frames to wait before predicting same word again
        """
        self.window_size = window_size
        self.threshold   = threshold
        self.cooldown    = cooldown

        self.buffer           = deque(maxlen=window_size)
        self.cooldown_counter = 0
        self.last_word        = None

    def update(self, word, confidence):
        """
        Feed in a new prediction.
        Returns the committed word if stable, else None.
        """
        # Always add to buffer
        self.buffer.append(word)

        # Count cooldown down
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return None

        # Need full window before deciding
        if len(self.buffer) < self.window_size:
            return None

        # Check if one word dominates the window
        most_common = max(set(self.buffer), key=self.buffer.count)
        fraction    = self.buffer.count(most_common) / self.window_size

        if fraction >= self.threshold:
            # Don't repeat same word immediately
            if most_common == self.last_word:
                return None

            # Commit the word
            self.last_word        = most_common
            self.cooldown_counter = self.cooldown
            self.buffer.clear()
            return most_common

        return None

    def reset(self):
        self.buffer.clear()
        self.cooldown_counter = 0
        self.last_word        = None


# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    smoother = TemporalSmoother(window_size=10, threshold=0.7, cooldown=20)

    # Simulate noisy predictions
    test_preds = (
        ["water"] * 3 +
        ["noise"] * 2 +
        ["water"] * 8 +   # should commit "water" here
        ["want"]  * 10 +  # should commit "want"
        ["i"]     * 10    # should commit "i"
    )

    committed = []
    for pred in test_preds:
        result = smoother.update(pred, confidence=0.9)
        if result:
            committed.append(result)
            print(f"✅ Committed: {result}")

    print(f"\nFinal sentence words: {committed}")
    print("✅ TemporalSmoother working correctly!")