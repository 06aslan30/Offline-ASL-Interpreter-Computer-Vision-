import subprocess
import sys
import os

# Try to import kokoro
try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False

class TTS:
    """
    Text-to-speech using Kokoro (offline, no internet needed).
    Falls back to Windows built-in TTS if Kokoro not available.
    """

    def __init__(self, voice="af_heart", speed=1.0):
        self.voice  = voice
        self.speed  = speed
        self.engine = None

        if KOKORO_AVAILABLE:
            try:
                self.engine = KPipeline(lang_code="a")  # "a" = American English
                print("✅ Kokoro TTS loaded!")
            except Exception as e:
                print(f"⚠️  Kokoro failed to load: {e}")
                self.engine = None
        else:
            print("⚠️  Kokoro not installed — using Windows TTS fallback")

    def speak(self, text):
        """Speak the given text aloud."""
        if not text:
            return

        print(f"🔊 Speaking: {text}")

        if self.engine is not None:
            self._speak_kokoro(text)
        else:
            self._speak_windows(text)

    def _speak_kokoro(self, text):
        try:
            import sounddevice as sd
            import numpy as np

            generator = self.engine(
                text,
                voice=self.voice,
                speed=self.speed
            )

            for _, _, audio in generator:
                if audio is not None and len(audio) > 0:
                    sd.play(audio, samplerate=24000)
                    sd.wait()

        except Exception as e:
            print(f"⚠️  Kokoro speak error: {e}")
            self._speak_windows(text)

    def _speak_windows(self, text):
        """Windows built-in TTS fallback using PowerShell."""
        try:
            clean = text.replace("'", "").replace('"', "")
            cmd = (
                f"Add-Type -AssemblyName System.Speech; "
                f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.Rate = 1; "
                f"$s.Speak('{clean}')"
            )
            subprocess.run(
                ["powershell", "-Command", cmd],
                capture_output=True
            )
        except Exception as e:
            print(f"⚠️  Windows TTS error: {e}")


# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tts = TTS()

    test_sentences = [
        "I want water.",
        "Help please.",
        "I need help.",
        "Thank you.",
    ]

    print("\nTesting TTS...\n")
    for sentence in test_sentences:
        tts.speak(sentence)
        import time
        time.sleep(0.5)

    print("\n✅ TTS test done!")