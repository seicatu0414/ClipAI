import math
import sys
import wave
from array import array
from collections.abc import Iterator
from pathlib import Path

from clipai.events.domain import AudioFeature


class WaveAudioFeatureExtractor:
    def __init__(self, window_seconds: float = 1.0) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._window_seconds = window_seconds

    def extract(self, audio_path: Path) -> Iterator[AudioFeature]:
        with wave.open(str(audio_path), "rb") as audio:
            if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
                raise ValueError("event detection requires mono 16-bit PCM audio")
            frame_rate = audio.getframerate()
            frames_per_window = max(1, round(frame_rate * self._window_seconds))
            start_frame = 0
            while raw := audio.readframes(frames_per_window):
                samples = array("h")
                samples.frombytes(raw)
                if sys.byteorder != "little":
                    samples.byteswap()
                if not samples:
                    break
                mean_square = sum(sample * sample for sample in samples) / len(samples)
                rms = math.sqrt(mean_square)
                dbfs = -96.0 if rms == 0 else 20 * math.log10(rms / 32768.0)
                end_frame = start_frame + len(samples)
                yield AudioFeature(
                    start_seconds=start_frame / frame_rate,
                    end_seconds=end_frame / frame_rate,
                    rms_dbfs=dbfs,
                )
                start_frame = end_frame
