import math
import wave
from pathlib import Path

from clipai.events.features import WaveAudioFeatureExtractor


def test_extracts_one_second_rms_windows(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    sample_rate = 16_000
    samples = (1000).to_bytes(2, "little", signed=True) * (sample_rate * 2)
    with wave.open(str(audio_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(samples)

    features = list(WaveAudioFeatureExtractor().extract(audio_path))

    assert len(features) == 2
    assert features[0].start_seconds == 0
    assert features[-1].end_seconds == 2
    assert math.isclose(features[0].rms_dbfs, -30.31, abs_tol=0.1)
