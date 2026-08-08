from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from clipai.media import FfmpegAudioExtractor


def test_ffmpeg_normalizes_audio_to_mono_16khz(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    destination = tmp_path / "normalized.wav"

    with patch("clipai.media.subprocess.run", return_value=CompletedProcess([], 0)) as run:
        FfmpegAudioExtractor().extract(source, destination)

    command = run.call_args.args[0]
    assert command[:4] == ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    assert ["-ac", "1"] == command[command.index("-ac") : command.index("-ac") + 2]
    assert ["-ar", "16000"] == command[command.index("-ar") : command.index("-ar") + 2]


def test_ffmpeg_missing_binary_has_clear_error(tmp_path: Path) -> None:
    with (
        patch("clipai.media.subprocess.run", side_effect=FileNotFoundError),
        pytest.raises(RuntimeError, match="FFmpeg is not installed"),
    ):
        FfmpegAudioExtractor().extract(tmp_path / "source.mp4", tmp_path / "audio.wav")
