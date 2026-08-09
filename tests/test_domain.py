from pathlib import Path

import pytest

from clipai.domain import SourceKind, SourceSpec


def test_accepts_youtube_video_url() -> None:
    source = SourceSpec.parse("https://www.youtube.com/watch?v=abc123")

    assert source.kind is SourceKind.YOUTUBE


def test_rejects_non_youtube_url() -> None:
    with pytest.raises(ValueError, match="only YouTube URLs"):
        SourceSpec.parse("https://example.com/video")


def test_accepts_existing_local_file(tmp_path: Path) -> None:
    video = tmp_path / "stream.mp4"
    video.write_bytes(b"video")

    source = SourceSpec.parse(str(video))

    assert source.kind is SourceKind.LOCAL_FILE
    assert source.value == str(video)


def test_rejects_missing_local_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        SourceSpec.parse(str(tmp_path / "missing.mp4"))
