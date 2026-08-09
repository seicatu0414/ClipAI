import shutil
import subprocess
from pathlib import Path
from typing import Protocol
from uuid import UUID

from clipai.domain import SourceKind, SourceSpec


class MediaAcquirer(Protocol):
    def acquire(self, source: SourceSpec, work_directory: Path) -> Path: ...


class SourceMediaAcquirer:
    def acquire(self, source: SourceSpec, work_directory: Path) -> Path:
        if source.kind is SourceKind.LOCAL_FILE:
            path = Path(source.value).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"local video file does not exist: {path}")
            return path

        from yt_dlp import YoutubeDL

        output_template = str(work_directory / "source.%(ext)s")
        options = {
            "format": "bestaudio/best",
            "noplaylist": True,
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
        }
        with YoutubeDL(options) as downloader:
            info = downloader.extract_info(source.value, download=True)
            downloaded = Path(downloader.prepare_filename(info))
        if not downloaded.is_file():
            raise RuntimeError("YouTube media acquisition completed without an output file")
        return downloaded


class AudioExtractor(Protocol):
    def extract(self, source: Path, destination: Path) -> None: ...


class FfmpegAudioExtractor:
    def extract(self, source: Path, destination: Path) -> None:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as error:
            raise RuntimeError("FFmpeg is not installed or not available on PATH") from error
        except subprocess.CalledProcessError as error:
            detail = error.stderr.strip() or "unknown FFmpeg error"
            raise RuntimeError(f"FFmpeg audio extraction failed: {detail}") from error


def prepare_work_directory(media_root: Path, job_id: UUID) -> Path:
    directory = media_root / "work" / str(job_id)
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def remove_work_directory(directory: Path) -> None:
    shutil.rmtree(directory, ignore_errors=True)
