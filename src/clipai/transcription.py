from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from clipai.domain import TranscriptMetadata, TranscriptSegment


@dataclass(frozen=True)
class TranscriptionResult:
    metadata: TranscriptMetadata
    segments: Iterable[TranscriptSegment]


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path, *, language: str) -> TranscriptionResult: ...


def resolve_device(requested: str, *, cpu_fallback: bool) -> tuple[str, str]:
    from ctranslate2 import get_cuda_device_count

    cuda_available = get_cuda_device_count() > 0
    if requested == "cpu":
        return "cpu", "int8"
    if requested == "cuda" and not cuda_available:
        if not cpu_fallback:
            raise RuntimeError("CUDA was requested but no compatible NVIDIA GPU is available")
        return "cpu", "int8"
    if requested == "cuda" or cuda_available:
        return "cuda", "float16"
    if cpu_fallback:
        return "cpu", "int8"
    raise RuntimeError("No NVIDIA GPU is available and CPU fallback is disabled")


class FasterWhisperTranscriber:
    def __init__(self, model_size: str, *, device: str, cpu_fallback: bool) -> None:
        from faster_whisper import WhisperModel

        resolved_device, compute_type = resolve_device(device, cpu_fallback=cpu_fallback)
        self.device = resolved_device
        self.model_size = model_size
        self._model = WhisperModel(model_size, device=resolved_device, compute_type=compute_type)

    def transcribe(self, audio_path: Path, *, language: str) -> TranscriptionResult:
        raw_segments, info = self._model.transcribe(
            str(audio_path),
            language=language,
            vad_filter=True,
            beam_size=5,
        )
        metadata = TranscriptMetadata(
            duration_seconds=self._optional_float(info, "duration"),
            detected_language=getattr(info, "language", None),
            language_probability=self._optional_float(info, "language_probability"),
            model_size=self.model_size,
            device=self.device,
        )

        def mapped_segments() -> Iterable[TranscriptSegment]:
            for index, segment in enumerate(raw_segments):
                text = str(segment.text).strip()
                if not text:
                    continue
                yield TranscriptSegment(
                    index=index,
                    start_seconds=float(segment.start),
                    end_seconds=float(segment.end),
                    text=text,
                    average_log_probability=self._optional_float(segment, "avg_logprob"),
                    no_speech_probability=self._optional_float(segment, "no_speech_prob"),
                )

        return TranscriptionResult(metadata=metadata, segments=mapped_segments())

    @staticmethod
    def _optional_float(value: Any, attribute: str) -> float | None:
        result = getattr(value, attribute, None)
        return None if result is None else float(result)
