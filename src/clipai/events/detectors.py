from collections.abc import Iterable, Sequence
from statistics import median
from typing import Protocol

from clipai.domain import TranscriptSegment
from clipai.events.domain import AudioFeature, DetectedEvent, EventType


class AudioEventDetector(Protocol):
    def detect(self, features: Sequence[AudioFeature]) -> Iterable[DetectedEvent]: ...


class TranscriptEventDetector(Protocol):
    def detect(self, segments: Iterable[TranscriptSegment]) -> Iterable[DetectedEvent]: ...


class LoudReactionDetector:
    def __init__(self, loudness_delta_db: float) -> None:
        self._loudness_delta_db = loudness_delta_db

    def detect(self, features: Sequence[AudioFeature]) -> Iterable[DetectedEvent]:
        if not features:
            return []
        audible = [feature.rms_dbfs for feature in features if feature.rms_dbfs > -60]
        if not audible:
            return []
        baseline = median(audible)
        events: list[DetectedEvent] = []
        for feature in features:
            delta = feature.rms_dbfs - baseline
            if delta < self._loudness_delta_db:
                continue
            confidence = min(0.95, 0.6 + (delta - self._loudness_delta_db) / 30)
            events.append(
                DetectedEvent(
                    event_type=EventType.LOUD_REACTION,
                    start_seconds=feature.start_seconds,
                    end_seconds=feature.end_seconds,
                    confidence=confidence,
                    source_signals={
                        "rms_dbfs": round(feature.rms_dbfs, 2),
                        "baseline_dbfs": round(baseline, 2),
                        "delta_db": round(delta, 2),
                    },
                    explanation="Audio loudness rose substantially above the stream baseline.",
                )
            )
        return events


class SilenceDetector:
    def __init__(self, silence_db: float, minimum_seconds: float) -> None:
        self._silence_db = silence_db
        self._minimum_seconds = minimum_seconds

    def detect(self, features: Sequence[AudioFeature]) -> Iterable[DetectedEvent]:
        events: list[DetectedEvent] = []
        run: list[AudioFeature] = []
        for feature in [*features, None]:
            if feature is not None and feature.rms_dbfs <= self._silence_db:
                run.append(feature)
                continue
            if run:
                duration = run[-1].end_seconds - run[0].start_seconds
                if duration >= self._minimum_seconds:
                    quietest = min(item.rms_dbfs for item in run)
                    confidence = min(0.95, 0.6 + (duration - self._minimum_seconds) / 20)
                    events.append(
                        DetectedEvent(
                            event_type=EventType.UNUSUAL_SILENCE,
                            start_seconds=run[0].start_seconds,
                            end_seconds=run[-1].end_seconds,
                            confidence=confidence,
                            source_signals={
                                "minimum_rms_dbfs": round(quietest, 2),
                                "duration_seconds": round(duration, 2),
                            },
                            explanation="Audio stayed below the configured silence threshold.",
                        )
                    )
                run = []
        return events


class JapaneseTranscriptRuleDetector:
    VERSION = "ja-rules-v1"
    _RULES: dict[EventType, tuple[str, ...]] = {
        EventType.LAUGHTER: ("（笑）", "(笑)", "ｗｗ", "笑った", "ウケる"),
        EventType.LOUD_REACTION: ("うわ", "えっ", "やば", "すごい", "まじで"),
        EventType.SINGING: ("歌います", "歌って", "歌う", "熱唱"),
        EventType.EMOTIONAL_VOICE: ("泣", "悲しい", "嬉しい", "感動", "つらい"),
        EventType.VICTORY_DEFEAT: ("勝った", "負けた", "クリア", "優勝", "全滅"),
        EventType.MEMORABLE_STATEMENT: ("絶対", "断言", "これだけは", "覚えて"),
        EventType.VIEWER_RESPONSE: ("コメント", "チャット", "みんな", "リスナー"),
        EventType.CALLBACK_CONTRADICTION: ("前にも", "またこれ", "いつもの", "さっきと違"),
    }

    def detect(self, segments: Iterable[TranscriptSegment]) -> Iterable[DetectedEvent]:
        for segment in segments:
            for event_type, keywords in self._RULES.items():
                matches = [keyword for keyword in keywords if keyword in segment.text]
                if not matches:
                    continue
                confidence = min(0.9, 0.62 + 0.08 * (len(matches) - 1))
                yield DetectedEvent(
                    event_type=event_type,
                    start_seconds=segment.start_seconds,
                    end_seconds=segment.end_seconds,
                    confidence=confidence,
                    source_signals={
                        "matched_keywords": matches,
                        "transcript_segment_index": segment.index,
                    },
                    explanation=(
                        f"Transcript matched {event_type.value} indicators: "
                        + ", ".join(matches)
                    ),
                )
