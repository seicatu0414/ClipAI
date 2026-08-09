from clipai.domain import TranscriptSegment
from clipai.events.detectors import (
    JapaneseTranscriptRuleDetector,
    LoudReactionDetector,
    SilenceDetector,
)
from clipai.events.domain import AudioFeature, EventType


def test_transcript_rules_cover_initial_non_audio_event_types() -> None:
    segments = [
        TranscriptSegment(
            0,
            0,
            5,
            "うわ、勝った！みんなコメントありがとう。前にもあったね（笑）歌います。嬉しい。絶対覚えて。",
        )
    ]

    detected = {event.event_type for event in JapaneseTranscriptRuleDetector().detect(segments)}

    assert detected == {
        EventType.LAUGHTER,
        EventType.LOUD_REACTION,
        EventType.SINGING,
        EventType.EMOTIONAL_VOICE,
        EventType.VICTORY_DEFEAT,
        EventType.MEMORABLE_STATEMENT,
        EventType.VIEWER_RESPONSE,
        EventType.CALLBACK_CONTRADICTION,
    }


def test_loudness_threshold_is_inclusive() -> None:
    features = [
        AudioFeature(0, 1, -40),
        AudioFeature(1, 2, -20),
        AudioFeature(2, 3, -30),
    ]

    events = list(LoudReactionDetector(10).detect(features))

    assert [(event.start_seconds, event.end_seconds) for event in events] == [(1, 2)]


def test_silence_at_minimum_duration_is_detected() -> None:
    features = [
        AudioFeature(0, 1, -60),
        AudioFeature(1, 2, -60),
        AudioFeature(2, 3, -20),
    ]

    events = list(SilenceDetector(-48, 2).detect(features))

    assert len(events) == 1
    assert events[0].event_type is EventType.UNUSUAL_SILENCE
    assert events[0].start_seconds == 0
    assert events[0].end_seconds == 2
