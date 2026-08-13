from uuid import UUID

from clipai.candidates.boundaries import EndBoundaryDetector
from clipai.candidates.domain import (
    CandidateEvent,
    CandidateWindow,
    EndBoundaryCandidate,
    TopicWindow,
)
from clipai.domain import TranscriptSegment
from clipai.events.domain import EventType
from clipai.knowledge.domain import KnowledgeObservation


class StableRanker:
    def select(self, window: CandidateWindow, topic: TopicWindow,
               candidates: tuple[EndBoundaryCandidate, ...], segments: list[TranscriptSegment],
               knowledge: tuple[KnowledgeObservation, ...]) -> tuple[str, float, str]:
        del window, segments, knowledge
        selected = min(
            candidates,
            key=lambda item: (abs(item.timestamp - topic.end_seconds), item.timestamp),
        )
        return selected.id, 0.84, "話題の完結に最も近く、次の話題を含まない"


def segment(index: int, start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(index, start, end, text)


def window(start: float = 90, event_start: float = 100, event_end: float = 104) -> CandidateWindow:
    event = CandidateEvent(
        UUID(int=1), EventType.LOUD_REACTION, event_start, event_end, 0.9, {}, "reaction"
    )
    return CandidateWindow(start, 116, (event,), 0.9)


def detect(segments: list[TranscriptSegment], candidate_window: CandidateWindow | None = None):
    return EndBoundaryDetector(StableRanker()).detect(
        candidate_window or window(), segments, (), minimum_seconds=15,
        maximum_seconds=120, candidate_count=5,
    )


def test_topic_ends_immediately_after_event() -> None:
    result = detect([segment(0, 90, 100, "このボスを倒すぞ"), segment(1, 100, 106, "やった！勝った！"),
                     segment(2, 109, 116, "はい次は装備を見ます")])
    assert result.topic_window.end_seconds == 109
    assert result.timestamp <= 109


def test_same_topic_can_continue_two_minutes_after_event() -> None:
    segments = [segment(0, 90, 100, "このボスの攻略を始める")]
    segments.extend(
        segment(index, 100 + index * 18, 116 + index * 18, "ボスの攻略と攻撃パターンの話")
        for index in range(1, 7)
    )
    segments.append(segment(8, 230, 238, "はい次は別のステージです"))
    result = detect(segments)
    assert result.topic_window.end_seconds >= 220
    assert 116 < result.timestamp <= 210


def test_temporary_silence_does_not_end_resumed_topic() -> None:
    result = detect([segment(0, 90, 100, "宝箱の話をする"), segment(1, 100, 108, "この宝箱すごい！"),
                     segment(2, 112, 120, "宝箱の中身もすごい"), segment(3, 122, 130, "宝箱はこれで終わりです。"),
                     segment(4, 134, 142, "はい次はボスです")])
    assert result.topic_window.end_seconds >= 134


def test_short_detour_returns_to_original_topic() -> None:
    result = detect([segment(0, 90, 100, "武器強化の素材を集める"), segment(1, 100, 108, "武器強化できた！"),
                     segment(2, 110, 118, "そういえば今日暑いね。"), segment(3, 120, 128, "武器強化の続きと性能を見る"),
                     segment(4, 132, 140, "はい次はボス戦です")])
    assert result.topic_window.end_seconds >= 132


def test_payoff_keeps_reaction_comment_and_ranks_multiple_choices() -> None:
    result = detect([segment(0, 90, 100, "謎解きの答えはこれだ"), segment(1, 100, 106, "開いた！やった！"),
                     segment(2, 106, 112, "いやこれは気持ちいいですね。"), segment(3, 115, 123, "はい次は別の部屋です")])
    assert result.timestamp >= 112
    assert len(result.candidates) >= 3
    assert result.reason and result.confidence == 0.84


def test_120_second_limit_and_repeated_input_are_stable() -> None:
    candidate_window = window(start=0, event_start=10, event_end=14)
    segments = [segment(index, index * 20, index * 20 + 18, "同じ長い話題を続けています") for index in range(8)]
    first = detect(segments, candidate_window)
    second = detect(segments, candidate_window)
    assert 15 <= first.timestamp <= 120
    assert first.timestamp == second.timestamp
    assert first.candidates == second.candidates


def test_sparse_context_still_generates_three_end_choices() -> None:
    result = detect(
        [
            segment(0, 90, 100, "話を始めます"),
            segment(1, 100, 110, "リアクションです"),
            segment(2, 110, 125, "感想を続けます"),
            segment(3, 125, 140, "補足を終えます"),
        ]
    )

    assert len(result.candidates) >= 3
