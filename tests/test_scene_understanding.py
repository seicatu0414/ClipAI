from uuid import UUID

from clipai.candidates.boundaries import EndBoundaryDetector, boundary_analysis
from clipai.candidates.domain import (
    CandidateEvent,
    CandidateWindow,
    EndBoundaryCandidate,
    EndBoundarySelection,
    ScenePhase,
    SceneWindow,
    TopicWindow,
)
from clipai.candidates.scenes import build_scene_timeline
from clipai.domain import TranscriptSegment
from clipai.events.domain import EventType
from clipai.knowledge.domain import KnowledgeObservation


class RecordingRanker:
    def __init__(self, normal_confidence: float = 0.84) -> None:
        self.normal_confidence = normal_confidence
        self.modes: list[bool] = []

    def select(
        self,
        window: CandidateWindow,
        topic: TopicWindow,
        scene: SceneWindow,
        candidates: tuple[EndBoundaryCandidate, ...],
        segments: list[TranscriptSegment],
        knowledge: tuple[KnowledgeObservation, ...],
        *,
        detailed: bool,
    ) -> tuple[str, float, str]:
        del window, topic, scene, segments, knowledge
        self.modes.append(detailed)
        resolved = [
            item for item in candidates if "open_thread_resolved" in item.source_signals
        ]
        aftermath = [
            item for item in candidates if "aftermath_completion" in item.source_signals
        ]
        transitions = [
            item for item in candidates if "scene_transition" in item.source_signals
        ]
        pool = aftermath or resolved or transitions or list(candidates)
        selected = max(pool, key=lambda item: (item.timestamp, item.confidence))
        confidence = 0.88 if detailed else self.normal_confidence
        return selected.id, confidence, "Sceneの因果と余韻が完結する候補を選択"


class PrematureRanker(RecordingRanker):
    def select(
        self,
        window: CandidateWindow,
        topic: TopicWindow,
        scene: SceneWindow,
        candidates: tuple[EndBoundaryCandidate, ...],
        segments: list[TranscriptSegment],
        knowledge: tuple[KnowledgeObservation, ...],
        *,
        detailed: bool,
    ) -> tuple[str, float, str]:
        del window, topic, scene, segments, knowledge
        self.modes.append(detailed)
        return candidates[0].id, 0.9, "最初の候補"


def _segment(index: int, start: float, text: str, duration: float = 28) -> TranscriptSegment:
    return TranscriptSegment(index, start, start + duration, text)


def _window(start: float, event_start: float, event_end: float) -> CandidateWindow:
    event = CandidateEvent(
        UUID(int=int(event_start) + 1),
        EventType.LOUD_REACTION,
        event_start,
        event_end,
        0.9,
        {},
        "anchor reaction",
    )
    return CandidateWindow(start, event_end + 12, (event,), 0.9)


def _detect(
    segments: list[TranscriptSegment],
    *,
    start: float = 0,
    event_start: float = 32,
    event_end: float = 38,
    maximum: float = 900,
    normal_confidence: float = 0.84,
) -> tuple[EndBoundarySelection, RecordingRanker]:
    ranker = RecordingRanker(normal_confidence)
    result = EndBoundaryDetector(ranker).detect(
        _window(start, event_start, event_end),
        segments,
        (),
        minimum_seconds=15,
        maximum_seconds=maximum,
        candidate_count=5,
        context_window_seconds=600,
        maximum_context_seconds=1800,
        context_expansion_seconds=300,
        detailed_confidence_threshold=0.65,
    )
    return result, ranker


def test_scene_progresses_setup_development_climax_aftermath_transition() -> None:
    segments = [
        _segment(0, 0, "このボスを倒せるかな？作戦をやってみよう。"),
        _segment(1, 30, "攻撃パターンを見ながら進めています。"),
        _segment(2, 60, "やった！！ついにボスを倒した！"),
        _segment(3, 90, "危なかったけど、最後はよかったですね。"),
        _segment(4, 120, "はい次は装備を確認します。"),
    ]
    timeline = build_scene_timeline(segments)

    phases = {chunk.phase for chunk in timeline.chunks}
    assert {
        ScenePhase.SETUP,
        ScenePhase.DEVELOPMENT,
        ScenePhase.CLIMAX,
        ScenePhase.AFTERMATH,
        ScenePhase.TRANSITION,
    } <= phases
    assert timeline.clipped_scene(65, 0, 180).phase is ScenePhase.CLIMAX


def test_open_thread_is_not_cut_before_resolution_and_after_comment() -> None:
    segments = [
        _segment(0, 0, "この罠は何だろう？正体を確かめよう。"),
        _segment(1, 30, "慎重に部屋の奥へ進んでいきます。"),
        _segment(2, 60, "うわ！！床が全部落ちた！"),
        _segment(3, 90, "罠の仕組みが判明した。なるほど、そういうことか。"),
        _segment(4, 120, "いや危なかったけど面白かったですね。"),
        _segment(5, 150, "はい次は別の部屋へ行きます。"),
    ]
    result, _ = _detect(segments, event_start=62, event_end=68)

    assert result.timestamp >= 118
    assert result.scene_window.resolved_threads
    assert any(
        "open_thread_resolved" in item.source_signals
        for item in result.candidates
    )


def test_event_can_end_immediately_when_next_scene_starts() -> None:
    segments = [
        _segment(0, 0, "やった！！成功した！"),
        _segment(1, 30, "はい次は別の依頼へ進みます。"),
    ]
    result, _ = _detect(segments, event_start=2, event_end=8)

    assert result.timestamp <= 30
    assert any("scene_transition" in item.source_signals for item in result.candidates)


def test_climax_keeps_aftermath_for_laughter_victory_and_defeat() -> None:
    cases = (
        "大笑いした！！これは面白すぎる！",
        "やった！！勝った！",
        "うわ！！負けた！悔しい！",
    )
    for climax in cases:
        segments = [
            _segment(0, 0, "この勝負はどうなるかな？"),
            _segment(1, 30, climax),
            _segment(2, 60, "いや今のは本当にすごかったですね。"),
            _segment(3, 90, "はい次の勝負へ進みます。"),
        ]
        result, _ = _detect(segments)

        assert result.timestamp >= 88
        assert any(
            "aftermath_completion" in item.source_signals
            for item in result.candidates
        )


def test_multiple_threads_keep_unresolved_item_after_partial_resolution() -> None:
    segments = [
        _segment(0, 0, "この扉は罠かな？調べます。"),
        _segment(1, 30, "奥の宝箱は何だろう？あとで確認します。"),
        _segment(2, 60, "やっぱり扉は罠だった。"),
        _segment(3, 90, "宝箱はまだ開けずに先へ進みます。"),
    ]
    result, _ = _detect(segments, event_start=62, event_end=68)

    assert result.scene_window.resolved_threads
    assert result.scene_window.open_threads
    assert result.detailed_analysis_used is True


def test_callback_resolution_creates_end_candidate() -> None:
    segments = [
        _segment(0, 0, "この台詞を覚えておこう。あとで意味が分かるかな？"),
        _segment(1, 30, "別の会話をしながら物語を進めます。"),
        _segment(2, 60, "さっきの台詞の答えが判明した！"),
        _segment(3, 90, "なるほど、きれいに回収されましたね。"),
        _segment(4, 120, "はい次の場面へ進みます。"),
    ]
    result, _ = _detect(segments, event_start=62, event_end=68)

    assert result.timestamp >= 88
    assert result.scene_window.resolved_threads


def test_same_topic_can_start_a_new_scene_after_aftermath() -> None:
    segments = [
        _segment(0, 0, "ボス戦を始めます。"),
        _segment(1, 30, "やった！！最初のボスに勝った！"),
        _segment(2, 60, "危なかったけどよかったですね。"),
        _segment(3, 90, "次のボスも倒せるかな？作戦を考えます。"),
    ]
    timeline = build_scene_timeline(segments)

    assert len(timeline.scenes) == 2
    assert timeline.scene_for(35).end_seconds <= 88


def test_high_confidence_complete_scene_skips_detailed_analysis() -> None:
    segments = [
        _segment(0, 0, "この勝負に勝てるかな？"),
        _segment(1, 30, "やった！！勝った！"),
        _segment(2, 60, "危なかったけどよかったですね。"),
        _segment(3, 90, "はい次の勝負へ進みます。"),
    ]
    result, ranker = _detect(segments, normal_confidence=0.9)

    assert ranker.modes == [False]
    assert result.detailed_analysis_used is False


def test_active_reaction_cannot_end_before_first_aftermath_candidate() -> None:
    segments = [
        _segment(0, 0, "この勝負に勝てるかな？"),
        _segment(1, 30, "やった！！勝った！"),
        _segment(2, 60, "まだ興奮している、すごかったですね。"),
        _segment(3, 90, "はい次の勝負へ進みます。"),
    ]
    ranker = PrematureRanker()
    result = EndBoundaryDetector(ranker).detect(
        _window(0, 32, 38),
        segments,
        (),
        minimum_seconds=15,
        maximum_seconds=900,
        candidate_count=5,
    )

    aftermath = [
        item
        for item in result.candidates
        if "aftermath_completion" in item.source_signals
    ]
    assert aftermath
    assert result.timestamp >= min(item.timestamp for item in aftermath)


def test_topic_change_does_not_end_scene_before_delayed_payoff() -> None:
    segments = [
        _segment(0, 0, "この敵を倒せるかな？作戦を試します。"),
        _segment(1, 30, "ところで装備の色は少し変ですね。"),
        _segment(2, 60, "作戦に戻って敵の攻撃を避けます。"),
        _segment(3, 90, "やった！！敵を倒した！"),
        _segment(4, 120, "危なかったけど成功してよかったですね。"),
        _segment(5, 150, "はい次は街へ戻ります。"),
    ]
    result, _ = _detect(segments, event_start=32, event_end=38)

    assert result.timestamp >= 118
    assert result.scene_window.end_seconds > result.topic_window.end_seconds


def test_adaptive_analysis_extends_context_in_five_minute_steps() -> None:
    segments = [
        _segment(0, 0, "この長い挑戦に勝てるかな？最後までやってみよう。", 58),
        *[
            _segment(index, index * 60, "同じ挑戦を続けて結果を待っています。", 58)
            for index in range(1, 11)
        ],
        _segment(11, 660, "やった！！長い挑戦に勝った！"),
        _segment(12, 690, "本当に大変だったけどよかったですね。"),
        _segment(13, 720, "はい次は別のゲームへ進みます。"),
    ]
    result, ranker = _detect(
        segments,
        event_start=32,
        event_end=38,
        normal_confidence=0.42,
    )

    assert ranker.modes == [False, True]
    assert result.detailed_analysis_used is True
    assert result.context_end_seconds >= 720
    assert result.timestamp >= 688


def test_natural_scene_lengths_from_two_minutes_to_ten_minutes_are_allowed() -> None:
    for resolution in (150, 240, 600):
        segments = [_segment(0, 0, "この課題を解けるかな？挑戦します。")]
        segments.extend(
            _segment(index, index * 30, "同じ課題を進めています。")
            for index in range(1, resolution // 30)
        )
        segments.extend(
            [
                _segment(100, resolution, "やった！！課題を解決できた！"),
                _segment(101, resolution + 30, "難しかったけどよかったですね。"),
                _segment(102, resolution + 60, "はい次は別の課題です。"),
            ]
        )
        result, _ = _detect(segments, event_start=32, event_end=38)

        assert result.timestamp > 120
        assert result.timestamp <= 900


def test_hard_maximum_cuts_scene_safely_at_fifteen_minutes() -> None:
    segments = [
        _segment(index, index * 60, "未解決の長い挑戦を続けています。", 58)
        for index in range(20)
    ]
    first, _ = _detect(segments, event_start=32, event_end=38)
    second, _ = _detect(segments, event_start=32, event_end=38)

    assert first.timestamp <= 900
    assert first.timestamp == second.timestamp
    assert any("hard_maximum_guard" in item.source_signals for item in first.candidates)
    assert first.confidence <= 0.45


def test_explainability_keeps_scene_threads_candidates_and_confidence() -> None:
    segments = [
        _segment(0, 0, "この謎の答えは何だろう？"),
        _segment(1, 30, "うわ！！答えが分かった！"),
        _segment(2, 60, "なるほど、これは面白かったですね。"),
        _segment(3, 90, "はい次の話へ進みます。"),
    ]
    result, _ = _detect(segments)
    analysis = boundary_analysis(_window(0, 32, 38), result)

    assert analysis["scene_phase"]
    assert "open_threads" in analysis
    assert "resolved_threads" in analysis
    assert analysis["end_boundary_candidates"]
    selected = analysis["selected_end_boundary"]
    boundary_confidence = analysis["boundary_confidence"]
    scene_confidence = analysis["scene_completion_confidence"]
    assert isinstance(selected, dict) and selected["reason"]
    assert isinstance(boundary_confidence, int | float)
    assert isinstance(scene_confidence, int | float)
    assert 0 <= boundary_confidence <= 1
    assert 0 <= scene_confidence <= 1
