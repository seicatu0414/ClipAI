from dataclasses import dataclass, replace

from clipai.candidates.domain import (
    OpenThread,
    ScenePhase,
    SceneWindow,
    SemanticChunk,
    ThreadStatus,
)
from clipai.domain import TranscriptSegment

_TRANSITION_MARKERS = (
    "はい次",
    "次は",
    "さて",
    "それでは",
    "じゃあ行こう",
    "次行こう",
    "移動しよう",
    "話は変わ",
)
_CLIMAX_MARKERS = (
    "やった",
    "勝った",
    "負けた",
    "倒した",
    "死んだ",
    "うわ",
    "嘘",
    "成功",
    "クリア",
)
_AFTERMATH_MARKERS = (
    "すごかった",
    "面白かった",
    "危なかった",
    "よかった",
    "だめだった",
    "びっくり",
    "笑った",
    "なるほど",
    "感想",
    "反省",
    "ということ",
)
_SETUP_MARKERS = (
    "やってみ",
    "行ってみ",
    "倒す",
    "勝てる",
    "どうなる",
    "何だろ",
    "絶対",
    "目標",
    "罠",
)
_OPEN_MARKERS = (
    "どうなる",
    "何だろ",
    "やってみ",
    "倒す",
    "勝てる",
    "罠",
    "あとで",
    "覚えて",
)
_RESOLUTION_MARKERS = (
    "やっぱり",
    "できた",
    "勝った",
    "負けた",
    "倒した",
    "なるほど",
    "答え",
    "だった",
    "判明",
    "回収",
    "さっきの",
)
_REACTION_MARKERS = ("うわ", "えっ", "まじ", "やった", "笑", "すご", "びっくり")


@dataclass(frozen=True)
class SceneTimeline:
    chunks: tuple[SemanticChunk, ...]
    scenes: tuple[SceneWindow, ...]

    def scene_for(self, timestamp: float) -> SceneWindow:
        containing = [
            scene
            for scene in self.scenes
            if scene.start_seconds <= timestamp <= scene.end_seconds
        ]
        if containing:
            return containing[0]
        if not self.scenes:
            return _empty_scene(timestamp)
        return min(
            self.scenes,
            key=lambda scene: min(
                abs(timestamp - scene.start_seconds), abs(timestamp - scene.end_seconds)
            ),
        )

    def clipped_scene(
        self, timestamp: float, context_start: float, context_end: float
    ) -> SceneWindow:
        scene = self.scene_for(timestamp)
        anchor_phase = next(
            (
                chunk.phase
                for chunk in self.chunks
                if chunk.start_seconds <= timestamp <= chunk.end_seconds
            ),
            scene.phase,
        )
        clipped_end = min(scene.end_seconds, context_end)
        incomplete = scene.end_seconds > context_end
        resolved_threads = tuple(
            item
            for item in scene.resolved_threads
            if item.resolved_at is not None and item.resolved_at <= context_end
        )
        unresolved_in_context = tuple(
            replace(item, status=ThreadStatus.OPEN, resolved_at=None)
            for item in scene.resolved_threads
            if item.opened_at <= context_end
            and item.resolved_at is not None
            and item.resolved_at > context_end
        )
        open_threads = tuple(
            item for item in scene.open_threads if item.opened_at <= context_end
        ) + unresolved_in_context
        signals = scene.source_signals + (("context_edge",) if incomplete else ())
        return replace(
            scene,
            start_seconds=max(scene.start_seconds, context_start),
            end_seconds=clipped_end,
            phase=anchor_phase,
            open_threads=open_threads,
            resolved_threads=resolved_threads,
            completion_confidence=(
                min(scene.completion_confidence, 0.35)
                if incomplete
                else scene.completion_confidence
            ),
            source_signals=tuple(dict.fromkeys(signals)),
        )


def build_scene_timeline(
    segments: list[TranscriptSegment],
    *,
    target_chunk_seconds: float = 30.0,
    maximum_chunk_seconds: float = 45.0,
) -> SceneTimeline:
    chunks = _semantic_chunks(segments, target_chunk_seconds, maximum_chunk_seconds)
    phased = _infer_phase_sequence(chunks)
    return SceneTimeline(phased, _group_scenes(phased))


def _semantic_chunks(
    segments: list[TranscriptSegment],
    target_seconds: float,
    maximum_seconds: float,
) -> tuple[SemanticChunk, ...]:
    if not segments:
        return ()
    chunks: list[SemanticChunk] = []
    group: list[TranscriptSegment] = []
    for segment in segments:
        group.append(segment)
        duration = group[-1].end_seconds - group[0].start_seconds
        complete = segment.text.rstrip().endswith(("。", "！", "？", "!", "?"))
        if duration >= maximum_seconds or (
            duration >= target_seconds * 0.75 and complete
        ):
            chunks.append(_make_chunk(group))
            group = []
    if group:
        chunks.append(_make_chunk(group))
    return tuple(chunks)


def _make_chunk(segments: list[TranscriptSegment]) -> SemanticChunk:
    text = " ".join(item.text.strip() for item in segments if item.text.strip())
    phase, signals = _classify_phase(text)
    return SemanticChunk(
        segments[0].start_seconds,
        segments[-1].end_seconds,
        text,
        phase,
        signals,
    )


def _classify_phase(text: str) -> tuple[ScenePhase, tuple[str, ...]]:
    if any(marker in text for marker in _TRANSITION_MARKERS):
        return ScenePhase.TRANSITION, ("explicit_transition",)
    exclamations = text.count("！") + text.count("!")
    if exclamations >= 2 or any(marker in text for marker in _CLIMAX_MARKERS):
        return ScenePhase.CLIMAX, ("strong_reaction_or_outcome",)
    if any(marker in text for marker in _AFTERMATH_MARKERS):
        return ScenePhase.AFTERMATH, ("reflection_or_emotional_settling",)
    if any(marker in text for marker in _SETUP_MARKERS) or "？" in text or "?" in text:
        return ScenePhase.SETUP, ("goal_or_expectation",)
    return ScenePhase.DEVELOPMENT, ("continuing_action",)


def _infer_phase_sequence(
    chunks: tuple[SemanticChunk, ...],
) -> tuple[SemanticChunk, ...]:
    result: list[SemanticChunk] = []
    previous: ScenePhase | None = None
    for chunk in chunks:
        phase = chunk.phase
        signals = list(chunk.source_signals)
        if (
            previous is ScenePhase.CLIMAX
            and phase is ScenePhase.DEVELOPMENT
            and not any(marker in chunk.text for marker in _SETUP_MARKERS)
        ):
            phase = ScenePhase.AFTERMATH
            signals.append("post_climax_continuation")
        result.append(replace(chunk, phase=phase, source_signals=tuple(signals)))
        previous = phase
    return tuple(result)


def _group_scenes(chunks: tuple[SemanticChunk, ...]) -> tuple[SceneWindow, ...]:
    if not chunks:
        return ()
    groups: list[list[SemanticChunk]] = []
    current: list[SemanticChunk] = []
    for chunk in chunks:
        if chunk.phase is ScenePhase.TRANSITION:
            if current:
                groups.append(current)
                current = []
            groups.append([chunk])
            continue
        if current and _starts_new_scene(current, chunk):
            groups.append(current)
            current = []
        current.append(chunk)
    if current:
        groups.append(current)
    return tuple(_scene_from_chunks(group) for group in groups if group)


def _starts_new_scene(
    current: list[SemanticChunk], chunk: SemanticChunk
) -> bool:
    previous = current[-1]
    gap = chunk.start_seconds - previous.end_seconds
    duration = previous.end_seconds - current[0].start_seconds
    aftermath_to_setup = (
        previous.phase is ScenePhase.AFTERMATH and chunk.phase is ScenePhase.SETUP
    )
    scene_closing_silence = gap >= 45 or (
        gap >= 20 and previous.phase is ScenePhase.AFTERMATH
    )
    return scene_closing_silence or aftermath_to_setup or (
        duration >= 600 and chunk.phase is ScenePhase.SETUP
    )


def _scene_from_chunks(chunks: list[SemanticChunk]) -> SceneWindow:
    open_threads, resolved_threads = _track_threads(chunks)
    phases = tuple(dict.fromkeys(chunk.phase for chunk in chunks))
    phase = next(
        (
            item
            for item in (
                ScenePhase.CLIMAX,
                ScenePhase.AFTERMATH,
                ScenePhase.DEVELOPMENT,
                ScenePhase.SETUP,
                ScenePhase.TRANSITION,
            )
            if item in phases
        ),
        chunks[-1].phase,
    )
    transition = next(
        (
            marker
            for marker in _TRANSITION_MARKERS
            if marker in chunks[-1].text
        ),
        None,
    )
    has_aftermath = ScenePhase.AFTERMATH in phases
    has_transition = ScenePhase.TRANSITION in phases
    important_open = [item for item in open_threads if item.confidence >= 0.65]
    completion = 0.35
    completion_signals: list[str] = []
    if has_aftermath:
        completion += 0.2
        completion_signals.append("aftermath_observed")
    if has_transition:
        completion += 0.3
        completion_signals.append("next_scene_transition")
    if resolved_threads:
        completion += 0.15
        completion_signals.append("open_thread_resolved")
    if important_open:
        completion -= min(0.3, 0.1 * len(important_open))
        completion_signals.append("important_open_thread_remains")
    emotional = _emotional_state(chunks)
    reaction = _reaction_state(chunks)
    goal = _primary_goal(chunks)
    confidence = min(
        1.0,
        0.45
        + (0.15 if len(phases) >= 3 else 0)
        + (0.15 if has_transition else 0)
        + (0.1 if resolved_threads else 0),
    )
    return SceneWindow(
        chunks[0].start_seconds,
        chunks[-1].end_seconds,
        phase,
        phases,
        goal,
        open_threads,
        resolved_threads,
        emotional,
        reaction,
        transition,
        round(confidence, 4),
        round(max(0.0, min(completion, 1.0)), 4),
        tuple(completion_signals or ("scene_continuing",)),
    )


def _track_threads(
    chunks: list[SemanticChunk],
) -> tuple[tuple[OpenThread, ...], tuple[OpenThread, ...]]:
    open_items: list[OpenThread] = []
    resolved: list[OpenThread] = []
    for chunk in chunks:
        resolved_this_chunk = bool(open_items) and any(
            marker in chunk.text for marker in _RESOLUTION_MARKERS
        )
        if resolved_this_chunk:
            item = open_items.pop(0)
            resolved.append(
                replace(
                    item,
                    status=ThreadStatus.RESOLVED,
                    confidence=min(1.0, item.confidence + 0.1),
                    resolved_at=chunk.end_seconds,
                )
            )
        marker = next((item for item in _OPEN_MARKERS if item in chunk.text), None)
        if marker and not resolved_this_chunk:
            statement = _thread_label(chunk.text, marker)
            if marker in ("罠", "倒す", "勝てる", "覚えて"):
                confidence = 0.78
            elif marker == "あとで":
                confidence = 0.58
            else:
                confidence = 0.66
            open_items.append(
                OpenThread(
                    statement,
                    ThreadStatus.OPEN,
                    confidence,
                    chunk.start_seconds,
                )
            )
    return tuple(open_items), tuple(resolved)


def _thread_label(text: str, marker: str) -> str:
    compact = " ".join(text.split())
    excerpt = compact[:80]
    return f"{excerpt} [{marker}]" if excerpt else marker


def _primary_goal(chunks: list[SemanticChunk]) -> str | None:
    for chunk in chunks:
        if chunk.phase is ScenePhase.SETUP:
            return " ".join(chunk.text.split())[:120]
    return None


def _emotional_state(chunks: list[SemanticChunk]) -> str:
    text = " ".join(item.text for item in chunks[-2:])
    if any(marker in text for marker in ("笑", "面白", "楽しい")):
        return "amused_settling" if chunks[-1].phase is ScenePhase.AFTERMATH else "amused"
    if any(marker in text for marker in ("悔しい", "負け", "だめ")):
        return "disappointed_reflection"
    if any(marker in text for marker in ("やった", "勝った", "よかった")):
        return "relieved_or_excited"
    return "neutral_or_unknown"


def _reaction_state(chunks: list[SemanticChunk]) -> str:
    last = chunks[-1]
    if any(marker in last.text for marker in _REACTION_MARKERS):
        return (
            "aftermath_complete"
            if last.phase in (ScenePhase.AFTERMATH, ScenePhase.TRANSITION)
            else "reaction_active"
        )
    if ScenePhase.CLIMAX in (item.phase for item in chunks[-2:]):
        return "reaction_pending"
    return "settled"


def _empty_scene(timestamp: float) -> SceneWindow:
    return SceneWindow(
        timestamp,
        timestamp,
        ScenePhase.DEVELOPMENT,
        (ScenePhase.DEVELOPMENT,),
        None,
        (),
        (),
        "neutral_or_unknown",
        "settled",
        None,
        0.2,
        0.2,
        ("no_transcript",),
    )
