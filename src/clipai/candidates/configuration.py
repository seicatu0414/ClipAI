from clipai.config import Settings
from clipai.events.domain import JsonValue

PIPELINE_VERSION = "candidate-ranking-v2-end-boundary"
PROMPT_VERSION = "v1"


def candidate_configuration(settings: Settings) -> dict[str, JsonValue]:
    return {
        "target_count": settings.candidate_target_count,
        "minimum_seconds": settings.candidate_minimum_seconds,
        "maximum_seconds": settings.candidate_maximum_seconds,
        "padding_before_seconds": settings.candidate_padding_before_seconds,
        "padding_after_seconds": settings.candidate_padding_after_seconds,
        "merge_gap_seconds": settings.candidate_merge_gap_seconds,
        "overlap_threshold": settings.candidate_overlap_threshold,
        "maximum_knowledge_observations": settings.candidate_maximum_knowledge_observations,
        "context_window_seconds": settings.candidate_context_window_seconds,
        "end_boundary_count": settings.candidate_end_boundary_count,
    }
