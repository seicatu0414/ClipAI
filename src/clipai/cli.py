import argparse
import json
from dataclasses import asdict
from pathlib import Path
from uuid import UUID

from clipai.config import get_settings
from clipai.domain import SourceSpec
from clipai.events.configuration import event_configuration
from clipai.events.detectors import JapaneseTranscriptRuleDetector
from clipai.events.repository import EventRepository
from clipai.repository import TranscriptionRepository


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clipai")
    commands = parser.add_subparsers(dest="command", required=True)

    submit = commands.add_parser("submit", help="submit a local file or YouTube URL")
    submit.add_argument("source")
    submit.add_argument("--model-size")
    submit.add_argument("--language")

    inspect = commands.add_parser("status", help="inspect a transcription job")
    inspect.add_argument("job_id", type=UUID)

    detect = commands.add_parser("detect-events", help="detect events for a transcript")
    detect.add_argument("transcript_id", type=UUID)

    events = commands.add_parser("events", help="show an event timeline")
    events.add_argument("transcript_id", type=UUID)
    events.add_argument("--job-id", type=UUID)
    return parser


def main() -> None:
    args = _parser().parse_args()
    settings = get_settings()
    repository = TranscriptionRepository(settings.database_url)
    if args.command == "submit":
        source = SourceSpec.parse(args.source)
        job = repository.create_job(
            source,
            model_size=args.model_size or settings.model_size,
            language=args.language or settings.language,
        )
        print(json.dumps(asdict(job), default=str, ensure_ascii=False))
        return

    if args.command == "detect-events":
        source = repository.get_transcript_source(args.transcript_id)
        if source is None or not Path(source.audio_artifact_path).is_file():
            raise SystemExit("transcript has no normalized audio artifact")
        job = EventRepository(settings.database_url).create_job(
            args.transcript_id,
            detector_version=f"audio-rms-v1+{JapaneseTranscriptRuleDetector.VERSION}+merge-v1",
            configuration=event_configuration(settings),
        )
        print(json.dumps(asdict(job), default=str, ensure_ascii=False))
        return

    if args.command == "events":
        events = EventRepository(settings.database_url).list_events(
            args.transcript_id, args.job_id
        )
        print(json.dumps([asdict(event) for event in events], default=str, ensure_ascii=False))
        return

    snapshot = repository.get_job(args.job_id)
    if snapshot is None:
        raise SystemExit("transcription job not found")
    print(json.dumps(asdict(snapshot), default=str, ensure_ascii=False))
