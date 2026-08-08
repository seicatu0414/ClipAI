import argparse
import json
from dataclasses import asdict
from uuid import UUID

from clipai.config import get_settings
from clipai.domain import SourceSpec
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

    snapshot = repository.get_job(args.job_id)
    if snapshot is None:
        raise SystemExit("transcription job not found")
    print(json.dumps(asdict(snapshot), default=str, ensure_ascii=False))
