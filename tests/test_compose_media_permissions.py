from pathlib import Path


def test_compose_initializes_writable_media_directories() -> None:
    compose = Path("compose.yaml").read_text(encoding="utf-8")

    assert "media-init:" in compose
    assert 'user: "0:0"' in compose
    assert "chown -R 10001:10001 /data/work /data/artifacts" in compose
    assert compose.count("condition: service_completed_successfully") == 2
