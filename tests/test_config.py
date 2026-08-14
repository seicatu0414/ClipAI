from clipai.config import Settings


def test_settings_have_local_safe_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.environment == "local"
    assert settings.log_level == "INFO"
    assert settings.database_url.startswith("postgresql://")
    assert settings.candidate_maximum_seconds == 900
    assert settings.candidate_context_window_seconds == 600
    assert settings.candidate_context_expansion_seconds == 300
