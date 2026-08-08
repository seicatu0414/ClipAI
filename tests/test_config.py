from clipai.config import Settings


def test_settings_have_local_safe_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.environment == "local"
    assert settings.log_level == "INFO"
    assert settings.database_url.startswith("postgresql://")
