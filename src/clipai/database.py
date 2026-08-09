from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import files

import psycopg
from psycopg import Connection


@contextmanager
def connect(database_url: str) -> Iterator[Connection[tuple[object, ...]]]:
    with psycopg.connect(database_url) as connection:
        yield connection


def database_is_ready(database_url: str) -> bool:
    try:
        with connect(database_url) as connection, connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            return cursor.fetchone() == (1,)
    except psycopg.Error:
        return False


def apply_migrations(database_url: str) -> None:
    migration_root = files("clipai.migrations")
    migrations = sorted(
        item for item in migration_root.iterdir() if item.name.endswith(".sql")
    )
    with connect(database_url) as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version text PRIMARY KEY,
                applied_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        cursor.execute("SELECT pg_advisory_xact_lock(hashtext('clipai_schema_migrations'))")
        for migration in migrations:
            cursor.execute("SELECT 1 FROM schema_migrations WHERE version = %s", (migration.name,))
            if cursor.fetchone() is not None:
                continue
            cursor.execute(migration.read_text(encoding="utf-8"))
            cursor.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s)",
                (migration.name,),
            )
        connection.commit()
