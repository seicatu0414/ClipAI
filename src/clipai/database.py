from collections.abc import Iterator
from contextlib import contextmanager

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
