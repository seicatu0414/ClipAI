from clipai.knowledge.repository import next_version_number


def test_version_numbers_are_monotonic() -> None:
    assert next_version_number(None) == 1
    assert next_version_number(1) == 2
    assert next_version_number(8) == 9
