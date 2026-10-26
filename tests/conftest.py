import pytest

from freshguard.pipeline import FreshGuard
from freshguard.storage import Storage


@pytest.fixture()
def storage(tmp_path):
    s = Storage(tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture()
def fg(storage):
    return FreshGuard(storage=storage)
