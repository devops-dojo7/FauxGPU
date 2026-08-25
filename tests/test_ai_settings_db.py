import importlib

import pytest


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_SETTINGS_DB_PATH", str(tmp_path / "ai_settings.db"))
    monkeypatch.setenv("AI_SETTINGS_SECRET", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
    from api import ai_settings_db

    return importlib.reload(ai_settings_db)


def test_set_and_get_key_round_trips(db):
    db.set_key("openai", "sk-test-123")
    assert db.get_key("openai") == "sk-test-123"


def test_get_key_missing_returns_none(db):
    assert db.get_key("openai") is None


def test_list_configured_reflects_stored_providers(db):
    db.set_key("openai", "sk-a")
    db.set_key("groq", "gk-b")
    assert db.list_configured() == {"openai", "groq"}


def test_delete_key_removes_it(db):
    db.set_key("openai", "sk-a")
    db.delete_key("openai")
    assert db.get_key("openai") is None
    assert db.list_configured() == set()


def test_set_key_updates_existing_provider(db):
    db.set_key("openai", "sk-old")
    db.set_key("openai", "sk-new")
    assert db.get_key("openai") == "sk-new"
    assert db.list_configured() == {"openai"}


def test_missing_secret_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_SETTINGS_DB_PATH", str(tmp_path / "ai_settings.db"))
    monkeypatch.delenv("AI_SETTINGS_SECRET", raising=False)
    from api import ai_settings_db

    importlib.reload(ai_settings_db)
    with pytest.raises(RuntimeError, match="AI_SETTINGS_SECRET"):
        ai_settings_db.set_key("openai", "sk-a")
