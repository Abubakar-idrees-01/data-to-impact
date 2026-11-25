"""Where StudyRAG keeps its data: the project folder for a source checkout, ~/.studyrag for a pip install."""

from pathlib import Path

from studyrag import config


def test_source_checkout_uses_the_project_folder(monkeypatch):
    monkeypatch.delenv("STUDYRAG_HOME", raising=False)
    assert config.default_home() == config.SOURCE_ROOT
    assert (config.SOURCE_ROOT / "pyproject.toml").exists()


def test_installed_package_uses_home_folder(monkeypatch, tmp_path):
    monkeypatch.delenv("STUDYRAG_HOME", raising=False)
    monkeypatch.setattr(config, "SOURCE_ROOT", tmp_path / "lib" / "python3.12")      # site-packages layout
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "user")
    assert config.default_home() == tmp_path / "user" / ".studyrag"


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("STUDYRAG_HOME", str(tmp_path / "custom"))
    assert config.default_home() == tmp_path / "custom"
