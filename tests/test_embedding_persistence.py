"""Tests for embedding persistence helpers in runner.py."""

import json
import os
from pathlib import Path

import pytest

from shinka.core.runner import EvolutionRunner


class TestPersistEmbedding:
    """Tests for EvolutionRunner._persist_embedding."""

    def test_round_trip(self, tmp_path):
        embedding = [0.1, 0.2, 0.3, -0.5, 1.0]
        EvolutionRunner._persist_embedding(tmp_path, embedding)

        loaded = EvolutionRunner._load_persisted_embedding(tmp_path)
        assert loaded == embedding

    def test_writes_valid_json(self, tmp_path):
        embedding = [1.0, 2.0, 3.0]
        EvolutionRunner._persist_embedding(tmp_path, embedding)

        path = tmp_path / "embedding.json"
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data == embedding

    def test_overwrites_existing(self, tmp_path):
        EvolutionRunner._persist_embedding(tmp_path, [1.0, 2.0])
        EvolutionRunner._persist_embedding(tmp_path, [3.0, 4.0])

        loaded = EvolutionRunner._load_persisted_embedding(tmp_path)
        assert loaded == [3.0, 4.0]

    def test_no_temp_file_on_success(self, tmp_path):
        EvolutionRunner._persist_embedding(tmp_path, [1.0, 2.0])

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []

    def test_cleanup_on_serialization_error(self, tmp_path):
        # Pass something that json.dump can't serialize
        class Unserializable:
            pass

        EvolutionRunner._persist_embedding(tmp_path, [Unserializable()])

        # No embedding.json should exist
        assert not (tmp_path / "embedding.json").exists()
        # No orphaned temp files
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []


class TestLoadPersistedEmbedding:
    """Tests for EvolutionRunner._load_persisted_embedding."""

    def test_returns_none_when_no_file(self, tmp_path):
        assert EvolutionRunner._load_persisted_embedding(tmp_path) is None

    def test_returns_none_for_empty_list(self, tmp_path):
        path = tmp_path / "embedding.json"
        with open(path, "w") as f:
            json.dump([], f)
        assert EvolutionRunner._load_persisted_embedding(tmp_path) is None

    def test_returns_none_for_non_list(self, tmp_path):
        path = tmp_path / "embedding.json"
        with open(path, "w") as f:
            json.dump({"key": "val"}, f)
        assert EvolutionRunner._load_persisted_embedding(tmp_path) is None

    def test_returns_none_for_list_of_strings(self, tmp_path):
        path = tmp_path / "embedding.json"
        with open(path, "w") as f:
            json.dump(["a", "b", "c"], f)
        assert EvolutionRunner._load_persisted_embedding(tmp_path) is None

    def test_returns_none_for_corrupt_json(self, tmp_path):
        path = tmp_path / "embedding.json"
        path.write_text("{not valid json")
        assert EvolutionRunner._load_persisted_embedding(tmp_path) is None

    def test_accepts_int_elements(self, tmp_path):
        path = tmp_path / "embedding.json"
        with open(path, "w") as f:
            json.dump([1, 2, 3], f)
        result = EvolutionRunner._load_persisted_embedding(tmp_path)
        assert result == [1, 2, 3]

    def test_accepts_float_elements(self, tmp_path):
        path = tmp_path / "embedding.json"
        with open(path, "w") as f:
            json.dump([0.1, -0.5, 1.0], f)
        result = EvolutionRunner._load_persisted_embedding(tmp_path)
        assert result == [0.1, -0.5, 1.0]
