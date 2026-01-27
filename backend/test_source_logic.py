import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

# Mock torch for imports
sys.modules["torch"] = MagicMock()
sys.modules["safetensors.torch"] = MagicMock()
sys.modules["blake3"] = MagicMock()

# Import modules to test
from app.services import model_manager, downloader
from app.core import database as db
from app.core.database import Model
from app.core import state

@pytest.fixture
def mock_session():
    mock_session = MagicMock()
    # Correct context manager mocking
    mock_session.__enter__.return_value = mock_session
    mock_session.__exit__.return_value = None
    return mock_session

@pytest.fixture
def mock_db_get_session(mock_session):
    with patch("app.core.database.get_session", return_value=mock_session):
        yield mock_session

def test_sync_models_sets_local_source(mock_db_get_session):
    # Setup
    with patch("app.services.prompt_manager.get_available_models_from_disk") as mock_get_models, \
         patch("app.services.model_manager.compute_model_hash", return_value="hash123"), \
         patch("app.services.model_manager.scan_model_type", return_value=("sd1.5", "epsilon", False)), \
         patch("app.core.database.Session.exec"), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.stat") as mock_stat:

        mock_get_models.return_value = [{
            "name": "test_model",
            "path": "/models/test_model.safetensors",
            "hash": "hash123"
        }]

        mock_stat_obj = MagicMock()
        mock_stat_obj.st_mtime = 1000
        mock_stat_obj.st_size = 5000
        mock_stat.return_value = mock_stat_obj

        # Mock existing model check -> returns None (New model)
        mock_db_get_session.exec.return_value.first.return_value = None
        mock_db_get_session.get.return_value = None

        # Execute
        model_manager.sync_models_with_db()

        # Verify
        # Check if Model was instantiated with source="Local"
        # We can verify the calls to session.add()
        # Find the call that adds a Model instance
        added_models = []
        for call in mock_db_get_session.add.call_args_list:
            arg = call[0][0]
            if isinstance(arg, Model):
                added_models.append(arg)

        assert len(added_models) > 0
        added_model = added_models[0]

        assert added_model.name == "test_model"
        assert added_model.source == "Local"
        assert added_model.hash == "hash123"


def test_downloader_updates_source():
    # Setup
    with patch("app.services.model_manager.compute_model_hash", return_value="hash_dl"), \
         patch("app.services.model_manager.scan_model_type", return_value=("sdxl", "v_prediction", True)), \
         patch("app.core.database.get_session") as mock_get_session_ctx, \
         patch("requests.get") as mock_get, \
         patch("builtins.open", new_callable=MagicMock), \
         patch("pathlib.Path.stat") as mock_stat:

        mock_session = MagicMock()
        mock_get_session_ctx.return_value.__enter__.return_value = mock_session

        # Mock Response
        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": "100"}
        mock_resp.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_resp

        # Mock file stat
        mock_stat_obj = MagicMock()
        mock_stat_obj.st_mtime = 2000
        mock_stat_obj.st_size = 100
        mock_stat.return_value = mock_stat_obj

        # Scenario 1: New Model
        mock_session.get.return_value = None

        # Execute
        downloader.download_model_task("http://example.com/model.safetensors", "dl_model", "Civitai")

        # Verify
        args, _ = mock_session.add.call_args
        added_model = args[0]
        assert added_model.source == "Civitai"
        assert added_model.name == "dl_model"
        assert added_model.hash == "hash_dl"

        # Reset mocks for Scenario 2
        mock_session.reset_mock()

        # Scenario 2: Existing Model (Update)
        existing_model = Model(
            hash="hash_dl",
            name="Old Name",
            source="Local",
            filename="old.safetensors",
            path="/old/path.safetensors"
        )
        mock_session.get.return_value = existing_model

        # Execute
        downloader.download_model_task("http://example.com/model.safetensors", "dl_model", "HuggingFace")

        # Verify
        # It should update the existing model instance
        assert existing_model.source == "HuggingFace"
        mock_session.add.assert_called_with(existing_model)
