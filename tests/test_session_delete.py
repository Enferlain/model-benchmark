from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

# Add backend to path
sys.path.append("backend")

# Mock database module before importing state/main
sys.modules["database"] = MagicMock()
import database as db

# Now import the modules to test
import state
import main
import scanner


def test_delete_model_session_only():
    print("Running test_delete_model_session_only...")
    # Setup initial state
    mock_model = MagicMock()
    mock_model.id = "hash123"
    mock_model.name = "Test Model"
    state.models_db = [mock_model]

    # Mock DB session
    mock_session = MagicMock()
    db.get_session.return_value.__enter__.return_value = mock_session

    # Action: Delete without file deletion
    main.delete_model("hash123", delete_file=False)

    # Assertions
    # 1. Model removed from session
    assert len(state.models_db) == 0, "Model should be removed from session"

    # 2. DB delete NOT called
    mock_session.delete.assert_not_called()
    print("Passed.")


def test_delete_model_with_file():
    print("Running test_delete_model_with_file...")
    # Setup initial state
    mock_model = MagicMock()
    mock_model.id = "hash456"
    mock_model.path = "d:/Projects/model-benchmark-explorer/models/test.safetensors"
    state.models_db = [mock_model]

    # Mock DB session and return value
    mock_session = MagicMock()
    db.get_session.return_value.__enter__.return_value = mock_session
    mock_db_model = MagicMock()
    mock_db_model.path = "d:/Projects/model-benchmark-explorer/models/test.safetensors"
    mock_session.get.return_value = mock_db_model

    # Mock Path
    with patch("main.Path") as MockPath:
        mock_path_instance = MagicMock()
        # Ensure resolve returns the mock itself (common pattern) or another mock
        MockPath.return_value.resolve.return_value = mock_path_instance

        mock_path_instance.exists.return_value = True
        mock_path_instance.is_dir.return_value = False
        mock_path_instance.is_symlink.return_value = False

        # Mock data_loader.MODELS_DIR
        with patch("main.data_loader.MODELS_DIR") as MockModelsDir:
            # Force string representation to be completely consistent
            # Use double backslashes for safety in python strings
            mock_path_instance.__str__.return_value = (
                "D:\\Projects\\model-benchmark-explorer\\models\\test.safetensors"
            )
            MockModelsDir.resolve.return_value = MagicMock()
            MockModelsDir.resolve.return_value.__str__.return_value = (
                "D:\\Projects\\model-benchmark-explorer\\models"
            )

            main.delete_model("hash456", delete_file=True)

            # Assertions
            # 1. Model removed from session
            assert len(state.models_db) == 0, "Model should be removed from session"

            # 2. File unlink CALLED
            mock_path_instance.unlink.assert_called_once()

            # 3. DB delete NOT called
            mock_session.delete.assert_not_called()
    print("Passed.")


def test_scanner_uses_session_state():
    print("Running test_scanner_uses_session_state...")
    # Setup state with 1 active model
    mock_model = MagicMock()
    mock_model.hash = "hash789"
    mock_model.name = "ActiveModel"
    mock_model.path = "some/path"
    state.models_db = [mock_model]

    # Mock DB session (should NOT be used for fetching models list)
    mock_session = MagicMock()
    db.get_session.return_value.__enter__.return_value = mock_session

    # Mock data_loader
    with patch("scanner.data_loader") as mock_loader:
        mock_loader.get_all_prompts_metadata.return_value = [
            {"text": "prompt", "alias": "p"}
        ]
        mock_loader.ASSETS_DIR = Path("assets")

        # Mock inferencer to avoid actual generation
        with patch("scanner.inference.SDXLInferencer") as MockInf:
            # Action: Generate
            scanner.generate_images_only(state.ScanOptions())

            # Assertions
            pass
    print("Passed.")


if __name__ == "__main__":
    try:
        test_delete_model_session_only()
        test_delete_model_with_file()
        test_scanner_uses_session_state()
        print("All tests passed!")
    except Exception as e:
        print(f"Tests failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
