import numpy as np
import pytest

from src.file_handlers.base_handler import FileFormatHandler


class TestFileFormatHandler:
    file_path = "test_file"

    def test_load_raises_not_implemented_error(
        self, base_handler: FileFormatHandler
    ) -> None:
        with pytest.raises(NotImplementedError):
            base_handler.load(self.file_path)

    def test_save_raises_not_implemented_error(
        self, base_handler: FileFormatHandler
    ) -> None:
        with pytest.raises(NotImplementedError):
            base_handler.save(np.array([1, 2, 3]), self.file_path)
