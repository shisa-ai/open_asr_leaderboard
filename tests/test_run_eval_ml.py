import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
API_DIR = REPO_ROOT / "api"


def _load_run_eval_ml():
    sys.path.insert(0, str(API_DIR))
    spec = importlib.util.spec_from_file_location("api_run_eval_ml", API_DIR / "run_eval_ml.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RunEvalMLTests(unittest.TestCase):
    def test_use_url_duration_accepts_multilingual_rows_duration_field(self):
        module = _load_run_eval_ml()

        row = {"duration": 25.86}

        self.assertEqual(module.get_audio_duration_from_row(row), 25.86)

    def test_use_url_duration_keeps_audio_length_s_for_legacy_rows(self):
        module = _load_run_eval_ml()

        row = {"audio_length_s": 3.5, "duration": 25.86}

        self.assertEqual(module.get_audio_duration_from_row(row), 3.5)


if __name__ == "__main__":
    unittest.main()
