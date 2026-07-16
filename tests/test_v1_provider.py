import importlib
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_vendor_sdk_stubs():
    assemblyai = types.ModuleType("assemblyai")
    assemblyai.settings = types.SimpleNamespace(api_key=None)
    assemblyai.Transcriber = type("Transcriber", (), {})
    assemblyai.TranscriptionConfig = type("TranscriptionConfig", (), {"__init__": lambda self, **kwargs: None})
    assemblyai.TranscriptStatus = types.SimpleNamespace(error="error")
    sys.modules.setdefault("assemblyai", assemblyai)

    elevenlabs = types.ModuleType("elevenlabs")
    elevenlabs_client = types.ModuleType("elevenlabs.client")
    elevenlabs_client.ElevenLabs = type("ElevenLabs", (), {"__init__": lambda self, **kwargs: None})
    sys.modules.setdefault("elevenlabs", elevenlabs)
    sys.modules.setdefault("elevenlabs.client", elevenlabs_client)

    rev_ai = types.ModuleType("rev_ai")
    rev_ai_apiclient = types.ModuleType("rev_ai.apiclient")
    rev_ai_apiclient.RevAiAPIClient = type("RevAiAPIClient", (), {})
    rev_ai_models = types.ModuleType("rev_ai.models")
    rev_ai_models.CustomerUrlData = type("CustomerUrlData", (), {"__init__": lambda self, *args, **kwargs: None})
    sys.modules.setdefault("rev_ai", rev_ai)
    sys.modules.setdefault("rev_ai.apiclient", rev_ai_apiclient)
    sys.modules.setdefault("rev_ai.models", rev_ai_models)

    speechmatics = types.ModuleType("speechmatics")
    speechmatics_models = types.ModuleType("speechmatics.models")
    speechmatics_models.ConnectionSettings = type("ConnectionSettings", (), {})
    speechmatics_models.BatchTranscriptionConfig = type("BatchTranscriptionConfig", (), {})
    speechmatics_models.FetchData = type("FetchData", (), {})
    speechmatics_batch_client = types.ModuleType("speechmatics.batch_client")
    speechmatics_batch_client.BatchClient = type("BatchClient", (), {})
    sys.modules.setdefault("speechmatics", speechmatics)
    sys.modules.setdefault("speechmatics.models", speechmatics_models)
    sys.modules.setdefault("speechmatics.batch_client", speechmatics_batch_client)

    requests_toolbelt = types.ModuleType("requests_toolbelt")
    requests_toolbelt.MultipartEncoder = type("MultipartEncoder", (), {})
    sys.modules.setdefault("requests_toolbelt", requests_toolbelt)

    gladiaio_sdk = types.ModuleType("gladiaio_sdk")
    gladiaio_sdk.GladiaClient = type("GladiaClient", (), {"__init__": lambda self, **kwargs: None})
    sys.modules.setdefault("gladiaio_sdk", gladiaio_sdk)


def _import_providers():
    _install_vendor_sdk_stubs()
    return importlib.import_module("api.providers")


class V1ProviderTests(unittest.TestCase):
    def test_v1_provider_is_registered(self):
        providers = _import_providers()

        provider, variant = providers.get_provider("v1/shisa-ai/shisa-asr-v0.9b")

        self.assertEqual(provider.__class__.__name__, "V1Provider")
        self.assertEqual(variant, "shisa-ai/shisa-asr-v0.9b")

    def test_normalize_base_url_accepts_root_or_v1_url(self):
        _import_providers()
        module = importlib.import_module("api.providers.v1_provider")

        self.assertEqual(module.normalize_base_url("http://localhost:9001"), "http://localhost:9001/v1")
        self.assertEqual(module.normalize_base_url("http://localhost:9001/v1/"), "http://localhost:9001/v1")

    def test_transcription_request_uses_configured_v1_endpoint(self):
        _import_providers()
        module = importlib.import_module("api.providers.v1_provider")
        client_inits = []
        transcription_calls = []

        class FakeTranscriptions:
            def create(self, **kwargs):
                transcription_calls.append(kwargs)
                return types.SimpleNamespace(text=" こんにちは ")

        class FakeAudio:
            transcriptions = FakeTranscriptions()

        class FakeOpenAI:
            def __init__(self, **kwargs):
                client_inits.append(kwargs)
                self.audio = FakeAudio()

        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            audio_file.write(b"RIFF")
            audio_file.flush()
            with patch.dict(os.environ, {"V1_BASE_URL": "http://localhost:9001", "V1_API_KEY": "secret"}):
                with patch.object(module, "OpenAI", FakeOpenAI):
                    text = module.V1Provider().transcribe(
                        "shisa-ai/shisa-asr-v0.9b",
                        audio_file.name,
                        sample={},
                        language="ja",
                        prompt="Transcribe the Japanese audio clip into text.",
                    )

        self.assertEqual(text, "こんにちは")
        self.assertEqual(client_inits, [{"base_url": "http://localhost:9001/v1", "api_key": "secret"}])
        self.assertEqual(len(transcription_calls), 1)
        request = transcription_calls[0]
        self.assertEqual(request["model"], "shisa-ai/shisa-asr-v0.9b")
        self.assertEqual(request["temperature"], 0.0)
        self.assertEqual(request["language"], "ja")
        self.assertEqual(request["prompt"], "Transcribe the Japanese audio clip into text.")
        self.assertNotIn("response_format", request)

    def test_missing_v1_base_url_is_actionable(self):
        _import_providers()
        module = importlib.import_module("api.providers.v1_provider")

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "V1_BASE_URL"):
                module.V1Provider().transcribe("model", "/tmp/audio.wav", sample={})

    def test_result_folders_are_gitignored(self):
        for path in ["results/example.csv", "api/results/example.csv"]:
            result = subprocess.run(
                ["git", "check-ignore", "-q", path],
                cwd=REPO_ROOT,
                check=False,
            )
            self.assertEqual(result.returncode, 0, f"{path} should be gitignored")


if __name__ == "__main__":
    unittest.main()
