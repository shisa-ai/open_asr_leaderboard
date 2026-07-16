import os
from io import BytesIO
from typing import Optional

import requests
from openai import OpenAI

from . import APIProvider, register


def normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def _extract_transcription_text(response: object) -> str:
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, dict):
        return str(response.get("text", "") or "").strip()
    return str(getattr(response, "text", "") or "").strip()


@register("v1")
class V1Provider(APIProvider):
    def _client(self) -> OpenAI:
        base_url = os.getenv("V1_BASE_URL") or os.getenv("V1_ENDPOINT")
        if not base_url:
            raise ValueError(
                "V1_BASE_URL environment variable not set. "
                "Set it to an OpenAI-compatible endpoint with or without /v1."
            )
        return OpenAI(
            base_url=normalize_base_url(base_url),
            api_key=os.getenv("V1_API_KEY") or "not-needed",
        )

    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> str:
        client = self._client()
        request_kwargs = {
            "model": model_variant,
            "temperature": 0.0,
        }
        if language:
            request_kwargs["language"] = language
        if prompt:
            request_kwargs["prompt"] = prompt

        if use_url:
            response = requests.get(sample["row"]["audio"][0]["src"], timeout=300)
            response.raise_for_status()
            audio_data = BytesIO(response.content)
            audio_data.name = "audio.wav"
            response = client.audio.transcriptions.create(
                file=audio_data,
                **request_kwargs,
            )
        else:
            if audio_file_path is None:
                raise ValueError("audio_file_path is required when use_url is False.")
            with open(audio_file_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    file=audio_file,
                    **request_kwargs,
                )

        return _extract_transcription_text(response)
