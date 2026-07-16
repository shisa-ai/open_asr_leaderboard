import base64
import mimetypes
import os
from io import BytesIO
from typing import Optional

import requests
from openai import OpenAI

from . import APIProvider, register


DEFAULT_CHAT_PROMPT = "Transcribe the audio clip into text."
DEFAULT_MAX_TOKENS = 1024
DEFAULT_SEED = 42
DEFAULT_TOP_P = 1.0


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


def _extract_chat_text(response: object) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if text:
                parts.append(str(text))
        return "\n".join(parts).strip()
    if content is None:
        return ""
    return str(content).strip()


def _request_mode() -> str:
    mode = os.getenv("V1_REQUEST_MODE", "chat").strip().lower()
    if mode not in {"chat", "transcription"}:
        raise ValueError("V1_REQUEST_MODE must be either 'chat' or 'transcription'.")
    return mode


def _chat_prompt(prompt: Optional[str]) -> str:
    return prompt or os.getenv("V1_CHAT_PROMPT") or DEFAULT_CHAT_PROMPT


def _chat_settings() -> dict:
    request_kwargs = {
        "temperature": 0.0,
        "top_p": float(os.getenv("V1_TOP_P", DEFAULT_TOP_P)),
        "max_tokens": int(os.getenv("V1_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
    }
    seed = os.getenv("V1_SEED")
    request_kwargs["seed"] = DEFAULT_SEED if seed is None else int(seed)
    return request_kwargs


def _mime_type(path: str) -> str:
    guessed, _encoding = mimetypes.guess_type(path)
    return guessed or "audio/wav"


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
        if _request_mode() == "chat":
            return self._transcribe_chat(
                client,
                model_variant,
                audio_file_path,
                sample,
                use_url=use_url,
                prompt=prompt,
            )

        return self._transcribe_audio_transcription(
            client,
            model_variant,
            audio_file_path,
            sample,
            use_url=use_url,
            language=language,
            prompt=prompt,
        )

    def _transcribe_audio_transcription(
        self,
        client: OpenAI,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> str:
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

    def _transcribe_chat(
        self,
        client: OpenAI,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        prompt: Optional[str] = None,
    ) -> str:
        content = [{"type": "text", "text": _chat_prompt(prompt)}]
        if use_url:
            content.append(
                {
                    "type": "audio_url",
                    "audio_url": {"url": sample["row"]["audio"][0]["src"]},
                }
            )
        else:
            if audio_file_path is None:
                raise ValueError("audio_file_path is required when use_url is False.")
            with open(audio_file_path, "rb") as audio_file:
                audio_data = base64.b64encode(audio_file.read()).decode("ascii")
            content.append(
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": f"data:{_mime_type(audio_file_path)};base64,{audio_data}"
                    },
                }
            )

        request_kwargs = {
            "model": model_variant,
            "messages": [{"role": "user", "content": content}],
            **_chat_settings(),
        }

        try:
            response = client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            if "seed" not in str(exc).lower():
                raise
            request_kwargs.pop("seed", None)
            response = client.chat.completions.create(**request_kwargs)
        return _extract_chat_text(response)
