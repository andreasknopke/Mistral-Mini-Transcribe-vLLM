"""Optionaler LLM-Client für Transcript-Review."""

import os
from typing import Optional

import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))

OPENAI_BASE_URL = os.getenv("LLM_OPENAI_BASE_URL", "").rstrip("/")
OPENAI_MODEL = os.getenv("LLM_OPENAI_MODEL", "")
OPENAI_API_KEY = os.getenv("LLM_OPENAI_API_KEY", "")


class LLMClient:
    """Client für optionale Korrektur über Ollama oder OpenAI-kompatible APIs."""

    def __init__(self) -> None:
        self._backend = self._detect_backend()
        if self._backend:
            print(f"--- LLM Client: Backend '{self._backend}' konfiguriert ---")
        else:
            print("--- LLM Client: Kein Backend verfügbar (optional) ---")

    def _detect_backend(self) -> Optional[str]:
        if OPENAI_BASE_URL and OPENAI_MODEL:
            try:
                response = requests.get(
                    f"{OPENAI_BASE_URL}/models",
                    timeout=5,
                    headers=self._openai_headers(),
                )
                if response.status_code == 200:
                    return "openai"
            except Exception:
                pass

        if OLLAMA_BASE_URL and OLLAMA_MODEL:
            try:
                response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
                if response.status_code == 200:
                    return "ollama"
            except Exception:
                pass

        return None

    def is_available(self) -> bool:
        return self._backend is not None

    def review_transcription(self, text: str, language: str = "German") -> str:
        if not text.strip() or not self._backend:
            return text

        prompt = (
            "Du bist ein Korrektur-LLM für Transkriptionen. "
            "Verbessere nur offensichtliche ASR-Fehler, Rechtschreibung und Interpunktion. "
            "Erfinde keine Inhalte, kürze nicht und ändere keine Bedeutung. "
            f"Sprache: {language}.\n\n"
            f"Transkript:\n{text}"
        )

        if self._backend == "openai":
            return self._query_openai(prompt)
        if self._backend == "ollama":
            return self._query_ollama(prompt)
        return text

    def _query_openai(self, prompt: str) -> str:
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Korrigiere Transkriptionsfehler minimal-invasiv.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }
        response = requests.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            json=payload,
            timeout=LLM_TIMEOUT,
            headers=self._openai_headers(),
        )
        response.raise_for_status()
        data = response.json()
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
            or prompt
        )

    def _query_ollama(self, prompt: str) -> str:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=LLM_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip() or prompt

    def _openai_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if OPENAI_API_KEY:
            headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
        return headers
