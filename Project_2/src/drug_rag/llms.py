from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class OpenAICompatibleChat:
    model: str = "qwen-max-latest"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: str = "DASHSCOPE_API_KEY"
    temperature: float = 0.1
    timeout: int = 120

    def invoke(self, messages: list[dict[str, str]]) -> str:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing environment variable: {self.api_key_env}")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        request = urllib.request.Request(
            _join_url(self.base_url, "/chat/completions"),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM request failed: HTTP {exc.code} {detail}") from exc
        payload = json.loads(body)
        return payload["choices"][0]["message"]["content"].strip()


def build_chat_model(
    provider: str = "none",
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str = "DASHSCOPE_API_KEY",
    temperature: float = 0.1,
) -> OpenAICompatibleChat | None:
    """按配置创建受控生成模型；provider=none 时返回 None，问答链会走模板兜底。"""
    provider = (provider or "none").lower()
    if provider in {"none", "off", "false"}:
        return None
    if provider in {"openai_compatible", "dashscope_compatible", "dashscope"}:
        return OpenAICompatibleChat(
            model=model or "qwen-max-latest",
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key_env=api_key_env,
            temperature=temperature,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")
