from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class HashEmbeddings:
    """Dependency-light embedding fallback for teaching and smoke tests."""

    dimensions: int = 512

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokenize(text):
            digest = hashlib.md5(token.encode("utf-8", errors="ignore")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


@dataclass(slots=True)
class OpenAICompatibleEmbeddings:
    """OpenAI-compatible embedding client used for DashScope compatible mode."""

    model: str
    base_url: str
    api_key_env: str = "DASHSCOPE_API_KEY"
    dimensions: int | None = None
    timeout: int = 60
    batch_size: int = 10

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed(texts[start : start + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing environment variable: {self.api_key_env}")
        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            _join_url(self.base_url, "/embeddings"),
            data=data,
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
            raise RuntimeError(f"Embedding request failed: HTTP {exc.code} {detail}") from exc
        payload = json.loads(body)
        items = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in items]


def build_embeddings(
    provider: str = "hash",
    model: str | None = None,
    dimensions: int | None = 512,
    base_url: str | None = None,
    api_key_env: str = "DASHSCOPE_API_KEY",
):
    """按配置创建 embedding 对象。

    生产配置使用 DashScope OpenAI-compatible 接口；hash 只作为无外部依赖的
    教学/测试兜底。上传入库和运行时检索必须使用同一个 embedding 配置。
    """
    provider = (provider or "hash").lower()
    if provider == "hash":
        return HashEmbeddings(dimensions=dimensions or 512)

    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise RuntimeError("Missing dependency: langchain-openai.") from exc
        return OpenAIEmbeddings(model=model or "text-embedding-3-small")

    if provider in {"openai_compatible", "dashscope_compatible"}:
        return OpenAICompatibleEmbeddings(
            model=model or "text-embedding-v3",
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key_env=api_key_env,
            dimensions=dimensions,
        )

    if provider == "dashscope":
        try:
            from langchain_community.embeddings import DashScopeEmbeddings
        except ImportError as exc:
            raise RuntimeError("Missing dependency: langchain-community with DashScope support.") from exc
        return DashScopeEmbeddings(model=model or "text-embedding-v2")

    raise ValueError(f"Unsupported embedding provider: {provider}")


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text)
    bigrams = [text[i : i + 2] for i in range(max(0, len(text) - 1)) if text[i : i + 2].strip()]
    return words + bigrams[:800]
