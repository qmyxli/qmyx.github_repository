"""LLM 客户端 - OpenAI 兼容接口封装

通过 base_url 支持任意 OpenAI 兼容的大模型服务（DeepSeek、通义千问、智谱 GLM、
Kimi、OpenAI 等）。对外暴露 `invoke(prompt)`，返回带 `.content` 属性的响应对象，
与 ReAct 智能体、NL2SQL 引擎现有的调用位保持一致。

配置从环境变量（.env）读取：
    LLM_API_KEY     接口密钥（必填，未配置时 is_configured 为 False）
    LLM_BASE_URL    接口地址（可选，默认使用 openai SDK 的官方地址）
    LLM_MODEL       模型名（可选，默认 gpt-4o-mini）

未配置密钥时，调用方应降级为内置规则逻辑，而不是直接调用本客户端。
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv
# 加载当前目录下 .env 文件
load_dotenv()

logger = logging.getLogger(__name__)


class LLMResponse:
    """统一的大模型返回结构，供 ReAct / NL2SQL 调用位使用"""

    def __init__(self, content: str):
        self.content = content


class LLMClient:
    """OpenAI 兼容的大模型客户端"""

    def __init__(self, api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None):
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self._client = None

    @property
    def is_configured(self) -> bool:
        """是否已配置密钥（决定是否启用大模型，还是降级为规则逻辑）"""
        return bool(self.api_key)

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            return None
        # 延迟导入，未安装 openai 时不影响其余功能
        from openai import OpenAI

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def invoke(self, prompt: str, **kwargs) -> LLMResponse:
        """发送请求，返回带 `.content` 属性的响应对象

        Args:
            prompt: 完整提示词（现有调用位会把系统提示与用户输入拼成一个字符串）
            **kwargs: 透传给 openai 的 chat.completions.create（如 temperature）
        """
        client = self._ensure_client()
        if client is None:
            raise RuntimeError("未配置 LLM_API_KEY，无法调用大模型")

        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            content = resp.choices[0].message.content or ""
            return LLMResponse(content.strip())
        except Exception as e:
            logger.error("大模型调用失败: %s", e)
            raise


if __name__ == "__main__":
    # 测试用例：未配置密钥时 is_configured 为 False，且 invoke 抛出可预期异常
    import os
    os.environ.pop("LLM_API_KEY", None)
    os.environ.pop("OPENAI_API_KEY", None)
    client = LLMClient()
    assert client.is_configured is False, "未配置密钥时应为未配置状态"
    try:
        client.invoke("你好")
        raise AssertionError("未配置密钥时不应调用成功")
    except RuntimeError:
        pass
    print("测试通过: 未配置密钥时正确降级为不可调用状态")
