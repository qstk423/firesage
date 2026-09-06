# -*- coding: utf-8 -*-
"""可插拔LLM客户端（OpenAI兼容：DeepSeek/通义/Kimi/vLLM）
不配置 key 时自动降级为抽取式回答。
"""
import os
import urllib.request
import json


def _load_local_env():
    """加载 backend/.env.local；已有系统环境变量拥有更高优先级。"""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env.local")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as source:
        for raw_line in source:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip("\"'")
            if key.startswith("LLM_"):
                os.environ.setdefault(key, value)


class LLMClient:
    def __init__(self):
        _load_local_env()
        self.base_url = os.getenv("LLM_BASE_URL", "")
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.model = os.getenv("LLM_MODEL", "")
        self.timeout = int(os.getenv("LLM_TIMEOUT", "30"))
        # 演示默认只重试 1 次，避免超时后再等一轮把体感拖到十几秒
        self.max_retries = int(os.getenv("LLM_MAX_RETRIES", "1"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "700") or "700")
        self.enabled = bool(self.base_url and self.api_key and self.model)

    def complete(self, system, user):
        if not self.enabled:
            return None
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": self.max_tokens,
        }
        last_error = None
        for attempt in range(max(1, self.max_retries)):
            try:
                req = urllib.request.Request(url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {self.api_key}"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_error = e
                # 超时/网络类错误重试；4xx 配置类错误重试无意义
                if isinstance(e, urllib.error.HTTPError) and e.code < 500:
                    break
        print(f"[LLM] 调用失败（已重试{self.max_retries}次，降级为抽取式）: {last_error}")
        return None
