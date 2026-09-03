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
        self.enabled = bool(self.base_url and self.api_key and self.model)

    def complete(self, system, user):
        if not self.enabled:
            return None
        try:
            url = self.base_url.rstrip("/") + "/chat/completions"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.3,
            }
            req = urllib.request.Request(url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.api_key}"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[LLM] 调用失败（降级为抽取式）: {e}")
            return None
