# -*- coding: utf-8 -*-
"""可插拔LLM客户端（OpenAI兼容：DeepSeek/通义/Kimi/vLLM）
不配置 key 时自动降级为抽取式回答。
"""
import os
import urllib.request
import json


class LLMClient:
    def __init__(self):
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