# -*- coding: utf-8 -*-
"""可插拔LLM客户端（OpenAI兼容：DeepSeek/通义/Kimi/vLLM）
不配置 key 时自动降级为抽取式回答。
"""
import os
import time
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
        # 流式输出（默认关闭，保持云端生产行为不变；本地 .env.local-model 置 1 开启）
        self.use_stream = os.getenv("LLM_STREAM", "").strip() in ("1", "true", "True", "yes")
        self.enabled = bool(self.base_url and self.api_key and self.model)
        # 最近一次响应里本地推理服务的五层守卫触发记录（云端 API 无此字段则为空）
        self.last_protections: list = []
        # 最近一次流式调用的首字响应时间（毫秒；非流式为 None）
        self.last_ttft_ms = None

    def _request(self, payload):
        return urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})

    def _complete_stream(self, payload):
        """SSE 流式调用：边收边记首字时间；最终内容以 final_content 为准。

        serve_local_qwen 的流式协议：增量 delta.content 仅作预览，最后一个
        chunk 携带守卫处理后的 final_content / firesage_protections / ttft_ms。
        云端 LLM 无 final_content 时回退为增量拼接（即原始生成内容）。
        """
        t0 = time.time()
        self.last_ttft_ms = None
        parts, final_content, protections = [], None, []
        with urllib.request.urlopen(self._request(payload), timeout=self.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except Exception:
                    continue
                choices = chunk.get("choices") or [{}]
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    if self.last_ttft_ms is None:
                        self.last_ttft_ms = int((time.time() - t0) * 1000)
                    parts.append(delta["content"])
                if chunk.get("final_content") is not None:
                    final_content = chunk["final_content"]
                if chunk.get("firesage_protections"):
                    protections = list(chunk["firesage_protections"])
                if chunk.get("ttft_ms") is not None:
                    # 生成侧 TTFT 更准（不含 HTTP 开销），优先采信
                    self.last_ttft_ms = chunk["ttft_ms"]
        self.last_protections = protections
        return final_content if final_content is not None else "".join(parts)

    def complete(self, system, user):
        if not self.enabled:
            return None
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": self.max_tokens,
        }
        if self.use_stream:
            payload["stream"] = True
        last_error = None
        for attempt in range(max(1, self.max_retries)):
            try:
                if self.use_stream:
                    return self._complete_stream(payload)
                with urllib.request.urlopen(self._request(payload), timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                # 本地 serve_local_qwen 五层守卫触发记录（DeepSeek 等云端无此字段）
                self.last_protections = list(data.get("firesage_protections") or [])
                self.last_ttft_ms = data.get("ttft_ms")
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_error = e
                # 超时/网络类错误重试；4xx 配置类错误重试无意义
                if isinstance(e, urllib.error.HTTPError) and e.code < 500:
                    break
        print(f"[LLM] 调用失败（已重试{self.max_retries}次，降级为抽取式）: {last_error}")
        return None
