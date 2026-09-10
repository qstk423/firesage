# -*- coding: utf-8 -*-
"""探测 8320 现行服务的引用守卫行为（d27 场景复现）。"""
import io
import json
import sys
import urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

URL = "http://127.0.0.1:8320/v1/chat/completions"

# 模拟 pipeline 桥接提示词（d27：模型倾向输出《全名》短键·第N条 格式）
system = ("你是消防法规领域的专业助手「消安智答」。请严格依据提供的法规条款回答问题，不得编造。"
          "回答必须使用以下 JSON 结构（不要输出 JSON 以外的内容）：")
user = (
    "用户问题：高层公共建筑的业主单位应当履行哪些消防安全职责？\n\n"
    "以下为检索到的法规条款：\n"
    "【高层规定·第七条 高层公共建筑的消防安全职责】高层公共建筑的业主单位、使用单位应当履行下列"
    "消防安全职责：（一）遵守消防法律法规……\n\n"
    "【61号令·第六条 单位的消防安全职责】单位的消防安全责任人应当履行下列消防安全职责……\n\n"
    "请基于上述条款作答。"
)
payload = {
    "model": "firesage-qwen25-3b-lora-v2",
    "messages": [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
    "temperature": 0.2,
    "max_tokens": 700,
}
req = urllib.request.Request(
    URL, data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json",
             "Authorization": "Bearer local-firesage"})
with urllib.request.urlopen(req, timeout=180) as resp:
    data = json.loads(resp.read().decode("utf-8"))
print("guards:", data.get("firesage_protections"))
print("latency_ms:", data.get("latency_ms"))
content = json.loads(data["choices"][0]["message"]["content"])
print("basis:")
for b in content.get("basis") or []:
    print(" -", b[:80])
