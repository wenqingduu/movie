import json
import os
import urllib.error
import urllib.request


# 方式 1：推荐把 key 放到环境变量里：
#   export DASHSCOPE_API_KEY="你的百炼 API Key"
#
# 方式 2：也可以直接填在这里测试，用完建议删掉：
#   API_KEY = "你的百炼 API Key"


BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()

MODEL = os.getenv("MULTISHOT_QWEN_MODEL", "qwen3-vl-plus")


def call_qwen(prompt: str) -> str:
    if not API_KEY:
        raise RuntimeError(
            "请先设置 DASHSCOPE_API_KEY 环境变量，或者在脚本里把 API_KEY 填成你的百炼 API Key。"
        )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是一个简洁、可靠的中文助手。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }

    request = urllib.request.Request(
        BASE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {error_body}") from exc

    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    answer = call_qwen("请用一句话介绍一下你自己，并说出 1+1 等于几。")
    print(answer)
