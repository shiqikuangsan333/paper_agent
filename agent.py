"""
文献调研 Agent v1
================
核心思想：一个最小 Agent 循环
  用户提问 → 模型思考 → 需要信息就调用工具 → 工具返回结果 → 模型继续 → 给出最终答案

运行前：
  1. pip install -r requirements.txt
  2. 设置环境变量：export DEEPSEEK_API_KEY="你的key"
  3. python agent.py
"""

import os
from dotenv import load_dotenv
load_dotenv()  # 加在 import os 之后
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",  # DeepSeek 兼容 OpenAI 接口格式
)

# ---------- 1. 工具定义 ----------
# Agent 能用什么工具，需要先用"说明书"的形式告诉模型：
# 工具叫什么、干什么用、需要哪些参数

def search_papers(query: str, max_results: int = 5) -> list:
    """搜索论文/技术资料。国内网络环境可能访问不通，可替换成其他搜索源。"""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query + " paper", max_results=max_results))
        return [
            {"title": r["title"], "url": r["href"], "snippet": r["body"]}
            for r in results
        ]
    except Exception as e:
        return [{"error": f"搜索失败: {e}，请根据你已有知识回答"}]


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "搜索与主题相关的论文或技术资料，当需要最新/具体信息时使用",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，英文关键词通常效果更好",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "返回结果数量，默认5",
                    },
                },
                "required": ["query"],
            },
        },
    }
]

TOOL_HANDLERS = {"search_papers": search_papers}  # 工具名 -> 真正的函数

# ---------- 2. 系统提示词 ----------
SYSTEM_PROMPT = """你是一名科研调研助手。工作方式：
1. 先用 search_papers 工具搜索用户主题相关资料
2. 基于搜索结果，整理出结构化综述，包含：研究背景、主要技术路线、代表性工作、未来方向
3. 标注信息来源的出处（url）
4. 如果搜索失败，基于你的知识回答，并明确说明"以下基于模型内部知识"
"""

# ---------- 3. Agent 主循环 ----------
def run_agent(topic: str, max_rounds: int = 6) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": topic},
    ]

    for _ in range(max_rounds):
        # 模型每轮做一次决策：直接回答，还是调用工具
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
        )
        msg = response.choices[0].message
        messages.append(msg)

        # 情况A：模型没调用工具 → 说明它给出了最终答案
        if not msg.tool_calls:
            return msg.content

        # 情况B：模型要求调用工具 → 我们执行，并把结果喂回去
        for call in msg.tool_calls:
            print(f"\n[工具调用] {call.function.name}({call.function.arguments})")
            args = json.loads(call.function.arguments)
            result = TOOL_HANDLERS[call.function.name](**args)
            messages.append({
                "role": "tool",               # 工具结果作为一种特殊消息
                "tool_call_id": call.id,      # 告诉模型这是哪次调用的结果
                "content": json.dumps(result, ensure_ascii=False),
            })

    return "达到最大循环次数，任务未完成（检查是否陷入了循环调用）"


if __name__ == "__main__":
    topic = input("请输入调研主题（例如：RIS 辅助的毫米波通信）: ")
    print("\n" + "=" * 50)
    print(run_agent(topic))
