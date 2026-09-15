"""
文献调研 Agent v2
================
v1 → v2 的变化：搜索源从 DuckDuckGo 换成 arXiv 官方 API
- 用 urllib（标准库）发 HTTP 请求，解析返回的 XML
- 论文有真实标题、作者、日期、摘要、链接，综述可验证、可溯源

对比 v1 看 diff，重点理解：工具就是一个普通的 Python 函数，
Agent 的能力 = 你给它注册了什么样的工具
"""
import time   # 顶部加
import ssl   # 文件顶部加
import os
from dotenv import load_dotenv
load_dotenv()  # 加在 import os 之后
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

# ---------- 1. 工具：arXiv 论文搜索 ----------

def search_arxiv(query: str, max_results: int = 5) -> list:
    """调用 arXiv 官方 API 搜索论文，返回标题/作者/日期/摘要/链接"""
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    })
    url = f"https://export.arxiv.org/api/query?{params}"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    # try:
    #     with urllib.request.urlopen(url, timeout=15, context=ctx) as resp:
    #         data = resp.read()
    # except Exception as e:
    #     return [{"error": f"arXiv 请求失败: {e}"}]
    # 替换原来的 try/except 部分：
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PaperAgent/0.1 (learning project)"},
    )
    data = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                data = resp.read()
            break
        except Exception as e:
            if attempt == 2:
                return [{"error": f"arXiv 请求失败: {e}"}]
            time.sleep(5 * (attempt + 1))

    # arXiv 返回的是 Atom 格式的 XML，用标准库解析
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(data)
    papers = []
    for entry in root.findall("a:entry", ns):
        papers.append({
            "title": entry.find("a:title", ns).text.strip().replace("\n", " "),
            "authors": ", ".join(
                a.find("a:name", ns).text for a in entry.findall("a:author", ns)
            ),
            "published": entry.find("a:published", ns).text[:10],
            "summary": " ".join(entry.find("a:summary", ns).text.split())[:600],
            "link": entry.find("a:id", ns).text,
        })
    if not papers:
        return [{"error": f"未找到与 '{query}' 相关的论文，换个关键词试试"}]
    return papers


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_arxiv",
            "description": "在 arXiv 论文库中搜索学术论文，获取真实论文的标题、作者、摘要和链接",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "英文搜索关键词，可用空格分隔多个词",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "返回论文数量，默认5，最大10",
                    },
                },
                "required": ["query"],
            },
        },
    }
]

TOOL_HANDLERS = {"search_arxiv": search_arxiv}

# ---------- 2. 系统提示词 ----------

SYSTEM_PROMPT = """你是一名科研调研助手。工作方式：
1. 先用 search_arxiv 工具搜索用户主题的相关论文（必要时换关键词多搜几轮）
2. 基于搜到的论文摘要，整理结构化综述：研究背景、主要技术路线、代表性工作（注明作者/年份）、未来方向
3. 每个重要论点必须标注来源论文的链接
4. 只做基于搜索结果的总结，不要编造论文
5. 无论任何情况，始终使用中文回复用户
"""

# ---------- 3. Agent 主循环（和 v1 完全一样） ----------

def run_agent(topic: str, max_rounds: int = 8) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": topic},
    ]

    for _ in range(max_rounds):
        print(f"\n===== 第 {_ + 1} 轮 =====")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content

        for call in msg.tool_calls:
            print(f"\n[工具调用] {call.function.name}({call.function.arguments})")
            args = json.loads(call.function.arguments)
            result = TOOL_HANDLERS[call.function.name](**args)
            time.sleep(3)   # 对 arXiv 友好一点，避免并行burst触发限流
            print(f"[工具返回] {len(result)} 条结果")  # ← 加这行
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

    return "达到最大循环次数，任务未完成"


if __name__ == "__main__":
    topic = input("请输入调研主题（例如：RIS 辅助的毫米波通信）: ")
    print("\n" + "=" * 50)
    print(run_agent(topic))
