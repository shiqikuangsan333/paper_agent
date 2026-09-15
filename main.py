"""
文献调研 Agent v3
================
v2 → v3 的两个升级：

1. 数据源故障转移（failover）：
   search_papers 内部依次尝试 arXiv → Semantic Scholar → Crossref
   任何一个数据源返回结果就停止 —— 工程上应对外部服务不可靠的标准做法

2. 多工具：
   新增 save_report 工具，Agent 自己决定把综述存成 Markdown 文件
   Agent 的能力 = 注册的工具集，现在它有"搜索"和"写文件"两只手了
"""

import os
from dotenv import load_dotenv
load_dotenv()
import json
import time
import ssl
import urllib.request
import urllib.parse
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE  # 学习项目妥协：跳过证书验证（生产环境勿用）

UA = {"User-Agent": "PaperAgent/0.1 (learning project)"}


# ---------- 工具实现：三个数据源 ----------

def search_arxiv(query: str, max_results: int = 5) -> list:
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    })
    url = f"https://export.arxiv.org/api/query?{params}"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.read())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("a:entry", ns):
        papers.append({
            "title": " ".join(entry.find("a:title", ns).text.split()),
            "authors": ", ".join(
                a.find("a:name", ns).text for a in entry.findall("a:author", ns)
            ),
            "published": entry.find("a:published", ns).text[:10],
            "summary": " ".join(entry.find("a:summary", ns).text.split())[:600],
            "link": entry.find("a:id", ns).text,
        })
    return papers


def search_semantic_scholar(query: str, max_results: int = 5) -> list:
    params = urllib.parse.urlencode({
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,year,authors,url",
    })
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
        data = json.loads(resp.read())
    papers = []
    for p in data.get("data", []):
        papers.append({
            "title": p.get("title", "无标题"),
            "authors": ", ".join(a["name"] for a in p.get("authors", [])),
            "published": str(p.get("year", "未知")),
            "summary": (p.get("abstract") or "无摘要")[:600],
            "link": p.get("url", ""),
        })
    return papers


def search_crossref(query: str, max_results: int = 5) -> list:
    params = urllib.parse.urlencode({
        "query": query,
        "rows": max_results,
        "select": "title,author,issued,DOI,URL",
    })
    url = f"https://api.crossref.org/works?{params}"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
        data = json.loads(resp.read())
    papers = []
    for item in data["message"]["items"]:
        papers.append({
            "title": (item.get("title") or ["无标题"])[0],
            "authors": ", ".join(
                a.get("family", "") for a in item.get("author", [])
            ) or "未知",
            "published": str(item.get("issued", {}).get("date-parts", [[None]])[0][0]),
            "summary": "Crossref 不提供免费摘要，请通过链接访问原文",
            "link": item.get("URL", ""),
        })
    return papers


def search_papers(query: str, max_results: int = 5) -> list:
    """故障转移：依次尝试多个数据源，任何一个成功就返回"""
    errors = []
    for source in (search_arxiv, search_semantic_scholar, search_crossref):
        try:
            papers = source(query, max_results)
            if papers:
                print(f"    （数据源 {source.__name__} 成功，{len(papers)} 条）")
                return papers
            errors.append(f"{source.__name__}: 无结果")
        except Exception as e:
            errors.append(f"{source.__name__}: {type(e).__name__}")
        time.sleep(2)  # 对上游服务保持礼貌
    return [{"error": "所有数据源都失败 | " + "; ".join(errors)}]


def save_report(filename: str, content: str) -> str:
    """把综述内容保存为 Markdown 文件，存到项目下的 reports/ 文件夹"""
    os.makedirs("reports", exist_ok=True)
    if not filename.endswith(".md"):
        filename += ".md"
    path = os.path.join("reports", filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已保存到 {path}"


# ---------- 工具说明书（JSON Schema） ----------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "搜索学术论文，内部会自动在多个数据源间切换，获取真实论文的标题、作者、摘要和链接",
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
    },
    {
        "type": "function",
        "function": {
            "name": "save_report",
            "description": "把整理好的综述内容保存为 Markdown 文件，完成任务前必须调用",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "文件名，例如 ris_mmwave_review.md",
                    },
                    "content": {
                        "type": "string",
                        "description": "完整的综述 Markdown 内容",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
]

TOOL_HANDLERS = {
    "search_papers": search_papers,
    "save_report": save_report,
}

SYSTEM_PROMPT = """你是一名科研调研助手。工作流程（必须按顺序执行）：
1. 用 search_papers 搜索用户主题的相关论文（必要时换关键词多搜几轮）
2. 基于搜到的论文摘要，整理结构化中文综述：研究背景、主要技术路线、代表性工作（注明作者/年份/链接）、未来方向
3. 用 save_report 工具把综述保存成 Markdown 文件
4. 最后向用户简要汇报：文件保存路径 + 综述要点摘要
规则：只基于搜索结果总结，不得编造论文；每个重要论点标注来源链接；始终使用中文。
"""

# ---------- Agent 主循环（逻辑不变，工具变多了而已） ----------

def run_agent(topic: str, max_rounds: int = 10) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": topic},
    ]

    for i in range(max_rounds):
        print(f"\n===== 第 {i + 1} 轮 =====")
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
            print(f"[工具调用] {call.function.name}({call.function.arguments})")
            args = json.loads(call.function.arguments)
            result = TOOL_HANDLERS[call.function.name](**args)
            print(f"[工具返回] {len(result) if isinstance(result, list) else 1} 条结果")
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })
            time.sleep(3)

    return "达到最大循环次数，任务未完成"


if __name__ == "__main__":
    topic = input("请输入调研主题（例如：RIS 辅助的毫米波通信）: ")
    print("\n" + "=" * 50)
    print(run_agent(topic))
