"""
文献调研 Agent v4
================
v3 → v4 的三个升级：

1. 多轮对话：Agent 有记忆，可追问/换方向/换主题，不用每次从头开始
2. 搜索缓存：查询结果落盘到 .search_cache.json，重复查询零 API 成本
3. 流式输出：模型回答逐字打印，不再干等

记得把 .search_cache.json 加进 .gitignore！
"""

import os
import json
import time
import ssl
import urllib.request
import urllib.parse
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE  # 学习项目妥协（生产环境勿用）

UA = {"User-Agent": "PaperAgent/0.1 (learning project)"}

# ---------- 搜索缓存 ----------

CACHE_FILE = ".search_cache.json"
_cache = {}

def load_cache():
    global _cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                _cache = json.load(f)
            print(f"[缓存] 已加载 {len(_cache)} 条历史查询")
        except Exception:
            _cache = {}

def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(_cache, f, ensure_ascii=False, indent=1)

# ---------- 三个数据源（实现同 v3） ----------

def search_arxiv(query: str, max_results: int = 5) -> list:
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0, "max_results": max_results, "sortBy": "relevance",
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
                a.find("a:name", ns).text for a in entry.findall("a:author", ns)),
            "published": entry.find("a:published", ns).text[:10],
            "summary": " ".join(entry.find("a:summary", ns).text.split())[:600],
            "link": entry.find("a:id", ns).text,
        })
    return papers


def search_semantic_scholar(query: str, max_results: int = 5) -> list:
    params = urllib.parse.urlencode({
        "query": query, "limit": max_results,
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
        "query": query, "rows": max_results,
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
                a.get("family", "") for a in item.get("author", [])) or "未知",
            "published": str(item.get("issued", {}).get("date-parts", [[None]])[0][0]),
            "summary": "Crossref 不提供免费摘要，请通过链接访问原文",
            "link": item.get("URL", ""),
        })
    return papers


def search_papers(query: str, max_results: int = 5) -> list:
    """带缓存的故障转移搜索"""
    key = f"{query.strip().lower()}::{max_results}"
    if key in _cache:
        print(f"    （缓存命中：{query}）")
        return _cache[key]

    errors = []
    for source in (search_arxiv, search_semantic_scholar, search_crossref):
        try:
            papers = source(query, max_results)
            if papers:
                print(f"    （数据源 {source.__name__} 成功，{len(papers)} 条）")
                _cache[key] = papers      # 只缓存成功结果
                save_cache()
                return papers
            errors.append(f"{source.__name__}: 无结果")
        except Exception as e:
            errors.append(f"{source.__name__}: {type(e).__name__}")
        time.sleep(2)
    return [{"error": "所有数据源都失败 | " + "; ".join(errors)}]


def save_report(filename: str, content: str) -> str:
    os.makedirs("reports", exist_ok=True)
    if not filename.endswith(".md"):
        filename += ".md"
    path = os.path.join("reports", filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已保存到 {path}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "搜索学术论文，内部会自动在多个数据源间切换，获取真实论文的标题、作者、摘要和链接",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "英文搜索关键词"},
                    "max_results": {"type": "integer", "description": "返回数量，默认5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_report",
            "description": "把整理好的综述内容保存为 Markdown 文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "文件名，如 ris_mmwave_review.md"},
                    "content": {"type": "string", "description": "完整综述 Markdown 内容"},
                },
                "required": ["filename", "content"],
            },
        },
    },
]

TOOL_HANDLERS = {"search_papers": search_papers, "save_report": save_report}

SYSTEM_PROMPT = """你是一名科研调研助手。工作流程：
1. 用 search_papers 搜索用户主题的相关论文（必要时换关键词多搜几轮）
2. 基于搜到的论文摘要，整理结构化中文综述：研究背景、主要技术路线、代表性工作（注明作者/年份/链接）、未来方向
3. 用户要求保存时，用 save_report 工具把综述保存成 Markdown 文件
规则：
- 只基于搜索结果总结，不得编造论文；重要论点标注来源链接
- 始终使用中文
- 这是多轮对话：用户可能追加要求（聚焦某个子方向、换主题、补充搜索），基于已有上下文继续工作，不要重复已完成的部分
- 所有中间说明和解释也用中文
"""

# ---------- Agent 主循环（带流式输出） ----------

def run_agent(messages: list, max_rounds: int = 10) -> str:
    for i in range(max_rounds):
        print(f"\n===== 第 {i + 1} 轮 =====")
        # stream=True：回答逐 token 到达，边到边打印
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
            stream=True,
        )

        content = ""
        tool_calls = {}  # index -> 累积中的工具调用
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:                    # 普通文本：直接打印
                print(delta.content, end="", flush=True)
                content += delta.content
            for tc in (delta.tool_calls or []):  # 工具调用：流式累积
                slot = tool_calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function.arguments:
                        slot["arguments"] += tc.function.arguments

        # 把这条（流式拼装出来的）助手消息存进历史，供下一轮使用
        messages.append({
            "role": "assistant",
            "content": content or None,
            "tool_calls": [
                {"id": t["id"], "type": "function",
                 "function": {"name": t["name"], "arguments": t["arguments"]}}
                for t in tool_calls.values()
            ] or None,
        })

        if not tool_calls:           # 收敛：模型没再要工具，回答已打印完
            print()
            return content

        print()  # 换行后开始执行工具
        for t in tool_calls.values():
            print(f"[工具调用] {t['name']}({t['arguments']})")
            args = json.loads(t["arguments"] or "{}")
            result = TOOL_HANDLERS[t["name"]](**args)
            if isinstance(result, list):
                print(f"[工具返回] {len(result)} 条结果")
            messages.append({
                "role": "tool",
                "tool_call_id": t["id"],
                "content": json.dumps(result, ensure_ascii=False),
            })
            time.sleep(3)

    return "达到最大循环次数，任务未完成"


def main():
    load_cache()
    print("=" * 50)
    print("📚 文献调研 Agent v4（输入 exit 退出）")
    print("=" * 50)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = input("\n🧑 你: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("再见！")
            break
        messages.append({"role": "user", "content": user_input})
        answer = run_agent(messages)
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
