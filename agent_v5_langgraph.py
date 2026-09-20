"""
文献调研 Agent v5 —— LangGraph 重写版
=====================================
学习目标：亲手体验框架接管了什么、留下了什么。

对照 v4 看：
- 被框架接管的：主循环、tool_calls 分片拼装、tool_call_id 对账、消息历史、终止判断
- 依然属于你的：三个数据源实现、缓存、故障转移、prompt、工具的业务语义
"""

import os
import json
import time
import ssl
import urllib.request
import urllib.parse
from typing import Annotated, TypedDict

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI          # DeepSeek 兼容 OpenAI 协议
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# ========== 业务层：与 v4 完全相同（框架不管这些） ==========

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

UA = {"User-Agent": "PaperAgent/0.1 (learning project)"}

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


def search_arxiv(query: str, max_results: int = 5) -> list:
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}", "start": 0,
        "max_results": max_results, "sortBy": "relevance",
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


# ========== 工具层：@tool 装饰器 ==========
# 对比 v4：不用手写 JSON Schema 了！
# 框架从【类型注解 + docstring】自动生成工具说明书：
#   - 参数类型来自注解（query: str）
#   - 默认值来自函数签名（max_results: int = 5）
#   - description 来自 docstring 第一行

@tool
def search_papers(query: str, max_results: int = 5) -> list:
    """搜索学术论文，内部自动在多个数据源间切换，获取标题、作者、摘要和链接。

    Args:
        query: 英文搜索关键词
        max_results: 返回论文数量，默认 5
    """
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
                _cache[key] = papers
                save_cache()
                return papers
            errors.append(f"{source.__name__}: 无结果")
        except Exception as e:
            errors.append(f"{source.__name__}: {type(e).__name__}")
        time.sleep(2)
    return [{"error": "所有数据源都失败 | " + "; ".join(errors)}]


@tool
def save_report(filename: str, content: str) -> str:
    """把整理好的综述内容保存为 Markdown 文件。

    Args:
        filename: 文件名，例如 ris_mmwave_review.md
        content: 完整的综述 Markdown 内容
    """
    os.makedirs("reports", exist_ok=True)
    if not filename.endswith(".md"):
        filename += ".md"
    path = os.path.join("reports", filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已保存到 {path}"


tools = [search_papers, save_report]

# ========== 编排层：被框架接管的部分 ==========

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    streaming=True,
)
llm_with_tools = llm.bind_tools(tools)   # 等价于 v4 里手动传 tools=TOOLS

class State(TypedDict):
    # Agent 的全部记忆就是一个状态对象；
    # add_messages 是归并器：新消息自动追加而不是覆盖（对比 v4 手动 append）
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    # 这个"节点函数"等价于 v4 主循环里的一次 API 调用
    # 注意：流式拼装、tool_calls 累积、消息格式——全都不见了
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))
# ToolNode = v4 里"执行工具+对账+回写结果"那一整段的框架版

graph_builder.add_edge(START, "chatbot")
# tools_condition = v4 里"if not msg.tool_calls: return" 的框架版：
# 模型要工具 → 走 tools 节点；不要 → 走到 END
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")   # 工具结果喂回模型，闭环

memory = MemorySaver()   # 内存检查点：跨轮对话状态自动持久化
graph = graph_builder.compile(checkpointer=memory)

SYSTEM_PROMPT = """你是一名科研调研助手。工作流程：
1. 用 search_papers 搜索用户主题的相关论文（必要时换关键词多搜几轮）
2. 基于搜到的论文摘要，整理结构化中文综述：研究背景、主要技术路线、代表性工作（注明作者/年份/链接）、未来方向
3. 用户要求保存时，用 save_report 工具把综述保存成 Markdown 文件
规则：
- 只基于搜索结果总结，不得编造论文；重要论点标注来源链接
- 始终使用中文
- 这是多轮对话：用户可能追加要求，基于已有上下文继续工作，不要重复已完成的部分
"""


def main():
    load_cache()
    print("=" * 50)
    print("📚 文献调研 Agent v5 · LangGraph 版（输入 exit 退出）")
    print("=" * 50)

    # thread_id：检查点按会话隔离。换 id = 开新对话（记忆清零）
    config = {"configurable": {"thread_id": "demo"}}

    first = True
    while True:
        user_input = input("\n🧑 你: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("再见！")
            break

        if first:
            graph_input = {"messages": [SystemMessage(content=SYSTEM_PROMPT),
                                        HumanMessage(content=user_input)]}
            first = False
        else:
            graph_input = {"messages": [HumanMessage(content=user_input)]}

        # stream_mode="messages"：逐 token 流式输出（等价 v4 的手动拼装）
        for msg, metadata in graph.stream(graph_input, config, stream_mode="messages"):
            if metadata["langgraph_node"] == "chatbot" and msg.content:
                print(msg.content, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
