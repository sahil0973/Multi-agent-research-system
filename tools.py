import streamlit as st
from langchain.tools import tool
from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup

# ======================
# API KEY
# ======================

TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY missing")

tavily = TavilyClient(api_key=TAVILY_API_KEY)

# ======================
# TOOLS
# ======================

@tool
def web_search(query: str) -> str:
    """Search the web and return top results."""
    try:
        res = tavily.search(query=query, max_results=5)
        results = []

        for r in res.get("results", []):
            results.append(f"{r['title']}\n{r['content']}\n{r['url']}")

        return "\n\n".join(results)

    except Exception as e:
        return f"Error: {e}"


@tool
def scrape_url(url: str) -> str:
    """Scrape text content from a URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)

        soup = BeautifulSoup(res.text, "html.parser")

        for s in soup(["script", "style"]):
            s.extract()

        text = soup.get_text()
        clean = "\n".join([t.strip() for t in text.splitlines() if t.strip()])

        return clean[:3000]

    except Exception as e:
        return f"Error: {e}"