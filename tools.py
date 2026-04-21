import streamlit as st
from langchain.tools import tool
from tavily import TavilyClient

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
    """Search the web and return results."""
    try:
        res = tavily.search(query=query, max_results=5)
        results = []

        for r in res.get("results", []):
            results.append(f"{r['title']}\n{r['content']}\n{r['url']}")

        return "\n\n".join(results)

    except Exception as e:
        return f"Error: {e}"


@tool
def scrape_url(data: str) -> str:
    """Process text (temporary reader)."""
    return data[:3000]