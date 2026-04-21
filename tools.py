import streamlit as st
from langchain.tools import tool
from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup

# ======================
# API KEYS (SAFE LOAD)
# ======================

# Works for both Streamlit Cloud + local .env
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in Streamlit secrets")

tavily = TavilyClient(api_key=TAVILY_API_KEY)

# ======================
# TOOLS
# ======================

@tool
def web_search(query: str) -> str:
    """
    Search the web using Tavily API and return top results.
    Input: query (string)
    Output: summarized search results
    """
    try:
        res = tavily.search(query=query, max_results=5)
        results = []

        for r in res.get("results", []):
            results.append(f"{r['title']}\n{r['content']}\n{r['url']}\n")

        return "\n\n".join(results)

    except Exception as e:
        return f"Web search error: {str(e)}"


@tool
def scrape_url(url: str) -> str:
    """
    Scrape and extract readable text content from a given URL.
    Input: url (string)
    Output: extracted text content from webpage
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return f"Failed to fetch URL: {response.status_code}"

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts & styles
        for script in soup(["script", "style"]):
            script.extract()

        text = soup.get_text(separator="\n")

        # Clean text
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned_text = "\n".join(lines)

        return cleaned_text[:5000]  # limit output size

    except Exception as e:
        return f"Scraping error: {str(e)}"