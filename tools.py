import streamlit as st
from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup

TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# ======================
# SEARCH
# ======================

def web_search(query: str):
    res = tavily.search(query=query, max_results=5)

    results = []
    urls = []

    for r in res.get("results", []):
        results.append(r["content"])
        urls.append(r["url"])

    return {"content": "\n".join(results), "urls": urls}

# ======================
# SCRAPE
# ======================

def scrape_urls(urls):
    all_text = ""

    for url in urls[:3]:
        try:
            res = requests.get(url, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")

            for s in soup(["script", "style"]):
                s.extract()

            text = soup.get_text()
            clean = "\n".join([t.strip() for t in text.splitlines() if t.strip()])

            all_text += clean[:1000] + "\n\n"

        except:
            continue

    return all_text

# ======================
# IMAGE (Unsplash)
# ======================

def image_search(query):
    return [f"https://source.unsplash.com/featured/?{query}"]

# ======================
# VIDEO (YouTube)
# ======================

def video_search(query):
    return f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"