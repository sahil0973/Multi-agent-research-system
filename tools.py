import streamlit as st
from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup

tavily = TavilyClient(api_key=st.secrets.get("TAVILY_API_KEY"))

def web_search(query):
    res = tavily.search(query=query, max_results=5)
    results, urls = [], []

    for r in res.get("results", []):
        results.append(r["content"])
        urls.append(r["url"])

    return {"content": "\n".join(results), "urls": urls}

def scrape_urls(urls):
    text = ""
    for url in urls[:3]:
        try:
            res = requests.get(url, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")

            for s in soup(["script", "style"]):
                s.extract()

            clean = "\n".join(
                [t.strip() for t in soup.get_text().splitlines() if t.strip()]
            )

            text += clean[:1000] + "\n\n"
        except:
            continue

    return text