import streamlit as st
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from langchain.tools import tool

tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])


@tool
def web_search(query: str) -> str:
    try:
        res = tavily.search(query=query, max_results=5)
        return "\n".join([
            f"{r['title']} - {r['url']}"
            for r in res["results"]
        ])
    except Exception as e:
        return str(e)


@tool
def scrape_url(url: str) -> str:
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script", "style"]):
            tag.decompose()

        return soup.get_text()[:2000]

    except Exception as e:
        return str(e)
