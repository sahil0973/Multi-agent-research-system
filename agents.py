import streamlit as st
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 🔁 Choose: "groq" | "gemini"
PROVIDER = "groq"

# ======================
# LLM SETUP
# ======================

if PROVIDER == "groq":
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0,
        api_key=st.secrets["GROQ_API_KEY"]
    )

elif PROVIDER == "gemini":
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

# ======================
# TOOLS
# ======================

from tools import web_search, scrape_url

# ======================
# AGENTS
# ======================

def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt="""
Find 3–5 reliable sources.
Return URLs with short descriptions.
Avoid repetition.
"""
    )

def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt="""
Extract key insights from URL.
Keep concise and factual.
"""
    )

# ======================
# WRITER
# ======================

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research writer."),
    ("human", """Topic: {topic}

Research:
{research}

Write:
- Introduction
- Key Findings
- Conclusion
- Sources"""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# ======================
# CRITIC
# ======================

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a strict critic."),
    ("human", """{report}

Give:
Score, strengths, weaknesses, verdict"""),
])

critic_chain = critic_prompt | llm | StrOutputParser()
