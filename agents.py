import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import initialize_agent, AgentType
from tools import web_search, scrape_url

# ======================
# LLM CONFIG
# ======================

def get_llm():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found")

    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama3-70b-8192",
        temperature=0.3
    )

# ======================
# SEARCH AGENT
# ======================

def build_search_agent():
    return initialize_agent(
        tools=[web_search],
        llm=get_llm(),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

# ======================
# READER AGENT
# ======================

def build_reader_agent():
    return initialize_agent(
        tools=[scrape_url],
        llm=get_llm(),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

# ======================
# WRITER CHAIN
# ======================

writer_prompt = ChatPromptTemplate.from_template("""
You are a professional research writer.

Topic:
{topic}

Research Data:
{research_data}

Write a detailed report with:
- Introduction
- Headings
- Deep explanation
- Insights
- Conclusion
""")

writer_chain = writer_prompt | get_llm() | StrOutputParser()

# ======================
# CRITIC CHAIN
# ======================

critic_prompt = ChatPromptTemplate.from_template("""
You are a strict critic.

Report:
{report}

Evaluate:
- Accuracy
- Clarity
- Depth
- Structure

Give:
1. Score /10
2. Improvements
3. Verdict
""")

critic_chain = critic_prompt | get_llm() | StrOutputParser()