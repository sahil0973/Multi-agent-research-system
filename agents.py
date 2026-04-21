import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url

# ======================
# LLM CONFIG (UPDATED)
# ======================

def get_llm():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY missing")

    # ✅ Use currently supported model
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama3-8b-8192",
        temperature=0.3
    )

llm = get_llm()

# ======================
# SEARCH
# ======================

def run_search(topic: str) -> str:
    query = f"Find detailed, recent, reliable information about: {topic}"
    return web_search.invoke(query)

# ======================
# READER
# ======================

def run_reader(data: str) -> str:
    return scrape_url.invoke(data)

# ======================
# WRITER
# ======================

writer_prompt = ChatPromptTemplate.from_template("""
You are a professional research writer.

Topic:
{topic}

Research Data:
{research_data}

Write a structured report:
- Introduction
- Key concepts
- Deep explanation
- Insights
- Conclusion
""")

writer_chain = writer_prompt | llm | StrOutputParser()

# ======================
# CRITIC
# ======================

critic_prompt = ChatPromptTemplate.from_template("""
You are a strict research critic.

Review the report:

{report}

Evaluate:
- Accuracy
- Clarity
- Depth
- Structure

Provide:
1. Score out of 10
2. Improvements
3. Final verdict
""")

critic_chain = critic_prompt | llm | StrOutputParser()