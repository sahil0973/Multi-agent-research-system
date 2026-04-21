import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
from tools import web_search, scrape_url

# ======================
# LLM CONFIG (FIXED)
# ======================

def get_llm():
    return ChatGroq(
        groq_api_key=st.secrets.get("GROQ_API_KEY"),
        model_name="llama3-70b-8192",   # ✅ stable model
        temperature=0.3,
        max_tokens=2048
    )

# ======================
# SEARCH AGENT
# ======================

def build_search_agent():
    llm = get_llm()

    tools = [web_search]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    return agent


# ======================
# READER AGENT
# ======================

def build_reader_agent():
    llm = get_llm()

    tools = [scrape_url]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    return agent


# ======================
# WRITER CHAIN
# ======================

def build_writer_chain():
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template("""
You are a professional research writer.

Using the following gathered information, write a well-structured research report.

Topic:
{topic}

Information:
{research_data}

Requirements:
- Clear introduction
- Detailed explanation
- Use headings
- Provide insights
- Conclusion at end
""")

    chain = LLMChain(
        llm=llm,
        prompt=prompt
    )

    return chain


# ======================
# CRITIC CHAIN
# ======================

def build_critic_chain():
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template("""
You are a strict research critic.

Review the following report:

{report}

Evaluate on:
- Accuracy
- Clarity
- Depth
- Structure

Give:
1. Score out of 10
2. Improvements
3. Final verdict
""")

    chain = LLMChain(
        llm=llm,
        prompt=prompt
    )

    return chain