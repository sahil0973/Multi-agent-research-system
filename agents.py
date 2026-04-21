import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
from tools import web_search, scrape_url

# ======================
# LLM CONFIG
# ======================

def get_llm():
    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in Streamlit secrets")

    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama3-70b-8192",
        temperature=0.3,
        max_tokens=2048
    )

# ======================
# SEARCH AGENT
# ======================

def build_search_agent():
    llm = get_llm()

    agent = initialize_agent(
        tools=[web_search],
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

    agent = initialize_agent(
        tools=[scrape_url],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    return agent

# ======================
# WRITER CHAIN
# ======================

writer_prompt = ChatPromptTemplate.from_template("""
You are a professional research writer.

Write a detailed research report.

Topic:
{topic}

Research Data:
{research_data}

Instructions:
- Clear introduction
- Use headings
- Deep explanation
- Add insights
- Conclusion at end
""")

writer_chain = LLMChain(
    llm=get_llm(),
    prompt=writer_prompt
)

# ======================
# CRITIC CHAIN
# ======================

critic_prompt = ChatPromptTemplate.from_template("""
You are a strict research critic.

Review the report below:

{report}

Evaluate:
- Accuracy
- Clarity
- Depth
- Structure

Provide:
1. Score (out of 10)
2. Improvements
3. Final verdict
""")

critic_chain = LLMChain(
    llm=get_llm(),
    prompt=critic_prompt
)