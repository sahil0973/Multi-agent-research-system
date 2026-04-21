import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_urls

# ======================
# LLM
# ======================

def get_llm():
    return ChatGroq(
        groq_api_key=st.secrets.get("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.3
    )

llm = get_llm()

# ======================
# BASIC PIPELINE
# ======================

def run_search(topic):
    return web_search(topic)

def run_reader(search_data):
    urls = search_data["urls"]
    content = scrape_urls(urls)
    return content, urls

# ======================
# WRITER
# ======================

writer_prompt = ChatPromptTemplate.from_template("""
You are a professional research writer.

Topic: {topic}

Data:
{research_data}

Write a structured report.
""")

writer_chain = writer_prompt | llm | StrOutputParser()

# ======================
# CRITIC
# ======================

critic_prompt = ChatPromptTemplate.from_template("""
Improve this report:

{report}
""")

critic_chain = critic_prompt | llm | StrOutputParser()

# ======================
# MULTI-AGENT SYSTEM
# ======================

# Planner
planner_prompt = ChatPromptTemplate.from_template("""
Break this goal into steps (max 5):

{goal}
""")

planner_agent = planner_prompt | llm | StrOutputParser()

# Researcher
research_prompt = ChatPromptTemplate.from_template("""
Research this step:

{step}
""")

research_agent = research_prompt | llm | StrOutputParser()

# Writer Agent
writer_multi_prompt = ChatPromptTemplate.from_template("""
Write report for:

{goal}

Using:
{data}
""")

writer_agent = writer_multi_prompt | llm | StrOutputParser()

# Critic Agent
critic_multi_prompt = ChatPromptTemplate.from_template("""
Improve clarity and quality:

{report}
""")

critic_agent = critic_multi_prompt | llm | StrOutputParser()

# ======================
# RUN MULTI AGENT
# ======================

def run_multi_agent(goal):
    plan = planner_agent.invoke({"goal": goal})

    steps = [s for s in plan.split("\n") if s.strip()]
    collected = ""

    for step in steps[:5]:
        data = research_agent.invoke({"step": step})
        collected += data + "\n\n"

    draft = writer_agent.invoke({
        "goal": goal,
        "data": collected
    })

    final = critic_agent.invoke({
        "report": draft
    })

    return plan, collected, final