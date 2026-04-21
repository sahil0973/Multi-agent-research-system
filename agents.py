import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_urls, image_search, video_search

# ======================
# LLM
# ======================

def get_llm():
    return ChatGroq(
        groq_api_key=st.secrets.get("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=1024
    )

llm = get_llm()

# ======================
# FIELD DETECTION
# ======================

field_prompt = ChatPromptTemplate.from_template("""
Classify the topic into one field:
AI, Technology, Physics, Biology, Medicine, Finance, Education, Environment, Other

Topic: {topic}

Return only the field name.
""")

field_chain = field_prompt | llm | StrOutputParser()

def detect_field(topic):
    return field_chain.invoke({"topic": topic}).strip()

# ======================
# PIPELINE
# ======================

def run_search(topic):
    return web_search(topic)

def run_reader(search_data):
    urls = search_data["urls"]
    content = scrape_urls(urls)
    return content, urls

def run_images(topic):
    return image_search(topic)

def run_videos(topic):
    return video_search(topic)

# ======================
# WRITER
# ======================

writer_prompt = ChatPromptTemplate.from_template("""
You are an expert in {field}.

Write a professional research report.

Topic:
{topic}

Research Data:
{research_data}

Instructions:
- Domain-specific explanation
- Clear structure
- Insights
- Conclusion

Add "Sources" section.
""")

writer_chain = writer_prompt | llm | StrOutputParser()

# ======================
# CRITIC
# ======================

critic_prompt = ChatPromptTemplate.from_template("""
Review the report:

{report}

Evaluate:
- Accuracy
- Clarity
- Depth
- Structure

Give score /10 and improvements.
""")

critic_chain = critic_prompt | llm | StrOutputParser()