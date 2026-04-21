import streamlit as st
import traceback

from agents import (
    run_search,
    run_reader,
    writer_chain,
    critic_chain
)

# ======================
# CONFIG
# ======================

st.set_page_config(page_title="ResearchMind", page_icon="🔬")

st.title("🔬 ResearchMind")
st.caption("Fast Multi-Step AI Research System")

# ======================
# HELPER
# ======================

def safe_text(text, limit=2000):
    if not text:
        return ""
    return text[:limit]

# ======================
# INPUT
# ======================

topic = st.text_input("Enter Topic")

# ======================
# RUN PIPELINE
# ======================

if st.button("Run") and topic:

    try:
        # -------- SEARCH --------
        st.info("🔍 Searching...")
        search_result = run_search(topic)
        st.success("Search Done")

        # -------- READER --------
        st.info("📄 Processing...")
        reader_result = run_reader(safe_text(search_result, 1000))
        st.success("Processing Done")

        # -------- WRITER --------
        st.info("✍️ Writing...")

        report = writer_chain.invoke({
            "topic": topic,
            "research_data": safe_text(reader_result, 2000)
        })

        st.subheader("📘 Report")
        st.write(report)

        # -------- CRITIC --------
        st.info("🧠 Reviewing...")

        review = critic_chain.invoke({
            "report": safe_text(report, 2000)
        })

        st.subheader("📊 Review")
        st.write(review)

        st.success("✅ Completed")

    except Exception:
        st.error("❌ Error occurred")
        st.code(traceback.format_exc())