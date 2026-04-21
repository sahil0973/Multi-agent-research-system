import streamlit as st
import time

from agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain
)

# ======================
# PAGE CONFIG
# ======================

st.set_page_config(
    page_title="ResearchMind · Multi-Agent AI",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 ResearchMind")
st.caption("Multi-Agent AI System for Research Automation")

# ======================
# INPUT
# ======================

topic = st.text_input("Enter Research Topic", placeholder="e.g. LLM agents 2025")

run_btn = st.button("Run Research")

# ======================
# PIPELINE UI
# ======================

col1, col2, col3, col4 = st.columns(4)

with col1:
    search_status = st.empty()

with col2:
    reader_status = st.empty()

with col3:
    writer_status = st.empty()

with col4:
    critic_status = st.empty()

# ======================
# RUN PIPELINE
# ======================

if run_btn and topic:

    try:
        # -------- SEARCH --------
        search_status.info("🔍 Searching...")
        search_agent = build_search_agent()

        search_result = search_agent.run(
            f"Find recent, reliable and detailed information about: {topic}"
        )

        search_status.success("✅ Search Done")

        time.sleep(1)

        # -------- READER --------
        reader_status.info("📄 Reading content...")
        reader_agent = build_reader_agent()

        # NOTE: For now passing search_result directly
        # (you can later extract URLs for better pipeline)
        reader_result = reader_agent.run(search_result)

        reader_status.success("✅ Reading Done")

        time.sleep(1)

        # -------- WRITER --------
        writer_status.info("✍️ Writing report...")

        report = writer_chain.run({
            "topic": topic,
            "research_data": reader_result[:3000]  # limit tokens
        })

        writer_status.success("✅ Report Ready")

        # -------- SHOW REPORT --------
        st.subheader("📘 Research Report")
        st.write(report)

        time.sleep(1)

        # -------- CRITIC --------
        critic_status.info("🧠 Reviewing...")

        review = critic_chain.run({
            "report": report[:3000]
        })

        critic_status.success("✅ Review Done")

        # -------- SHOW REVIEW --------
        st.subheader("📊 Critic Review")
        st.write(review)

    except Exception as e:
        import traceback
        st.error("❌ Error occurred")
        st.code(traceback.format_exc())

else:
    st.info("Enter a topic and click 'Run Research'")