import streamlit as st
import traceback

from agents import (
    run_search,
    run_reader,
    writer_chain,
    critic_chain
)

st.set_page_config(page_title="ResearchMind", page_icon="🔬")

st.title("🔬 ResearchMind")
st.caption("Fast Multi-Step AI Research System")

topic = st.text_input("Enter Topic")

if st.button("Run") and topic:

    try:
        st.info("🔍 Searching...")
        search_result = run_search(topic)
        st.success("Search Done")

        st.info("📄 Processing...")
        reader_result = run_reader(search_result[:1000])
        st.success("Processing Done")

        st.info("✍️ Writing...")
        report = writer_chain.invoke({
            "topic": topic,
            "research_data": reader_result
        })

        st.subheader("📘 Report")
        st.write(report)

        st.info("🧠 Reviewing...")
        review = critic_chain.invoke({
            "report": report
        })

        st.subheader("📊 Review")
        st.write(review)

        st.success("✅ Completed")

    except Exception:
        st.error("❌ Error occurred")
        st.code(traceback.format_exc())