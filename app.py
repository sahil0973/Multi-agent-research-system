import streamlit as st
import time

from agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain
)

st.set_page_config(page_title="ResearchMind", page_icon="🔬", layout="wide")

st.title("🔬 ResearchMind")
st.caption("Multi-Agent AI Research System")

topic = st.text_input("Enter Topic")

if st.button("Run") and topic:

    try:
        # SEARCH
        st.info("Searching...")
        search_agent = build_search_agent()
        search_result = search_agent.run(
            f"Find detailed and recent info about: {topic}"
        )

        st.success("Search Done")

        # READER
        st.info("Reading...")
        reader_agent = build_reader_agent()
        reader_result = reader_agent.run(search_result)

        st.success("Reading Done")

        # WRITER
        st.info("Writing...")

        report = writer_chain.invoke({
            "topic": topic,
            "research_data": reader_result[:3000]
        })

        st.subheader("Report")
        st.write(report)

        # CRITIC
        st.info("Reviewing...")

        review = critic_chain.invoke({
            "report": report[:3000]
        })

        st.subheader("Review")
        st.write(review)

        st.success("Done ✅")

    except Exception as e:
        import traceback
        st.error("Error occurred")
        st.code(traceback.format_exc())