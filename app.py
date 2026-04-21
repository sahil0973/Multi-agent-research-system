import streamlit as st
import traceback

from agents import (
    run_search,
    run_reader,
    writer_chain,
    critic_chain,
    run_multi_agent
)

st.set_page_config(page_title="ResearchMind AI", page_icon="🧠")

st.title("🧠 ResearchMind AI")

# ======================
# MODE
# ======================

mode = st.selectbox("Mode", ["Chat", "Research", "Multi-Agent"])

topic = st.text_input("Enter topic")

# ======================
# RUN
# ======================

if st.button("Run") and topic:

    try:

        # ======================
        # CHAT MODE
        # ======================
        if mode == "Chat":
            st.write("💬 Chat mode (basic response)")
            st.write("Use Research or Multi-Agent for full features")

        # ======================
        # RESEARCH MODE
        # ======================
        elif mode == "Research":

            st.info("🔍 Searching...")
            search_data = run_search(topic)

            st.info("📄 Reading...")
            reader_text, urls = run_reader(search_data)

            st.info("✍️ Writing...")
            report = writer_chain.invoke({
                "topic": topic,
                "research_data": reader_text[:2000]
            })

            st.subheader("📘 Report")
            st.write(report)

            st.info("🧠 Reviewing...")
            review = critic_chain.invoke({
                "report": report
            })

            st.subheader("📊 Review")
            st.write(review)

        # ======================
        # MULTI-AGENT MODE
        # ======================
        elif mode == "Multi-Agent":

            st.info("🤖 Running multi-agent system...")

            plan, data, result = run_multi_agent(topic)

            st.subheader("🧠 Plan")
            st.write(plan)

            st.subheader("🔍 Research Data")
            st.write(data[:1500])

            st.subheader("📘 Final Output")
            st.write(result)

            st.success("✅ Completed")

    except Exception:
        st.error("❌ Error occurred")
        st.code(traceback.format_exc())