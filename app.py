import streamlit as st
import traceback
import matplotlib.pyplot as plt
from collections import Counter
import re

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
# SESSION MEMORY
# ======================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ======================
# DISPLAY CHAT
# ======================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ======================
# INPUT
# ======================

user_input = st.chat_input("Ask anything...")

# ======================
# HELPERS
# ======================

def get_keywords(text):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    return Counter(words).most_common(8)

def plot_chart(data):
    labels = [x[0] for x in data]
    values = [x[1] for x in data]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    plt.xticks(rotation=45)

    st.pyplot(fig)

# ======================
# MAIN LOGIC
# ======================

if user_input:

    try:
        # Save user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):

            # Decide mode automatically
            is_complex = len(user_input.split()) > 5

            # ======================
            # SIMPLE RESPONSE
            # ======================
            if not is_complex:
                st.markdown("💬 Try asking a detailed research question.")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Please ask a more detailed query."
                })

            # ======================
            # RESEARCH FLOW
            # ======================
            else:
                st.markdown("🔍 Searching...")
                search_data = run_search(user_input)

                st.markdown("📄 Reading sources...")
                reader_text, urls = run_reader(search_data)

                # SHOW SOURCES
                with st.expander("🔗 Sources"):
                    for u in urls:
                        st.write(u)

                # WRITE REPORT
                st.markdown("✍️ Generating report...")
                report = writer_chain.invoke({
                    "topic": user_input,
                    "research_data": reader_text[:2000]
                })

                st.markdown("## 📘 Report")
                st.markdown(report)

                # ANALYTICS
                st.markdown("## 📊 Insights")

                keywords = get_keywords(reader_text)
                plot_chart(keywords)

                # CONTEXT
                st.markdown("## 📰 Context")
                st.write(reader_text[:400])

                # REVIEW
                review = critic_chain.invoke({
                    "report": report
                })

                st.markdown("## 📊 Evaluation")
                st.markdown(review)

                st.download_button("⬇ Download Report", report)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": report
                })

    except Exception:
        st.error("❌ Error occurred")
        st.code(traceback.format_exc())