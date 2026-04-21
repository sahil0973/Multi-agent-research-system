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
# SESSION
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
        # Save user msg
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):

            # Decide complexity
            is_complex = len(user_input.split()) > 5

            # ======================
            # SIMPLE (CHATGPT STYLE)
            # ======================
            if not is_complex:

                response = writer_chain.invoke({
                    "topic": user_input,
                    "research_data": "Explain simply with examples"
                })

                st.markdown(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

            # ======================
            # FULL RESEARCH MODE
            # ======================
            else:

                st.markdown("🔍 Searching...")
                search_data = run_search(user_input)

                st.markdown("📄 Reading...")
                reader_text, urls = run_reader(search_data)

                with st.expander("🔗 Sources"):
                    for u in urls:
                        st.write(u)

                st.markdown("✍️ Writing report...")
                report = writer_chain.invoke({
                    "topic": user_input,
                    "research_data": reader_text[:2000]
                })

                st.markdown("## 📘 Report")
                st.markdown(report)

                # 📊 Analytics
                st.markdown("## 📊 Insights")
                keywords = get_keywords(reader_text)
                plot_chart(keywords)

                # 📰 Context
                st.markdown("## 📰 Context")
                st.write(reader_text[:400])

                # 🧠 Review
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