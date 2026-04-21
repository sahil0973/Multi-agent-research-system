import streamlit as st
import traceback
import time

from agents import (
    run_search,
    run_reader,
    writer_chain,
    critic_chain,
    run_images,
    run_videos,
    detect_field
)

# ======================
# CONFIG
# ======================

st.set_page_config(
    page_title="ResearchMind AI",
    page_icon="🧠",
    layout="wide"
)

# ======================
# STYLE
# ======================

st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.stChatMessage {
    border-radius: 12px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

# ======================
# HEADER
# ======================

st.title("🧠 ResearchMind AI")
st.caption("Explore knowledge with AI")

# ======================
# MEMORY
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

def safe(text, n=2000):
    return text[:n] if text else ""

def stream_text(text):
    placeholder = st.empty()
    output = ""
    for ch in text:
        output += ch
        placeholder.markdown(output)
        time.sleep(0.002)
    return output

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

            # Progress
            progress = st.progress(0)

            # -------- FIELD --------
            field = detect_field(user_input)
            progress.progress(10)

            st.markdown(f"### 🏷️ Field: `{field}`")

            # -------- SEARCH --------
            search_data = run_search(user_input)
            progress.progress(30)

            # -------- VISUALS --------
            images = run_images(user_input)
            video = run_videos(user_input)

            # -------- SCRAPE --------
            reader_text, urls = run_reader(search_data)
            progress.progress(60)

            # -------- INSIGHT PANEL --------
            st.markdown("## 🧠 Insights")

            col1, col2 = st.columns([2,1])

            with col1:
                if images:
                    st.image(images[0], use_container_width=True)

            with col2:
                st.markdown("### 🎥 Learn")
                st.markdown(f"[▶ Watch Video]({video})")

            # -------- SOURCES --------
            with st.expander("🔗 Sources"):
                for u in urls:
                    st.write(u)

            # -------- REPORT --------
            report = writer_chain.invoke({
                "topic": user_input,
                "research_data": safe(reader_text),
                "field": field
            })

            progress.progress(85)

            st.markdown("## 📘 Report")
            final_report = stream_text(report)

            # Download
            st.download_button("⬇ Download Report", report)

            # -------- REVIEW --------
            review = critic_chain.invoke({
                "report": safe(report)
            })

            progress.progress(100)

            st.markdown("## 📊 Evaluation")
            st.markdown(review)

        # Save assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": report
        })

    except Exception:
        st.error("❌ Something went wrong")
        st.code(traceback.format_exc())