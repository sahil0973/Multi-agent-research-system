import streamlit as st
import traceback
import time

from agents import (
    run_search, run_reader,
    writer_chain, critic_chain,
    run_images, run_videos,
    detect_field, chat_chain
)

st.set_page_config(page_title="ResearchMind AI", page_icon="🧠", layout="wide")

# ======================
# ADVANCED CSS
# ======================

st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.chat-card {
    background: #1c1f26;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
}
.suggestion-btn button {
    background: #1f2937;
    border-radius: 20px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ======================
# HEADER
# ======================

st.markdown("# 🧠 ResearchMind AI")
st.caption("Think • Explore • Understand")

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
# SUGGESTIONS
# ======================

st.markdown("### 💡 Try asking:")

suggestions = [
    "Explain quantum computing simply",
    "Future of AI in 2030",
    "How fusion energy works",
]

cols = st.columns(len(suggestions))
for i, s in enumerate(suggestions):
    with cols[i]:
        if st.button(s):
            st.session_state["quick"] = s

# ======================
# INPUT (CHAT + VOICE)
# ======================

user_input = st.chat_input("Ask anything...")

if "quick" in st.session_state:
    user_input = st.session_state.pop("quick")

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
        time.sleep(0.003)
    return output

# ======================
# MAIN LOGIC
# ======================

if user_input:

    try:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        is_research = len(user_input.split()) > 5

        with st.chat_message("assistant"):

            # ======================
            # CHAT MODE
            # ======================
            if not is_research:

                history = "\n".join(
                    [f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]]
                )

                response = chat_chain.invoke({
                    "history": history,
                    "input": user_input
                })

                final = stream_text(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final
                })

                # FOLLOW-UP SUGGESTIONS
                st.markdown("### 🔁 Follow-up")
                followups = [
                    "Explain in simple terms",
                    "Give real-world examples",
                    "Summarize briefly"
                ]

                cols = st.columns(3)
                for i, f in enumerate(followups):
                    with cols[i]:
                        if st.button(f, key=f):
                            st.session_state["quick"] = f

            # ======================
            # RESEARCH MODE
            # ======================
            else:

                progress = st.progress(0)

                field = detect_field(user_input)
                progress.progress(10)

                st.markdown(f"### 🏷️ Field: `{field}`")

                search_data = run_search(user_input)
                progress.progress(30)

                images = run_images(user_input)
                video = run_videos(user_input)

                reader_text, urls = run_reader(search_data)
                progress.progress(60)

                # SMART CARD UI
                st.markdown("## 🧠 Insights Panel")

                col1, col2 = st.columns([2,1])

                with col1:
                    if images:
                        st.image(images[0], use_container_width=True)

                with col2:
                    st.markdown("### 🎥 Learn")
                    st.markdown(f"[▶ Watch Video]({video})")

                # SOURCES
                with st.expander("🔗 Sources"):
                    for u in urls:
                        st.write(u)

                report = writer_chain.invoke({
                    "topic": user_input,
                    "research_data": safe(reader_text),
                    "field": field
                })

                progress.progress(85)

                st.markdown("## 📘 Report")
                st.markdown(report)

                st.download_button("⬇ Download", report)

                review = critic_chain.invoke({
                    "report": safe(report)
                })

                progress.progress(100)

                st.markdown("## 📊 Evaluation")
                st.markdown(review)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": report
                })

    except Exception:
        st.error("❌ Error")
        st.code(traceback.format_exc())