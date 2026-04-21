import streamlit as st
import traceback

from agents import (
    run_search, run_reader,
    writer_chain, critic_chain,
    run_images, run_videos,
    detect_field
)

st.set_page_config(page_title="ResearchMind Pro", page_icon="🔬", layout="wide")

st.title("🔬 ResearchMind Pro")
st.caption("AI Research Engine with visuals, sources & insights")

def safe(text, n=2000):
    return text[:n] if text else ""

topic = st.text_input("Ask anything...")

if st.button("🚀 Run") and topic:

    try:
        # FIELD
        st.info("🧠 Detecting field...")
        field = detect_field(topic)
        st.success(f"📌 Field: {field}")

        # SEARCH
        st.info("🔍 Searching...")
        search_data = run_search(topic)

        # IMAGES
        images = run_images(topic)

        # VIDEOS
        video = run_videos(topic)

        # SCRAPE
        st.info("📄 Extracting...")
        reader_text, urls = run_reader(search_data)

        # UI PANEL
        col1, col2 = st.columns([2,1])

        with col1:
            if images:
                st.image(images[0], use_container_width=True)

        with col2:
            st.markdown("### 🎥 Video")
            st.markdown(f"[Watch]({video})")

        # SOURCES
        with st.expander("🔗 Sources"):
            for u in urls:
                st.write(u)

        # REPORT
        st.info("✍️ Writing...")

        report = writer_chain.invoke({
            "topic": topic,
            "research_data": safe(reader_text),
            "field": field
        })

        st.subheader("📘 Report")
        st.write(report)

        st.download_button("⬇ Download", report)

        # REVIEW
        st.info("🧠 Reviewing...")

        review = critic_chain.invoke({
            "report": safe(report)
        })

        st.subheader("📊 Review")
        st.write(review)

        st.success("✅ Done")

    except Exception:
        st.error("❌ Error")
        st.code(traceback.format_exc())