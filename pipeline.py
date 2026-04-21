import time
from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain


def safe_invoke(agent, payload, retries=3):
    for i in range(retries):
        try:
            return agent.invoke(payload)
        except Exception as e:
            print(f"Retry {i+1} due to error: {e}")
            time.sleep(2 ** i)
    raise Exception("Max retries exceeded")


def extract_links(text):
    lines = text.split("\n")
    links = [l for l in lines if "http" in l]
    return list(set(links))[:3]   # limit to top 3


def run_research_pipeline(topic: str) -> dict:

    state = {}

    print("\n" + "="*50)
    print("STEP 1 - Search Agent")
    print("="*50)

    search_agent = build_search_agent()

    search_result = safe_invoke(search_agent, {
        "messages": [("user", f"Find 5 reliable sources about: {topic}")]
    })

    search_text = search_result['messages'][-1].content
    state["search_results"] = search_text

    print("\nSearch Results:\n", search_text)

    # ✅ Extract clean links
    links = extract_links(search_text)

    if not links:
        raise Exception("No valid links found. Improve search prompt.")

    print("\nTop Links:", links)

    print("\n" + "="*50)
    print("STEP 2 - Reader Agent")
    print("="*50)

    reader_agent = build_reader_agent()
    research_data = []

    for link in links:
        print(f"\nScraping: {link}")

        result = safe_invoke(reader_agent, {
            "messages": [("user", f"Extract key insights from: {link}")]
        })

        content = result['messages'][-1].content
        research_data.append(content)

        time.sleep(1)  # rate control

    combined_research = "\n\n".join(research_data)
    state["scraped_content"] = combined_research

    print("\nScraped Content:\n", combined_research[:1000])

    print("\n" + "="*50)
    print("STEP 3 - Writer Agent")
    print("="*50)

    report = writer_chain.invoke({
        "topic": topic,
        "research": combined_research[:4000]   # limit tokens
    })

    state["report"] = report

    print("\nFinal Report:\n", report)

    print("\n" + "="*50)
    print("STEP 4 - Critic Agent")
    print("="*50)

    feedback = critic_chain.invoke({
        "report": report
    })

    state["feedback"] = feedback

    print("\nCritic Feedback:\n", feedback)

    return state


if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    run_research_pipeline(topic)
