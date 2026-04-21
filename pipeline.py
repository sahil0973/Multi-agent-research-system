import time
from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain


def safe_invoke(agent, payload):
    for i in range(3):
        try:
            return agent.invoke(payload)
        except Exception:
            time.sleep(2 ** i)
    raise Exception("Failed after retries")


def extract_links(text):
    return list(set([l for l in text.split() if "http" in l]))[:3]


def run_research_pipeline(topic: str):

    state = {}

    # STEP 1: SEARCH
    search_agent = build_search_agent()

    sr = safe_invoke(search_agent, {
        "messages": [("user", f"Find sources about {topic}")]
    })

    search_text = sr["messages"][-1].content
    state["search_results"] = search_text

    links = extract_links(search_text)

    # STEP 2: READ
    reader_agent = build_reader_agent()
    data = []

    for link in links:
        rr = safe_invoke(reader_agent, {
            "messages": [("user", f"Extract from {link}")]
        })
        data.append(rr["messages"][-1].content)
        time.sleep(1)

    combined = "\n\n".join(data)
    state["scraped_content"] = combined

    # STEP 3: WRITE
    report = writer_chain.invoke({
        "topic": topic,
        "research": combined[:3000]
    })

    state["report"] = report

    # STEP 4: CRITIC
    feedback = critic_chain.invoke({
        "report": report
    })

    state["feedback"] = feedback

    return state
