import json
from typing import TypedDict

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Mock search tool — stands in for a real SearxNG instance
# ---------------------------------------------------------------------------

@tool
def mock_searxng_search(query: str) -> str:
    """Returns hardcoded recent headlines based on keywords in the query."""
    q = query.lower()

    if any(k in q for k in ("crypto", "bitcoin", "ethereum", "web3", "defi")):
        return (
            "Bitcoin hits new all-time high amid regulatory ETF approvals. "
            "Ethereum surges 40% on institutional demand."
        )
    if any(k in q for k in ("ai", "artificial intelligence", "llm", "gpt", "openai", "model")):
        return (
            "OpenAI GPT-5 announced with 10x capability improvement. "
            "EU AI Act enforcement begins next quarter with heavy fines."
        )
    if any(k in q for k in ("elon", "spacex", "tesla", "musk", "space")):
        return (
            "SpaceX Starship achieves full orbital flight on third attempt. "
            "Tesla FSD v13 rolls out to all US vehicles this week."
        )
    if any(k in q for k in ("market", "stock", "fed", "rate", "interest", "nasdaq", "trading")):
        return (
            "Fed signals two rate cuts in 2025 as inflation cools. "
            "S&P 500 hits record high on back of strong tech earnings beat."
        )
    if any(k in q for k in ("privacy", "surveillance", "data", "meta", "google", "monopoly")):
        return (
            "Meta fined $1.3B for illegal EU data transfers. "
            "New facial recognition surveillance tech deployed across major US cities."
        )
    if any(k in q for k in ("climate", "environment", "nature", "carbon", "fossil")):
        return (
            "IPCC report warns of accelerating climate tipping points by 2035. "
            "Record wildfires across Southern Europe destroy 2M acres."
        )

    return f"Multiple breaking developments on '{query}' reported across major outlets today."


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    bot_id: str
    persona: str
    search_query: str
    topic: str
    search_results: str
    post_content: str


# ---------------------------------------------------------------------------
# Structured output schema — enforces the required JSON shape
# ---------------------------------------------------------------------------

class PostOutput(BaseModel):
    bot_id: str
    topic: str
    post_content: str


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def node_decide_search(state: GraphState, llm: ChatOpenAI) -> GraphState:
    """
    The LLM reads the bot's persona and picks something to post about today.
    Outputs a search query and a topic label.
    """
    response = llm.invoke(
        f"You are a social media bot with this persona:\n{state['persona']}\n\n"
        "Pick one topic you genuinely want to post about right now given who you are. "
        'Reply ONLY with valid JSON (no markdown): {"search_query": "...", "topic": "..."}'
    )

    try:
        raw = response.content.strip().strip("```json").strip("```").strip()
        data = json.loads(raw)
        state["search_query"] = data["search_query"]
        state["topic"] = data["topic"]
    except (json.JSONDecodeError, KeyError):
        # Fallback so the graph doesn't crash on a bad LLM response
        state["search_query"] = "latest news today"
        state["topic"] = "current events"

    return state


def node_web_search(state: GraphState) -> GraphState:
    """Runs the mock search and drops the results into state."""
    state["search_results"] = mock_searxng_search.invoke({"query": state["search_query"]})
    return state


def node_draft_post(state: GraphState, llm_structured) -> GraphState:
    """
    Combines persona + real-world headlines → one opinionated post.
    Uses structured output so the JSON shape is guaranteed.
    """
    prompt = (
        f"You are bot '{state['bot_id']}'. Your persona:\n{state['persona']}\n\n"
        f"Breaking news you just found:\n{state['search_results']}\n\n"
        f"Write a single highly opinionated post about: {state['topic']}\n"
        "Stay completely in character. Hard limit: 280 characters. "
        "Sound like a real person with strong opinions, not a press release."
    )
    result: PostOutput = llm_structured.invoke(prompt)
    state["post_content"] = result.post_content
    return state


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_content_engine(llm: ChatOpenAI):
    llm_structured = llm.with_structured_output(PostOutput)

    # Wrap nodes so they close over the LLM without needing partial application
    def _decide(state):
        return node_decide_search(state, llm)

    def _search(state):
        return node_web_search(state)

    def _draft(state):
        return node_draft_post(state, llm_structured)

    g = StateGraph(GraphState)
    g.add_node("decide_search", _decide)
    g.add_node("web_search", _search)
    g.add_node("draft_post", _draft)

    g.set_entry_point("decide_search")
    g.add_edge("decide_search", "web_search")
    g.add_edge("web_search", "draft_post")
    g.add_edge("draft_post", END)

    return g.compile()


def run_content_engine(bot_id: str, persona: str, llm: ChatOpenAI) -> dict:
    engine = build_content_engine(llm)

    final = engine.invoke(
        GraphState(
            bot_id=bot_id,
            persona=persona,
            search_query="",
            topic="",
            search_results="",
            post_content="",
        )
    )

    return {
        "bot_id": final["bot_id"],
        "topic": final["topic"],
        "post_content": final["post_content"],
    }
