# USING BUILTIN CREATE_REACT_AGENT INSTEAD OF MANUALLY HANDLING THE REACT AGENT ARCHITECTURE

import os
import re
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
# from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from store.redis_store import RedisStore
from tools.flight_api import search_flights
from tools.hotel_api import search_hotels
from tools.guide_api import retrieve_tips
load_dotenv()

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_PROJECT"] = "AgenticTravelPlanner"

class State(TypedDict):
    messages: Annotated[list[dict], add_messages]
    preferences: dict   # to have a key value store
    thread_id: str      # to keep thread_id in state

llm = ChatOpenAI(model="gpt-3.5-turbo")

# tools=[search_flights, search_hotels]
tools=[search_flights, search_hotels, retrieve_tips] #Integrating RAG Tool

agent = create_react_agent(llm, tools)

# memory_cp = MemorySaver(namespace="travel")    # session replay
# memory_cp = MemorySaver()
store = RedisStore.from_url(os.environ["REDIS_URL"], namespace="prefs")

# Build the graph
builder = StateGraph(State)

def load_prefs_node(state):
    thread_id = state["thread_id"]
    prefs = {k: store.get(thread_id, k) for k in store.list_keys(thread_id)}
    return {"preferences": prefs}

def save_prefs_node(s):
    print("[save_prefs] writing:", s.get("preferences"))
    for k, v in s["preferences"].items():
        store.put(s["thread_id"], k, v)
    return s
    

# Compile once at module import
STAR_RE = re.compile(r"\b(\d+)\s*-\s*star\b|\b(\d+)\s*star\b", re.IGNORECASE)

BUDGET_RE = re.compile(
    r"""
    (?:                 # Case A: 'budget ... <amount>'
        \bbudget\b
        [\s:=-]*        # optional separator
        \$?\s*([\d,]+)
        (?:\s*(?:usd|dollars))?
    )
    |
    (?:                 # Case B: '$ <amount>' (optionally 'budget')
        \$\s*([\d,]+)
        (?:\s*(?:usd|dollars|budget))?
    )
    |
    (?:                 # Case C: 'under/less than/up to <amount>'
        \b(?:under|less\s+than|up\s*to|upto)\b
        \s*\$?\s*([\d,]+)
        (?:\s*(?:usd|dollars))?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

def parse_prefs_node(s):
    """
    Extract simple prefs from the most recent Human message and merge into state['preferences'].
    Parses:
      - hotel_class: '4-star', '5 star'
      - budget: '$2000', 'budget 2000', 'under 1500'
    """
    msgs = s.get("messages", [])
    if not msgs:
        return {}

    # find most recent USER message (works for LC objects and dict-style)
    last_user_text = None
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            last_user_text = m.content
            break
        if isinstance(m, dict) and m.get("role") == "user":
            last_user_text = m.get("content", "")
            break

    if not last_user_text:
        return {}

    prefs = dict(s.get("preferences", {}))

    # hotel class
    m_star = STAR_RE.search(last_user_text)
    if m_star:
        num = m_star.group(1) or m_star.group(2)
        prefs["hotel_class"] = f"{num}-star"

    # budget (IMPORTANT: pass the text to the compiled regex)
    m_budget = BUDGET_RE.search(last_user_text)
    if m_budget:
        # first non-None capture group from the pattern
        val = next(g for g in m_budget.groups() if g)
        prefs["budget"] = val.replace(",", "")

    return {"preferences": prefs}


def inject_prefs_node(s):
    prefs = s.get("preferences") or {}
    if not prefs:
        return {}
    bits = []
    if "hotel_class" in prefs: bits.append(f"prefer {prefs['hotel_class']} hotels")
    if "budget" in prefs:      bits.append(f"budget ≤ ${prefs['budget']}")
    return {
        "messages": [
            SystemMessage(
                content=f"User preferences (persist across turns): {', '.join(bits)}. "
                        f"Always honor these unless user overrides."
            )
        ]
    }

ACTION_VERBS = ("find", "show", "search", "book", "recommend", "get")
HOTEL_TRIGGERS = ("hotel", "stay", "accommodation")
FLIGHT_TRIGGERS = ("flight", "fly", "depart", "arrive")


def detect_intent_node(s):
    # read the last *user* message
    msgs = s.get("messages", [])
    last_user_text = ""
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            last_user_text = m.content or ""
            break
        if isinstance(m, dict) and m.get("role") == "user":
            last_user_text = m.get("content", "") or ""
            break
    low = last_user_text.lower()

    mentions_tool_domain = any(k in low for k in HOTEL_TRIGGERS + FLIGHT_TRIGGERS) or "guide" in low
    has_action = any(v in low for v in ACTION_VERBS)
    wants_tool = has_action and mentions_tool_domain
    return {"wants_tool": wants_tool}



# --- nodes ---
builder.add_node("load_prefs",   load_prefs_node)
builder.add_node("parse_prefs",  parse_prefs_node)
builder.add_node("inject_prefs", inject_prefs_node)
builder.add_node("detect_intent", detect_intent_node)
builder.add_node("react_agent",  agent)
builder.add_node("save_prefs",   save_prefs_node)

# --- edges ---
builder.add_edge(START,          "load_prefs")
builder.add_edge("load_prefs",   "parse_prefs")      # parse incoming prefs first
builder.add_edge("parse_prefs",  "inject_prefs")     # then inject them
builder.add_edge("inject_prefs", "detect_intent")

builder.add_conditional_edges(
    "detect_intent",
    lambda s: "react_agent" if s.get("wants_tool") else "parse_prefs",
    {"react_agent": "react_agent", "parse_prefs": "parse_prefs"},
)

builder.add_edge("react_agent",  "save_prefs")
builder.add_edge("save_prefs",   END)


graph = builder.compile()
# graph = builder.compile(checkpointer=memory_cp)

view_graph = graph


if __name__ == "__main__":
    thread_id = "user123"   # to identify this session

    single = {
        "messages": [{
            "role": "user",
            "content": (
                "My preferences: I prefer 4-star hotels and a $2000 total budget for the hotel stay. "
                "Please do three things:\n"
                "1) Find 4-star hotels in NYC (city code NYC) for check-in 2025-10-10 and check-out 2025-10-12, "
                "keeping the total under $2000.\n"
                "2) Find nonstop ECONOMY flights from New Delhi (DEL) to Mumbai (BOM) on 2025-10-25 under $150.\n"
                "3) Using the local guides, find 3 hidden gems in Paris and explain why each is special.\n"
                "Respond in three sections: Hotels, Flights, Hidden Gems."
            )
        }],
        "thread_id": thread_id
    }

    out = graph.invoke(single, config={"configurable": {"thread_id": thread_id}})
    print(out["messages"][-1].content)
