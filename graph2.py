# USING BUILTIN CREATE_REACT_AGENT INSTEAD OF MANUALLY HANDLING THE REACT AGENT ARCHITECTURE

import os
import re
import json
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
# from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler  # minimal token printer for streaming
from store.redis_store import RedisStore
from tools.flight_api import search_flights
from tools.hotel_api import search_hotels
from tools.guide_api import retrieve_tips
load_dotenv()

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_PROJECT"] = "AgenticTravelPlanner"

class PrintTokens(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)

class State(TypedDict):
    messages: Annotated[list[dict], add_messages]
    preferences: dict   # to have a key value store
    thread_id: str      # to keep thread_id in state

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)
extract_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=False)

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

    # Returning only prefs so the graph doesn't re-emit the last assistant message
    return {"preferences": s.get("preferences", {})}


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


def human_review_node(s):
    """Minimal HITL gate:
    - Return {"approved": True} to ship the answer
    - Or return {"approved": False, "messages":[HumanMessage(...)]} to revise
    """
    # Last assistant message
    last = s["messages"][-1] if s.get("messages") else None
    text = (
        getattr(last, "content", None)
        or (last.get("content", "") if isinstance(last, dict) else "")
        or ""
    )

    mode = os.getenv("HITL_MODE", "auto").lower()  # "auto" or "ask"
    if mode != "ask":
        return {"approved": True}

    print("\n\n--- REVIEW (HITL) ---\n")
    print(text)
    print("\nApprove to send? [y] yes / [e] edit & retry / anything else = reject")
    try:
        ans = input("> ").strip().lower()
    except EOFError:
        ans = "y"  # non-interactive fallback

    if ans == "y":
        return {"approved": True}

    if ans == "e":
        note = input("Reviewer feedback (will be added as a HumanMessage): ").strip()
        return {
            "approved": False,
            "messages": [HumanMessage(content=f"Reviewer feedback: {note}")]
        }

    # plain reject -> loop back to agent without extra guidance
    return {"approved": False}


# --- Structured data review gate ---

def structured_review_node(s):
    """
    If HITL_STRUCT=ask:
      - Extract compact JSON {hotels, flights, tips} from the last assistant draft.
      - Ask for approval on the DATA only.
    Else:
      - Auto-approve.
    Returns only small flags/optional feedback. Does NOT emit messages to avoid duplicates.
    """
    if os.getenv("HITL_STRUCT", "off").lower() != "ask":
        return {"approved_struct": True}

    msgs = s.get("messages", [])
    last = msgs[-1] if msgs else None
    draft = (
        getattr(last, "content", None)
        or (last.get("content", "") if isinstance(last, dict) else "")
        or ""
    )

    # Use your llm to extract compact JSON; you can swap to a cheaper model if you like.
    system = SystemMessage(content=(
        "Extract a compact JSON object from the draft with keys hotels, flights, tips.\n"
        "Schema:\n"
        "{\n"
        "  hotels: [{name, price?, currency?, address?, description?}],\n"
        "  flights: [{airline?, price?, currency?, departure?, arrival?, nonstop?}],\n"
        "  tips: [{title, why}]\n"
        "}\n"
        "Return ONLY valid JSON. No extra text."
    ))

    resp = extract_llm.invoke([system, HumanMessage(content=draft)])
    raw = (resp.content or "").strip()

    try:
        data = json.loads(raw)
    except Exception:
        print("\n--- STRUCTURED REVIEW (raw not valid JSON) ---\n")
        # print(raw)
        ok = input("Ship anyway? [y]/[n]: ").strip().lower() == "y"
        return {"approved_struct": ok}

    print("\n--- STRUCTURED REVIEW ---\n")
    print(json.dumps(data, indent=2))
    ans = input("Approve structured data? [y] / [e]dit & retry / [n]: ").strip().lower()

    if ans == "y":
            print("✓ Structured data approved.")
            return {"approved_struct": True}

    if ans == "e":
        note = input("Reviewer feedback on data: ").strip()
        # Feed reviewer guidance back into the conversation and loop to react_agent
        return {"approved_struct": False,
                "messages": [HumanMessage(content=f"Reviewer feedback on data: {note}")]}

    return {"approved_struct": False}



# --- nodes ---
builder.add_node("load_prefs",   load_prefs_node)
builder.add_node("parse_prefs",  parse_prefs_node)
builder.add_node("inject_prefs", inject_prefs_node)
builder.add_node("detect_intent", detect_intent_node)
builder.add_node("react_agent",  agent)
builder.add_node("structured_review", structured_review_node)
builder.add_node("save_prefs",   save_prefs_node)

# --- edges ---
builder.add_edge(START,          "load_prefs")
builder.add_edge("load_prefs",   "parse_prefs")    # to parse incoming prefs first
builder.add_edge("parse_prefs",  "inject_prefs")   # then inject them
builder.add_edge("inject_prefs", "detect_intent")

builder.add_conditional_edges(
    "detect_intent",
    lambda s: "react_agent" if s.get("wants_tool") else "parse_prefs",
    {"react_agent": "react_agent", "parse_prefs": "parse_prefs"},
)

# builder.add_edge("react_agent",  "save_prefs")

# to route through HITL
builder.add_edge("react_agent", "structured_review")  # Adding as a human structure review at the end of the flow

builder.add_conditional_edges(
    "structured_review",
    lambda s: "ok" if s.get("approved_struct") else "revise",
    {"ok": "save_prefs", "revise": "react_agent"},
)

builder.add_edge("save_prefs",   END)


graph = builder.compile()
# graph = builder.compile(checkpointer=memory_cp)

view_graph = graph


# Node-by-node updates including tool invocations, system injections, and intent decisions
def run_with_event_stream(payload, thread_id="user123"):
    print("\n\n===== STREAM (updates) =====")
    for update in graph.stream(
        payload,
        config={"configurable": {"thread_id": thread_id}, "callbacks": [PrintTokens()]},  # Adding callback for token stream (Live LLM tokens as they’re generated)
        stream_mode="updates",
    ):
        if not update:
            continue

        for node_name, payload in update.items():
            print(f"\n[stream] node={node_name}")

            # Some nodes yield None or non-dict payloads — guard it.
            if payload is None:
                print("  (no state changes)")
                continue

            # Normalize: updates can be a single dict-like, or a list of dict-likes
            items = payload if isinstance(payload, list) else [payload]

            printed_any = False
            for i, item in enumerate(items):
                # Try to read messages from dicts or objects
                messages = []
                if isinstance(item, dict):
                    messages = item.get("messages") or []
                else:
                    messages = getattr(item, "messages", None) or []

                if messages:
                    printed_any = True
                    last = messages[-1]
                    # Support LC message objects and dict-style messages
                    if hasattr(last, "content"):
                        content = last.content or ""
                        role = getattr(last, "type", "assistant")
                    elif isinstance(last, dict):
                        content = last.get("content", "") or ""
                        role = last.get("role", "assistant")
                    else:
                        content, role = "", "assistant"

                    preview = content[:300] + ("..." if len(content) > 300 else "")
                    print(f"  [{i}] {role}: {preview}")

                # Also surface small scalar fields if present (e.g., intent flags / prefs)
                if isinstance(item, dict):
                    for k in ("wants_tool", "preferences"):
                        if k in item:
                            printed_any = True
                            print(f"  {k}: {item[k]}")

            if not printed_any:
                print("  (no messages or tracked fields in this update)")



# if __name__ == "__main__":
#     thread_id = "user123"   # to identify this session

#     single = {
#         "messages": [{
#             "role": "user",
#             "content": (
#                 "My preferences: I prefer 4-star hotels and a $2000 total budget for the hotel stay. "
#                 "Please do three things:\n"
#                 "1) Find 4-star hotels in NYC (city code NYC) for check-in 2025-10-10 and check-out 2025-10-12, "
#                 "keeping the total under $2000.\n"
#                 "2) Find nonstop ECONOMY flights from New Delhi (DEL) to Mumbai (BOM) on 2025-10-25 under $150.\n"
#                 "3) Using the local guides, find 3 hidden gems in Paris and explain why each is special.\n"
#                 "Respond in three sections: Hotels, Flights, Hidden Gems."
#             )
#         }],
#         "thread_id": thread_id
#     }

#     out = graph.invoke(
#         single, 
#         config={
#             "configurable": {"thread_id": thread_id}},
#             callbacks=[PrintTokens()]
#     )
#     # print(out["messages"][-1].content)
#     print("\n\n--- FINAL MESSAGE ---\n", out["messages"][-1].content)



if __name__ == "__main__":
    thread_id = "user123"   # to identify this session
    prompt = {
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

    print("\n\n>>> EVENT STREAM (nodes/tools) <<<\n")
    run_with_event_stream(prompt, thread_id=thread_id)
