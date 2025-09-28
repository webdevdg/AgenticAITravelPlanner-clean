# USING BUILTIN CREATE_REACT_AGENT INSTEAD OF MANUALLY HANDLING THE REACT AGENT ARCHITECTURE

import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
# from langgraph.checkpoint.memory import MemorySaver
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
    for k, v in s["preferences"].items():
        store.put(s["thread_id"], k, v)
    return s

# builder.add_node("load_prefs", lambda s: {"preferences": {k: store.get(s["thread_id"], k) for k in store.list_keys(s["thread_id"])}})
builder.add_node("load_prefs", load_prefs_node)
builder.add_node("react_agent", agent)
builder.add_node("save_prefs", save_prefs_node)

builder.add_edge(START,        "load_prefs")
builder.add_edge("load_prefs", "react_agent")
builder.add_edge("react_agent","save_prefs")
builder.add_edge("save_prefs", END)

graph = builder.compile()
# graph = builder.compile(checkpointer=memory_cp)

view_graph = graph


# if _name_ == "_main_":
#     init_state = {
#         "messages": [
#             # {"role": "user", "content": "Plan a 3-day trip to Mumbai."}
#             # {"role": "user", "content": "Find me flights from New Delhi to Mumbai on 2025-08-10"}
#             # {"role": "user", "content": "Find me hotels in New York City for checkin on 2025-08-10 and checkout on 2025-08-12"}
#             # {"role": "user", "content": "Find me flights from New Delhi to Mumbai on 2025-08-10 and plan a 3 day trip in mumbai"}
#             {"role": "user", "content": "What are some hidden gems in Paris?"}
#             # {"role": "user", "content": "What is the capital of India?"}
#
#         ]
#     }
#
#     result = graph.invoke(init_state)
#
#     llm_msg = result["messages"][-1].content
#
#     print(llm_msg)

if __name__ == "__main__":
    thread_id = "user123"   # to identify this session

    # ── STEP 1: Set a preference ───────────────────────────────────────
    turn1 = graph.invoke(
        {
            "messages": [{"role":"user", "content":"I prefer 4-star hotels and a $2000 budget."}],
            "thread_id": thread_id
        },
        config={"configurable": {"thread_id": thread_id}}
    )
    print("Agent:", turn1["messages"][-1].content)

    # ── STEP 2: Ask something that should use that preference ─────────
    turn2 = graph.invoke(
                {
                "messages": [
                    {"role": "user", "content": "Find me hotels in NYC for checkin on 2025-10-10 and checkout on 2025-10-12"}
                ],
                "thread_id": thread_id
        },
        config={"configurable": {"thread_id": thread_id}}
    )

    print("Agent:", turn2["messages"][-1].content)
