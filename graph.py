#  MANUALLY HANDLING THE REACT AGENT ARCHITECTURE

import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from tools.flight_api import search_flights
from tools.hotel_api import search_hotels
load_dotenv()

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_PROJECT"] = "AgenticTravelPlanner"

class State(TypedDict):
    messages: Annotated[list[dict], add_messages]

llm = ChatOpenAI(model="gpt-3.5-turbo")

tools=[search_flights, search_hotels]
llm_with_tool=llm.bind_tools([search_flights, search_hotels])


def chatbot(state: State) -> State:
    """
    Receives state["messages"]: a list of {role,content} dicts.
    Returns {"messages": [<the LLM's response>]}.
    """

    chat_result = llm_with_tool.invoke(state["messages"])
    return {"messages": [chat_result]}

# 5) Build the graph
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_node("tools",ToolNode(tools))
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot",tools_condition)
# builder.add_edge("tools",END)
builder.add_edge("tools","chatbot")
graph = builder.compile()
view_graph = graph

if __name__ == "__main__":
    init_state = {
        "messages": [
            # {"role": "user", "content": "Plan a 3-day trip to Mumbai."}
            # {"role": "user", "content": "Find me flights from New Delhi to Mumbai on 2025-08-10"}
            # {"role": "user", "content": "Find me hotels in New York City for checkin on 2025-08-10 and checkout on 2025-08-12"}
            # {"role": "user", "content": "Find me flights from New Delhi to Mumbai on 2025-08-10 and plan a 3 day trip in mumbai"}
            # {"role": "user", "content": "Find me flights from New Delhi to Mumbai on 2025-08-10 and find hotels in mumbai for checkin on 2025-08-10 and checkout on 2025-08-12 plan a 3-day trip in Mumbai like what are the places i can visit in those 2 days"}
            {"role": "user", "content": "What are some hidden gems in Paris?"}
        ]
    }

    result = graph.invoke(init_state)
    llm_msg = result["messages"][-1].content

    print(llm_msg)
