# USING BUILTIN CREATE_REACT_AGENT INSTEAD OF MANUALLY HANDLING THE REACT AGENT ARCHITECTURE

import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from tools.flight_api import search_flights
from tools.hotel_api import search_hotels
from tools.guide_api import retrieve_tips
load_dotenv()

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_PROJECT"] = "AgenticTravelPlanner"

class State(TypedDict):
    messages: Annotated[list[dict], add_messages]

llm = ChatOpenAI(model="gpt-3.5-turbo")

# tools=[search_flights, search_hotels]
tools=[search_flights, search_hotels, retrieve_tips] #Integrating RAG Tool

agent = create_react_agent(llm, tools)

# Build the graph
builder = StateGraph(State)
builder.add_node("react_agent", agent)
builder.add_edge(START,     "react_agent")
builder.add_edge("react_agent", END)
graph = builder.compile()

view_graph = graph

if __name__ == "__main__":
    init_state = {
        "messages": [
            # {"role": "user", "content": "Plan a 3-day trip to Mumbai."}
            # {"role": "user", "content": "Find me flights from New Delhi to Mumbai on 2025-08-10"}
            # {"role": "user", "content": "Find me hotels in New York City for checkin on 2025-08-10 and checkout on 2025-08-12"}
            # {"role": "user", "content": "Find me flights from New Delhi to Mumbai on 2025-08-10 and plan a 3 day trip in mumbai"}
            {"role": "user", "content": "What are some hidden gems in Paris?"}
            # {"role": "user", "content": "What is the capital of India?"}

        ]
    }

    result = graph.invoke(init_state)
    llm_msg = result["messages"][-1].content

    print(llm_msg)
