# TRYING TO USE "AgentNode" FOR HANDLING REACT AGENT BUT NOT WORKING AT THE MOMENT
import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.agent import AgentNode
from tools.flight_api import search_flights
from tools.hotel_api import search_hotels

load_dotenv()

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_PROJECT"] = "AgenticTravelPlanner"

class State(TypedDict):
    messages: Annotated[list[dict], add_messages]

llm = ChatOpenAI(model="gpt-3.5-turbo")

# turn your flight_api & hotel_api calls into ToolNodes:
flight_tool = ToolNode(
  name="search_flights",
  func=search_flights,
  description="Search flights given origin, destination, dates"
)
hotel_tool = ToolNode(
  name="search_hotels",
  func=search_hotels,
  description="Search hotels given city, check-in, check-out"
)

agent = AgentNode(
  agent_type="react",
  llm_model="gpt-3.5-turbo",
  tools=[flight_tool, hotel_tool],
  # Optional: a custom prompt template or stop condition
)


builder = StateGraph(State)
# replace your old nodes & edges with:
builder.add_node("react_agent", agent)
builder.add_edge(START,     "react_agent")
builder.add_edge("react_agent", END)
graph = builder.compile()


if __name__ == "__main__":
    init_state = {
        "messages": [
            # {"role": "user", "content": "Plan a 3-day trip to Mumbai."}
            # {"role": "user", "content": "Find me flights from New Delhi to Mumbai on 2025-08-10"}
            # {"role": "user", "content": "Find me hotels in New York City for checkin on 2025-08-10 and checkout on 2025-08-12"}
            # {"role": "user", "content": "Find me flights from New Delhi to Mumbai on 2025-08-10 and plan a 3 day trip in mumbai"}
            {"role": "user", "content": "Find me flights from New Delhi to Mumbai on 2025-08-10 and find hotels in mumbai for checkin on 2025-08-10 and checkout on 2025-08-12 plan a 3-day trip in Mumbai like what are the places i can visit in those 2 days"}

        ]
    }

    result = graph.invoke(init_state)
    llm_msg = result["messages"][-1].content

    print(llm_msg)
