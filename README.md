# Agentic Travel Planner

A conversational assistant that uses Agentic AI to plan personalized multi‑day trips.  
Leverages LangGraph, ReAct agents, RAG over local travel guides, and OpenAI’s GPT models to search flights & hotels, retrieve insider tips, build itineraries, and validate against budget & schedule.

---

## 🛠️ Features

- **Natural‑Language Chat**  
  Ask for a trip (“Plan a 3‑day Paris getaway”) and get an itinerary.

- **ReAct‑Style Search Agents**  
  Calls stubbed (or real) flight & hotel APIs in a think→act→observe loop.

- **RAG over Local Guides**  
  Index your own Markdown travel tips (hidden gems, dining, sights) via FAISS + OpenAI embeddings.

- **Memory of Preferences**  
  Remembers your favorite airlines, hotel class, budget limits for follow‑up queries.

- **Supervisor Checks**  
  Validates final plan against budget and timing constraints.

- **Streaming Responses**  
  Itinerary unfolds token‑by‑token in real time.

- **Human‑in‑the‑Loop Feedback**  
  Swap a museum day for a cooking class, and the planner updates on the fly.

- **Extensible Architecture**  
  Add new tools, RAG nodes, multi‑agent supervisors, PDF export, calendar sync, price alerts, and more.

---

## 📦 Tech Stack

- **LangGraph** (StateGraph / MessageGraph) for orchestration  
- **langgraph‑prebuilt** for ReAct `AgentNode`, `ToolNode`, etc.  
- **OpenAI GPT‑3.5 / GPT‑4** via `langchain-openai`  
- **FAISS** + OpenAI Embeddings for RAG  
- **Python 3.10+**, `python-dotenv` for config  
- (Optional) **ReportLab** for PDF export  
- (Optional) **Docker** + **MCP** for deployment

---

## 🚀 Quick Start

1. **Clone & enter project**
   ```bash
   git clone git@github.com:<your-username>/agentic-travel-planner.git
   cd agentic-travel-planner

