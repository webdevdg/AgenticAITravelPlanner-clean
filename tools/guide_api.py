# Reloading the vector index and Creating a RAG retrieval tool

import os
from dotenv import load_dotenv
# load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=env_path)

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from typing import List

# # Reloading the vector store - This makes our index reusable across sessions or scripts.
# _store = FAISS.load_local("data/guide_index", OpenAIEmbeddings())

_store = FAISS.load_local("data/guide_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# Retriever Tool
def retrieve_tips(query: str, k: int = 5) -> List[str]:
    """
    Search the LOCAL travel guides (not the web). Return short passages with insider tips.
    Always pass the city name if the user asks about a specific city (e.g., Paris, Kyoto).
    Use this for hidden gems, dining, neighborhoods, and day-by-day planning.
    """

    # similarity_search takes a query string, Converts it into a vector using the embedding model
    # Searches the FAISS index for the most similar stored document vectors.
    # Returns the top-k matching documents. ( k is the number of most relevant documents to return.)

    docs = _store.similarity_search(query, k=k)  # FAISS index searches through the stored vector embeddings and returns the top k most similar documents (or chunk)
    print(f"[RAG] retrieve_tips: {query} (k={k}) -> {len(docs)} hits")
    return [d.page_content for d in docs]


