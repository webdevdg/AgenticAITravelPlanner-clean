# using LangChain, OpenAI, and FAISS to build and persist a local vector store (a vector index)
# from a directory of documents (ie:data/guides)

import os
from dotenv import load_dotenv
# Load environment variables from the .env file (in project root)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

from langchain_core.documents import Document as LCDocument
from llama_index.core import SimpleDirectoryReader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# Load LlamaIndex docs
llama_docs = SimpleDirectoryReader("data/guides").load_data()

# Convert LlamaIndex docs to LangChain-compatible documents
docs = [LCDocument(page_content=doc.get_content(), metadata={}) for doc in llama_docs]

# Initialize OpenAI embedding model (uses OPENAI_API_KEY from env)
embedding_model = OpenAIEmbeddings()

# Create FAISS vector store from documents
store = FAISS.from_documents(docs, embedding_model)

# Saving the vector store locally so that we don't need to recompute embeddings everytime
store.save_local("data/guide_index")



