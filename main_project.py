import os
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Define the directory where vector stores were saved
vector_store_path = os.path.join(os.getcwd(), "vector_stores")

# Paths to the saved FAISS vector stores
summary_store_path = os.path.join(vector_store_path, "summary_store")
detailed_store_path = os.path.join(vector_store_path, "detailed_store")

# Initialize OpenAI embeddings (ensure your API key is set)
embeddings = OpenAIEmbeddings()

# Load the stored vector databases
summary_store = FAISS.load_local(summary_store_path, embeddings, allow_dangerous_deserialization=True)
detailed_store = FAISS.load_local(detailed_store_path, embeddings, allow_dangerous_deserialization=True)

print("Vector stores loaded successfully!")