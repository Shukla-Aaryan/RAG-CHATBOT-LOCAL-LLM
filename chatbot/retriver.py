#from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def load_retriever(persist_directory: str = "embeddings"):
    """
    Load ChromaDB from disk and return a retriever object for querying.

    Args:
        persist_directory (str): Directory where the Chroma vector store is saved.

    Returns:
        Retriever: A retriever instance for similarity search.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # ✅

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    print(f"[RETRIEVER] Loaded retriever with top-k=4 from '{persist_directory}'")
    return retriever
