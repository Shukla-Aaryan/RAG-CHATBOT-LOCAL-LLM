#from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_community.vectorstores import Chroma
import os
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def create_embedding(documents: list[Document], persist_directory: str = "embeddings") -> Chroma:
    """
    Create embeddings for documents and persist to ChromaDB.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    #vectordb.persist()
    print(f"[embedder] embeddings stored at '{persist_directory}'")
    return vectordb.as_retriever(search_kwargs={"k": 4})