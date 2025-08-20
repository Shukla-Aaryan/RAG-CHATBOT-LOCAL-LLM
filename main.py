import os

# Disable TensorFlow to avoid unwanted imports by HuggingFace
os.environ["TRANSFORMERS_NO_TF"] = "1"

# ADD THESE LINES RIGHT HERE ▼ (Before any other imports)
import asyncio
import nest_asyncio
nest_asyncio.apply()  # Must be called before Streamlit starts
# ▲ Put this right after os.environ but before other imports

from chatbot.loader import load_documents
from chatbot.splitter import split_document
from chatbot.embedder import create_embedding
from chatbot.retriver import load_retriever
from chatbot.llm_chain import build_llm_chain
from chatbot.interface import chat_interface


def initialize_retriever(persist_dir="embeddings"):
    """
    Load or create a retriever depending on whether embeddings already exist.
    """
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print("[main] Loading existing retriever from disk...")
        retriever = load_retriever(persist_directory=persist_dir)
    else:
        print("[main] No existing embeddings found. Embeddings will be created at runtime.")
        retriever = None  # Will be created later by interface
    return retriever


def main():
    persist_dir = "embeddings"
    retriever = initialize_retriever(persist_dir)
    qa_chain = build_llm_chain(retriever) if retriever else None

    # Launch the Streamlit interface
    chat_interface(
        load_pdf=load_documents,
        split_text=split_document,
        create_vectorstore=create_embedding,
        get_qa_chain=(lambda r: build_llm_chain(r)) if not qa_chain else (lambda _: qa_chain)
    )


if __name__ == "__main__":
    main()
