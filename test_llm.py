from llm_chain import build_llm_chain
from retriever import load_retriever

def test_llm():
    # Test raw LLM
    from langchain_community.llms import LlamaCpp
    llm = LlamaCpp(
        model_path=r"C:\Users\aarya\OneDrive\Documents\ML Projects\RAG CHATBOT LLM\RAG-CHATBOT-LOCAL-LLM\models\llama-2-7b-chat.Q4_K_M.gguf",  # Raw string (note 'r')
        n_ctx=2048,
        n_gpu_layers=40,
        verbose=False  # Disable debug logs
    )
    print("LLM Response:", llm("What is 2+2?"))

def test_rag():
    # Test full RAG pipeline
    retriever = load_retriever()
    qa_chain = build_llm_chain(retriever)
    result = qa_chain("What is the main topic of the document?")
    print("RAG Answer:", result["result"])
    print("Sources:", [doc.metadata["source"] for doc in result["source_documents"]])

if __name__ == "__main__":
    test_llm()  # Test 1: Raw LLM
    test_rag()  # Test 2: RAG pipeline