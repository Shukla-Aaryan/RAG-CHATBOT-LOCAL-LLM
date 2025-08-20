from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from langchain_core.runnables import RunnableConfig  # For config support
def build_llm_chain(retriever):
    """
    Builds a RAG chain with a local Llama 2/Mistral model.
    Args:
        retriever: ChromaDB retriever from retriever.py
    Returns:
        RetrievalQA chain ready for queries
    """
    # System prompt with strict instructions
    prompt_template = """[INST] <<SYS>>
You are an AI assistant for question-answering. Follow these rules:
1. Use ONLY the provided context.
2. Be concise and factual.
3. If unsure, say "I don't know".
<</SYS>>

Context: {context}
Question: {question} [/INST]
Answer:"""

    # Local LLM setup (adjust based on your model)
    llm = LlamaCpp(
        model_path="models/llama-2-7b-chat.Q4_K_M.gguf",  # Update to your model file
        temperature=0.7,      # Controls creativity (0=strict, 1=creative)
        max_tokens=512,       # Max response length
        n_ctx=2048,           # Context window size
        n_gpu_layers=0,      # Enable GPU acceleration (set to 0 for CPU-only)
        verbose=False         # Disable debug logs
    )

    prompt = PromptTemplate.from_template(prompt_template)
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",   # Simple concatenation of chunks
        retriever=retriever,  # Your ChromaDB retriever
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,  # Enable source citations
        input_key="question"  # 👈 this is the key fix
    )


'''
# TEMPORARY TEST CODE (run only once)
if __name__ == "__main__":
    from retriever import load_retriever
    
    print("\nTesting LLM without RAG...")
    llm = LlamaCpp(
        model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
        n_ctx=2048,
        n_gpu_layers=40,
        verbose=False
    )
    print(llm("Tell me a one-sentence joke about AI."))

    print("\nTesting RAG pipeline...")
    retriever = load_retriever()
    qa_chain = build_llm_chain(retriever)
    print(qa_chain("What is the document about?"))


'''