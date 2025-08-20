from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def split_document(documents: list[Document], chunk_size=500, chunk_overlap=50) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[splitter] split into {len(chunks)} chunks")
    return chunks