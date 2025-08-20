from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
from tempfile import NamedTemporaryFile

def load_documents(uploaded_file) -> list[Document]:
    """
    Load a single uploaded PDF or text file.
    """
    documents = []

    # Save the uploaded file to a temporary file
    with NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Choose the loader based on file type
    if tmp_path.endswith(".pdf"):
        loader = PyPDFLoader(tmp_path)
    elif tmp_path.endswith(".txt"):
        loader = TextLoader(tmp_path)
    else:
        print(f"[loader] Unsupported file type: {tmp_path}")
        return []

    docs = loader.load()
    documents.extend(docs)

    print(f"[loader] loaded {len(documents)} document chunks")
    return documents
