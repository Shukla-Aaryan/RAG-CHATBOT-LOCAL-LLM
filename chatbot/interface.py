import streamlit as st

def chat_interface(load_pdf, split_text, create_vectorstore, get_qa_chain):
    st.set_page_config(page_title="RAG CHATBOT", layout="wide")
    st.title("📄 AI Chatbot — Ask Questions About Your PDF")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    # Step 1: Ask for PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None and st.session_state.qa_chain is None:
        with st.spinner("Processing PDF..."):
            raw_text = load_pdf(uploaded_file)
            chunks = split_text(raw_text)
            vectordb = create_vectorstore(chunks)
            st.session_state.qa_chain = get_qa_chain(vectordb)
            st.success("✅ PDF processed successfully! You can now ask questions.")

    # Step 2: Show question input only if QA chain is ready
    if st.session_state.qa_chain:
        user_input = st.text_input("Ask a question about the PDF:")

        if user_input:
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain.invoke({"question": user_input},
                config={"max_execution_time": 30}) #THIS WAS ALSO A CORRECTION HELPS IN PROVIDING A EXECUTION TIME
                answer = result["result"]
                source_docs = result.get("source_documents", [])

                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Bot", answer))

            # Show chat history
            for sender, message in st.session_state.chat_history:
                st.markdown(f"**{sender}:** {message}")

            # Optional: Show source documents
            if source_docs:
                with st.expander("Sources"):
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**Document {i+1}:**")
                        st.markdown(doc.page_content)
    else:
        st.info("📄 Please upload a PDF to start chatting.")

