import streamlit as st
from preprocessors import preprocess
from io import StringIO
import docx
import PyPDF2

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub  # Placeholder, can swap with a local LLM

st.title("HB-HR RAG System")

tab1, tab2, tab3 = st.tabs(["📁 Upload", "🧹 Preprocess", "💬 Ask Questions"])

with tab1:
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])
    raw_text = ""

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                raw_text += page.extract_text()
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                raw_text += para.text + "\n"
        elif uploaded_file.type == "text/plain":
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            raw_text += stringio.read()

        st.session_state.raw_text = raw_text
        st.success("✅ File uploaded and text extracted!")

with tab2:
    if "raw_text" in st.session_state:
        docs = preprocess(st.session_state.raw_text)
        st.session_state.docs = docs
        st.success(f"✅ Document chunked into {len(docs)} parts.")
        st.write("📄 Example:", docs[0].page_content)
    else:
        st.info("Please upload a document first in the Upload tab.")

with tab3:
    if "docs" in st.session_state:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(st.session_state.docs, embeddings)

        # Save vectorstore
        vectorstore.save_local("vector_db")

        retriever = vectorstore.as_retriever()

        # Option 1: Use HuggingFaceHub if you have access token
        # llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.5, "max_length": 100})

        # Option 2: Use a dummy response for demo (you can replace with local LLM)
        from langchain.llms.base import LLM
        from typing import Optional, List

        class SimpleLLM(LLM):
            def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                return "This is a demo answer. Replace this with a real model for better responses."
            @property
            def _llm_type(self) -> str:
                return "custom-simple-llm"

        qa_chain = RetrievalQA.from_chain_type(
            llm=SimpleLLM(),
            chain_type="stuff",
            retriever=retriever,
        )

        query = st.text_input("Ask something about the document:")
        if query:
            result = qa_chain.run(query)
            st.write("📣", result)
    else:
        st.info("Please preprocess the document in the Preprocess tab.")
