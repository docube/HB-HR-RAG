# Create the full project scaffold for a Streamlit RAG app with 3 tabs:
# 1. Upload multiple files
# 2. Preprocess and create vectorstore
# 3. Query using OpenAI LLM (RetrievalQA)

# We'll create a single streamlit app with tabbed layout and explanations for OpenAI key setup

scaffold_code = """
import streamlit as st
from io import StringIO
import os

# Langchain and OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

import PyPDF2
import docx

st.set_page_config(page_title="HB-HR RAG", layout="wide")

st.title("📄 HB-HR Document Q&A System")

tab1, tab2, tab3 = st.tabs(["📤 Upload Files", "🔧 Preprocess & Embed", "💬 Ask Questions"])

# Shared state
if "docs" not in st.session_state:
    st.session_state.docs = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# TAB 1: Upload Files
with tab1:
    st.header("📤 Upload Multiple Documents")
    uploaded_files = st.file_uploader("Upload your files (PDF, DOCX, TXT)", 
                                      type=["pdf", "docx", "txt"], 
                                      accept_multiple_files=True)

    combined_text = ""
    if uploaded_files:
        for file in uploaded_files:
            if file.type == "application/pdf":
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    combined_text += page.extract_text() + "\\n"
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = docx.Document(file)
                for para in doc.paragraphs:
                    combined_text += para.text + "\\n"
            elif file.type == "text/plain":
                stringio = StringIO(file.getvalue().decode("utf-8"))
                combined_text += stringio.read() + "\\n"
        
        st.session_state.raw_text = combined_text
        st.success("✅ Files uploaded successfully!")

# TAB 2: Preprocess and Embed
with tab2:
    st.header("🔧 Preprocess Text and Create Embeddings")

    if "raw_text" in st.session_state and st.session_state.raw_text:
        st.subheader("📄 Raw Text Sample:")
        st.text(st.session_state.raw_text[:1000])  # show only a sample
        
        if st.button("➡️ Preprocess and Embed"):
            text_splitter = CharacterTextSplitter(
                separator="\\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )

            texts = text_splitter.split_text(st.session_state.raw_text)
            docs = [Document(page_content=t) for t in texts]
            st.session_state.docs = docs

            # OpenAI Embeddings
            openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                st.error("❌ Please set your OpenAI API key in .env or Streamlit Secrets.")
            else:
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                vectorstore = FAISS.from_documents(docs, embeddings)
                st.session_state.vectorstore = vectorstore
                vectorstore.save_local("vector_db")
                st.success("✅ Documents embedded and vectorstore created!")

    else:
        st.warning("⬅️ Please upload files first in the 'Upload' tab.")

# TAB 3: Ask Questions
with tab3:
    st.header("💬 Ask Questions from Uploaded Documents")
    query = st.text_input("Ask a question about your uploaded documents")

    if query and st.session_state.vectorstore:
        openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            st.error("❌ Please set your OpenAI API key.")
        else:
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key),
                chain_type="stuff",
                retriever=st.session_state.vectorstore.as_retriever(),
            )
            result = qa_chain.run(query)
            st.success("✅ Answer:")
            st.write(result)
    elif query:
        st.warning("Please preprocess and embed your documents first.")
"""

scaffold_code

