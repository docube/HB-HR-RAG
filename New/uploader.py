# uploader.py

import streamlit as st
from preprocessors import preprocess
from io import StringIO
import docx
import PyPDF2

# File upload
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

    st.success("File uploaded and text extracted!")

    # 👇 Call the preprocessor
    docs = preprocess(raw_text)

    st.write("✅ Document chunked into", len(docs), "chunks.")
    st.write("📄 Example chunk:", docs[0].page_content)
