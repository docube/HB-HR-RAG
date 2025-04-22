import streamlit as st
from preprocessors import process_raw_text  # Import your function

# Upload file
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    # Read file content based on type
    file_type = uploaded_file.type

    if file_type == "text/plain":
        raw_text = uploaded_file.read().decode("utf-8")

    elif file_type == "application/pdf":
        import PyPDF2
        reader = PyPDF2.PdfReader(uploaded_file)
        raw_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        import docx
        doc = docx.Document(uploaded_file)
        raw_text = "\n".join([para.text for para in doc.paragraphs])

    else:
        st.error("Unsupported file type.")
        raw_text = ""

    # Call preprocessing function
    if raw_text:
        docs = process_raw_text(raw_text)

        # Optional: display chunks
        st.write(f"Document split into {len(docs)} chunks:")
        for i, doc in enumerate(docs):
            st.write(f"Chunk {i+1}:\n{doc.page_content}")
