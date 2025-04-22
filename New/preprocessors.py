# preprocessors.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def preprocess(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents([Document(page_content=raw_text)])
    return docs
