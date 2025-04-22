from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document  # or from langchain.schema import Document

def process_raw_text(raw_text):
    doc = Document(page_content=raw_text)

    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    docs = text_splitter.split_documents([doc])
    return docs
