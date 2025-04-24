from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Load the saved vector store
db = FAISS.load_local("vector_store", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
