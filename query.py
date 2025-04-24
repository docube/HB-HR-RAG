from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

# Load the saved vector store
db = FAISS.load_local("vector_store", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# Ask a question
query = "What does the HR policy say about leave days?"
results = db.similarity_search(query, k=3)

# Print the matched documents
for doc in results:
    print(doc.page_content)
    print("Metadata:", doc.metadata)
