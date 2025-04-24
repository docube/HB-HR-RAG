# query.py

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
load_dotenv()

# Step 1: Load the vector store
db = FAISS.load_local("vector_store", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# Step 2: Ask a question
query = "What does the HR policy say about leave days?"
results = db.similarity_search(query, k=3)

# Step 3: Show the matched documents
print("\n📄 Top Matching Documents:")
for doc in results:
    print("\n---")
    print(doc.page_content)
    print("Metadata:", doc.metadata)

# Step 4: Use LLM to refine the answer
llm = ChatOpenAI()
chain = load_qa_chain(llm, chain_type="stuff")
answer = chain.run(input_documents=results, question=query)

# Step 5: Print the final answer
print("\n🧠 Final Answer:")
print(answer)
