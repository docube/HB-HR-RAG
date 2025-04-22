from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever,
)

query = st.text_input("Ask a question about your document")

if query:
    result = qa_chain.run(query)
    st.write("📣 Answer:", result)
