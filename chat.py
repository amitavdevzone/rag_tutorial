from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def answer_question_with_context(question):
    messages = []
    persist_directory = "db"
    local_embeddings = OllamaEmbeddings(model="llama3.1:8b")
    
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=local_embeddings)
    
    docs = vectorstore.similarity_search(question)
    if not docs:
        messages.append("No relevant information was found")
        return
    
    # Define the RAG prompt template
    RAG_TEMPLATE = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Answer in about 3 lines and keep the answer concise.

    <context>
    {context}
    </context>

    Answer the following question:

    {question}"""
    
    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    model = ChatOllama(model="llama3.1:8b")
    
    chain = (
        RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
        | rag_prompt
        | model
        | StrOutputParser()
    )
    
    response = chain.invoke({"context": docs, "question": question})
    return {"response": response, "messages": messages}
