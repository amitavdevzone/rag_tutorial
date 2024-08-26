from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os

def fetch_and_persist_article(url):
    messages = []
    local_embeddings = OllamaEmbeddings(model="llama3.1:8b")
    persist_directory = "db"
    
    if os.path.exists(persist_directory):
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=local_embeddings)
        messages.append(f"Loaded the existing Chroma DB")
    else:
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=local_embeddings)
        messages.append(f"Created the Chroma DB")
    
    loader = WebBaseLoader(url)
    data = loader.load()
    messages.append(f"URL Loaded")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    
    vectorstore.add_documents(documents=all_splits)
    messages.append(f"Added to Chroma DB")
    
    return messages
    
    
