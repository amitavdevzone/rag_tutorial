import csv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import os

def ingest_csv_to_chroma(csv_path):
    messages = []
    local_embeddings = OllamaEmbeddings(model="llama3.1:8b")
    persist_directory = "db-intents"
    
    if os.path.exists(persist_directory):
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=local_embeddings)
        messages.append("Loaded the existing Chroma DB")
    else:
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=local_embeddings)
        messages.append("Created the Chroma DB")
    
    # Load data from CSV
    documents = []
    with open(csv_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            sentence = row['sentence']
            intent = row['intent']
            # Create a Document object with metadata for intent data
            doc = Document(page_content=sentence, metadata={"intent": intent, "type": "intent_data"})
            documents.append(doc)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(documents)
    
    vectorstore.add_documents(documents=all_splits)
    messages.append("Added intent data to Chroma DB")
    
    return messages

# Example usage
csv_path = "data/intent.csv"

# Ingest CSV data
csv_messages = ingest_csv_to_chroma(csv_path)
for message in csv_messages:
    print(message)
