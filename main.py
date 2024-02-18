from google.cloud import storage
import pandas as pd 
from langchain_google_vertexai import VertexAIEmbeddings 
from langchain_community.vectorstores import MatchingEngine
import numpy as np 
import csv
import os

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

class CharacterTextSplitter:
    def __init__(self, separator="\n", chunk_size=500, chunk_overlap=0, length_function=len):
        self.separator = separator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
    
    def split_documents(self, documents):
        # This is a simplified placeholder implementation
        split_docs = []
        for doc in documents:
            # Split document text into chunks based on chunk_size
            text = doc.page_content
            chunks = [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
            for chunk in chunks:
                split_docs.append(Document(page_content=chunk, metadata=doc.metadata))
        return split_docs

def download_csv_from_gcs(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded storage object {source_blob_name} from bucket {bucket_name} to local file {destination_file_name}.")

def parse_csv_and_chunk_documents_test_test(file_path):
    docs = []
    columns_to_embed = ["Workload use cases"]
    columns_to_metadata = ["Instance", "Family", "Workload types", "CPU types", "vCPUs", "vCPU definition", "Memory", "Max GPUs"]
    
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    for index, row in df.iterrows():
        to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        values_to_embed = {k: row[k] for k in columns_to_embed if k in row}
        to_embed = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values_to_embed.items())
        newDoc = Document(page_content=to_embed, metadata=to_metadata)
        docs.append(newDoc)

    # Chunking documents
    splitter = CharacterTextSplitter()
    documents = splitter.split_documents(docs)
    return documents

def parse_csv_and_chunk_documents(file_path):
    docs = []
    columns_to_embed = ["Workload use cases"]
    columns_to_metadata = ["Instance", "Family", "Workload types", "CPU types", "vCPUs", "vCPU definition", "Memory", "Max GPUs"]
    with open(file_path, newline="", encoding='utf-8-sig') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for i, row in enumerate(csv_reader):
            to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
            values_to_embed = {k: row[k] for k in columns_to_embed if k in row}
            to_embed = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values_to_embed.items())
            newDoc = Document(page_content=to_embed, metadata=to_metadata)
            print(newDoc)
            docs.append(newDoc)
    splitter = CharacterTextSplitter(separator = "\n",
                                chunk_size=500, 
                                chunk_overlap=0,
                                length_function=len)
    documents = splitter.split_documents(docs)
    return documents

def embed_documents_with_vertex_ai(documents):
    # Placeholder for embedding logic using Google Vertex AI PaLM
    embeddings = VertexAIEmbeddings()
    docs = [embeddings.embed_document(document.page_content) for document in documents]  # Placeholder
    return embeddings

def store_embeddings_in_google_vector_store(embeddings):
    vector_store = MatchingEngine()  # Hypothetical initialization
    for embedding in embeddings:
        vector_store.add(np.array(embedding).tolist())  # Convert numpy array to list if necessary
    print("Embeddings have been stored in Google Vector Store.")

def main():
    bucket_name = 'gke-opt-demo-compute-choose-doc'
    source_blob_name = 'choose-compute.csv'
    destination_file_name = 'choose-compute.csv'

    # Step 1: Download the CSV
    download_csv_from_gcs(bucket_name, source_blob_name, destination_file_name)

    # Step 2: Parse CSV and chunk documents
    documents = parse_csv_and_chunk_documents(destination_file_name)

    # Step 3: Embed documents with Vertex AI
    embeddings = embed_documents_with_vertex_ai(documents)

    # Step 4: Store embeddings in Google Vector Store
    #store_embeddings_in_google_vector_store(embeddings)

if __name__ == "__main__":
    main()
