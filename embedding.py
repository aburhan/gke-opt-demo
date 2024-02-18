# Import necessary libraries
from google.cloud import storage
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
import csv
import os


# Step 1: Download CSV file from Google Cloud Storage
def download_csv_from_gcs(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded storage object {source_blob_name} from bucket {bucket_name} to local file {destination_file_name}.")

# Step 2: Load the CSV file
def load_csv_file(file_path):
    csv_loader = CSVLoader(file_path="choose-compute.csv")
    document = csv_loader.load()
    print(document)
    return document

# Step 3: Chunk the document
def chunk_documen_simple(document):
    with open('choose-compute.csv') as f:
        long_text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )

    texts = text_splitter.create_documents([long_text])
    print(texts[0].page_content, "\n")
    print(texts[1].page_content)

def chunk_document(document):
    docs = []
    columns_to_embed = ["Workload use cases"]
    columns_to_metadata = ["Instance", "Family", "Workload types", "CPU types", "vCPUs", "vCPU definition", "Memory", "Max GPUs"]
    
    with open('choose-compute.csv', newline="", encoding='utf-8-sig') as csvfile:
        csv_reader = csv.DictReader(csvfile)
    for i, row in enumerate(csv_reader):
        to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        print(to_metadata)
        values_to_embed = {k: row[k] for k in columns_to_embed if k in row}
        to_embed = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values_to_embed.items())
        newDoc = Document(page_content=to_embed, metadata=to_metadata)
        docs.append(newDoc)
    
    splitter = CharacterTextSplitter(separator = "\n",
                                chunk_size=500, 
                                chunk_overlap=20,
                                length_function=len)
    documents = splitter.split_documents(docs)
    print(documents)
    return documents
    
# Step 4: Store embeddings in vector search
def embed_documents(document):
    embeddings = VertexAIEmbeddings()
    embeddings = [vertex_ai_embedding.embed_text(document.page_content) for document in documents] 
    return embeddings
    

# Main function to orchestrate the steps
def main():
    bucket_name = 'gke-opt-demo-compute-choose-doc'
    source_blob_name = 'choose-compute.csv'
    destination_file_name = 'choose-compute.csv'

    # Download CSV
    download_csv_from_gcs(bucket_name, source_blob_name, destination_file_name)
    
    # Load CSV
    document = load_csv_file(destination_file_name)
    
    # Chuck CSV
    chunk_document(document)
    # Embed documents
    text_embedding(document)
    #GoogleGenerativeAIEmbeddings
    embeddings = embed_documents(document)
    
    # Store embeddings
    #store_embeddings_in_vector_search(embeddings)

if __name__ == "__main__":
    main()
