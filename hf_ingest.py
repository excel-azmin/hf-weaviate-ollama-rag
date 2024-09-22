from weaviate import Client as WeaviateClient
from sentence_transformers import SentenceTransformer
import os

def convert_pdf_to_text(pdf_path):
    # Placeholder implementation; replace with actual PDF to text conversion logic
    return "Converted text from PDF."

def split_text_into_chunks(text, chunk_size=500, overlap=50):
    # Split the text into overlapping chunks for better retrieval performance
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def ingest_documents(pdf_paths, weaviate_url="http://localhost:8080"):
    # Initialize Weaviate client
    weaviate_client = WeaviateClient(weaviate_url)

    # Initialize the embedding model
    embedder = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=None)

    # Delete the "Document" class if it exists
    schema = weaviate_client.schema.get()
    class_names = [cls['class'] for cls in schema['classes']]
    if "Document" in class_names:
        weaviate_client.schema.delete_class("Document")
        print('Deleted existing "Document" class.')

    # Create the "Document" class
    class_obj = {
        "class": "Document",
        "description": "A class to store text documents and their embeddings",
        "properties": [
            {
                "name": "content",
                "description": "The textual content of the document",
                "dataType": ["text"]
            },
            {
                "name": "embedding",
                "description": "The vector embedding of the document",
                "dataType": ["number[]"]
            }
        ]
    }
    weaviate_client.schema.create_class(class_obj)
    print('Created new "Document" class.')

    # Loop through each PDF, convert it to text, split it, and ingest it into Weaviate
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"File {pdf_path} does not exist.")
            continue

        text = convert_pdf_to_text(pdf_path)
        chunks = split_text_into_chunks(text)

        # Generate embeddings and store in Weaviate
        for chunk in chunks:
            embedding = embedder.encode(chunk)
            data_object = {
                "content": chunk,
                "embedding": embedding.tolist()
            }
            weaviate_client.data_object.create(
                data_object=data_object,
                class_name="Document"
            )

    print("Documents have been ingested into Weaviate.")

if __name__ == "__main__":
    # Example usage
    pdf_paths = ["data/Engineering-DevOps.pdf"]
    ingest_documents(pdf_paths)