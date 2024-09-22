import json
from weaviate import Client as WeaviateClient
from sentence_transformers import SentenceTransformer
import requests

def build_rag_pipeline(weaviate_url="http://localhost:8080", ollama_url="http://localhost:11434/api/generate"):
    # Initialize Weaviate client
    weaviate_client = WeaviateClient(weaviate_url)

    # Initialize the embedding model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def rag_pipeline(question):
        # Retrieve documents from Weaviate based on the question
        query_embedding = embedder.encode(question).tolist()
        response = weaviate_client.query.get("Document", ["content"]) \
            .with_near_vector({"vector": query_embedding}) \
            .with_limit(5) \
            .do()

        # Combine the content from the retrieved documents
        context = " ".join([doc["content"] for doc in response["data"]["Get"]["Document"]])

        # Prepare the payload for the Ollama API
        payload = {
            "model": "llama3.1",
            "prompt": f"Based on the following context, answer the question: {context}\n\nQuestion: {question}"
        }

        # Send the context and question to the Ollama LLM for generating an answer
        ollama_response = requests.post(
            ollama_url,
            json=payload,
            stream=True  # Enable streaming response
        )

        # Check the status code
        if ollama_response.status_code == 200:
            ollama_answer = ""
            # Stream and concatenate the response chunks
            for line in ollama_response.iter_lines():
                if line:
                    # Decode JSON from each line
                    json_line = line.decode('utf-8')
                    response_part = json.loads(json_line).get("response", "")
                    ollama_answer += response_part
                    print(response_part, end="", flush=True)  # Print each part as it arrives

            return ollama_answer
        else:
            # Handle non-200 responses
            print(f"Unexpected response: {ollama_response.status_code}")
            print("Raw response:", ollama_response.text)
            return "Received an unexpected response from the server."

    return rag_pipeline

if __name__ == "__main__":
    # Example usage
    pipeline = build_rag_pipeline()
    question = "What is DevOPS?"
    answer = pipeline(question)
    print("\nFull Answer:", answer)
