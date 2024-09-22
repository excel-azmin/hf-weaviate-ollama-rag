# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from hf_pipeline import build_rag_pipeline

app = FastAPI()

class Question(BaseModel):
    query: str

@app.post("/ask")
def ask_question(question: Question):
    # Build the RAG pipeline
    rag_pipeline = build_rag_pipeline()

    # Run the pipeline to get an answer
    answer = rag_pipeline(question.query)

    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
