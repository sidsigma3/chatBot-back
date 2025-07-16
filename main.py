from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import faiss
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env variables

app = FastAPI()

# CORS setup for frontend (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Change if deploying elsewhere
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load model and index to reduce memory usage
@lru_cache()
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@lru_cache()
def get_index_and_chunks():
    index = faiss.read_index("data/vector.index")
    with open("data/chunks.txt", "r", encoding="utf-8") as f:
        chunks = f.read().split("\n---\n")
    return index, chunks

@app.post("/chat")
def chat(question: str = Form(...)):
    model = get_model()
    index, chunks = get_index_and_chunks()

    # Step 1: Embed question
    embedded = model.encode([question])

    # Step 2: Search similar chunks
    _, I = index.search(np.array(embedded), k=5)
    context = "\n".join([chunks[i] for i in I[0]])

    # Step 3: Build prompt
    prompt = (
        "You are a helpful assistant answering based only on the given context. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    # Step 4: Get response from OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return {"answer": response.choices[0].message.content.strip()}
