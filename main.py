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
    allow_origins=[
        "http://localhost:3000", 
        "https://www.inkndyes.com", 
        "https://inkndyes.com"
    ],
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


@app.get("/")
def health_check():
    return {"status": "running"}


@app.post("/chat")
def chat(question: str = Form(...)):
    try:
        model = get_model()
        index, chunks = get_index_and_chunks()

        embedded = model.encode([question])
        _, I = index.search(np.array(embedded), k=5)
        context = "\n".join([chunks[i] for i in I[0]])

        prompt = (
            "You are a helpful assistant answering based only on the given context. "
            "If the answer isn't in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return {"answer": response.choices[0].message.content.strip()}
    except Exception as e:
        print(f"❌ ERROR in /chat: {e}")
        return {"answer": "⚠️ Internal Server Error"}
