#code for the chatbot(rag)
#line number 44 for api key 
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# CORS for frontend (adjust if hosted differently)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # adjust if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load embedding model, FAISS index and text chunks
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("data/vector.index")

with open("data/chunks.txt", "r", encoding="utf-8") as f:
    chunks = f.read().split("\n---\n")

@app.post("/chat")
def chat(question: str = Form(...)):
    # Step 1: Embed the question
    embedded = model.encode([question])
    
    # Step 2: Search similar chunks
    _, I = index.search(np.array(embedded), k=5)
    context = "\n".join([chunks[i] for i in I[0]])

    # Step 3: Compose the prompt
    prompt = (
        f"You are a helpful assistant answering based only on the given context. "
        f"If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    # client = OpenAI()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Step 4: Query OpenAI
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return {"answer": response.choices[0].message.content.strip()}
