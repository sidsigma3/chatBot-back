import os
import fitz  # PyMuPDF for PDFs
import faiss
import numpy as np
import docx
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Input folder
data_dir = "Mo ink n dye"  # Folder containing your .docx and .pdf files
chunks = []
chunk_sources = []

# Function to read PDF file
def read_pdf(path):
    doc = fitz.open(path)
    text = " ".join([page.get_text() for page in doc])
    doc.close()
    return text

# Function to read Word document
def read_docx(path):
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

# Smart chunking function based on Q&A structure
import re
def split_into_chunks(text, max_length=1000):
    qa_blocks = re.split(r'(?=\n?\d+\.\s)', text)
    chunks = []

    for block in qa_blocks:
        block = block.strip()
        if not block:
            continue

        if len(block) <= max_length:
            chunks.append(block)
        else:
            # Split long block by paragraph
            paragraphs = block.split("\n")
            temp_chunk = ""
            for para in paragraphs:
                if len(temp_chunk) + len(para) + 1 <= max_length:
                    temp_chunk += para + "\n"
                else:
                    chunks.append(temp_chunk.strip())
                    temp_chunk = para + "\n"
            if temp_chunk:
                chunks.append(temp_chunk.strip())

    return chunks

# Read and process all files in the folder
for filename in os.listdir(data_dir):
    if filename.startswith("~$"):
        print(f"Skipping temporary file: {filename}")
        continue

    path = os.path.join(data_dir, filename)
    print(f"Processing: {filename}")

    try:
        if filename.lower().endswith(".pdf"):
            text = read_pdf(path)
        elif filename.lower().endswith(".docx"):
            text = read_docx(path)
        else:
            print(f"Skipping unsupported file: {filename}")
            continue

        file_chunks = split_into_chunks(text)
        chunks.extend(file_chunks)
        chunk_sources.extend([filename] * len(file_chunks))

    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")

# Save chunks
os.makedirs("data", exist_ok=True)

with open("data/chunks.txt", "w", encoding="utf-8") as f:
    f.write("\n---\n".join(chunks))

with open("data/sources.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(chunk_sources))

# Embed and index
print("Embedding and indexing chunks...")
vectors = model.encode(chunks)
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(np.array(vectors))
faiss.write_index(index, "data/vector.index")

print(f"✅ Embedded {len(chunks)} chunks from {len(set(chunk_sources))} files.")
