
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import json


# STEP 1: Extract text from the PDF
def extract_pages(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


# STEP 2: Split the text into chunks
def split_into_chunks(pages, chunk_size=1000, overlap=150):
    chunks = []

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "page": page["page"],
                    "text": chunk_text
                })

            if end == len(text):
                break

            start = end - overlap

    return chunks


# STEP 3: Load the embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


# STEP 4: Extract and split the PDF
pages = extract_pages("sample file.pdf")
chunks = split_into_chunks(pages)

print("Total chunks:", len(chunks))


# STEP 5: Convert each chunk into an embedding
texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

print("Embedding generation complete!")
print("Embedding matrix shape:", embeddings.shape)


# STEP 6: Save chunks and embeddings
with open("chunks.json", "w", encoding="utf-8") as file:
    json.dump(chunks, file, ensure_ascii=False, indent=2)

np.save("embeddings.npy", embeddings)

print("Saved chunks to chunks.json")
print("Saved embeddings to embeddings.npy")
