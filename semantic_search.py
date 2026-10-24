
import json
import numpy as np
from sentence_transformers import SentenceTransformer


# STEP 1: Load the saved text chunks
with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

# STEP 2: Load the saved embeddings
embeddings = np.load("embeddings.npy")

# STEP 3: Load the same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# STEP 4: Create a search function
def search_pdf(question, top_k=3):

    # Convert the user's question into an embedding
    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    # Compare the question with every saved chunk
    similarities = embeddings @ question_embedding

    # Get the indices of the best matches
    best_indices = np.argsort(similarities)[::-1][:top_k]

    # Display the best matching passages
    print("\nMost relevant passages:")

    for rank, index in enumerate(best_indices, start=1):
        chunk = chunks[int(index)]
        score = similarities[index]

        print(f"\n--- Result {rank} ---")
        print(f"PDF Page: {chunk['page']}")
        print(f"Similarity Score: {score:.4f}")
        print(chunk["text"])


# STEP 5: Ask the user a question
question = input("Ask a question about your PDF: ")

search_pdf(question)
