from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from chunk_text import split_text
# Path to cleaned text
text_file = "/Users/siva-16986/Desktop/Projects/RAG_mini_model/The-Jungle-Books-text.txt"

# Read text
with open(text_file, "r", encoding="utf-8") as f:
    full_text = f.read()
# Split into chunks
chunks = split_text(full_text)
print(f"Total chunks: {len(chunks)}")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert chunks to embeddings
embeddings = model.encode(chunks, convert_to_numpy=True)
print("Embedding shape:", embeddings.shape)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index & chunks
faiss.write_index(index, "book_index.faiss")
np.save("chunks.npy", np.array(chunks, dtype=object))

print("Embeddings created and stored in FAISS successfully.")