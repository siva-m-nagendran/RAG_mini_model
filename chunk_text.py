def split_text(text, chunk_size=300, overlap=50):  # smaller chunk size
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap  # overlap between chunks

    return chunks


# 🔹 Test code (runs ONLY when this file is executed directly)
if __name__ == "__main__":
    input_text = "/Users/siva-16986/Desktop/Projects/RAG_mini_model/The-Jungle-Books-text.txt"
    with open(input_text, "r", encoding="utf-8") as f:
        full_text = f.read()

    chunks = split_text(full_text)
    print(f"Total chunks created: {len(chunks)}")
    print("\nSample chunk:\n")
    print(chunks[0])
