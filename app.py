import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

# Initialize Groq client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Load resources ONLY once
@st.cache_resource
def load_resources():
    index = faiss.read_index("book_index.faiss")
    chunks = np.load("chunks.npy", allow_pickle=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    return index, chunks, model
index, chunks, model = load_resources()

# Streamlit WEB UI app

st.set_page_config(page_title = "Mini_RAG - Story_Book_Query", layout = "centered") # Given Browser tab title and also Alignment the Word
st.title("'The Jungle Book' - Story Query") # Big Heading on Webpage

# User Input "Asking a Questions"
query = st.text_input("Ask Your Questions: ") # Create a text input box

# Ask Button
#query = st.button("Get Answer") and query: # Created a button <--->
if query: # Runs when user presses Enter
    with st.spinner("Searching from the book and generating the answer......."): # Loading action just look like animation
        # Embed Query
        query_embeddings = model.encode([query], convert_to_numpy=True)

        #faiss searching
        k = 1 # -> Retrieve top 1 most similar chunks

        distances, indices = index.search(query_embeddings, k) # FAISS compares question embedding and Finds closest chunk embedding
            # retrieved_chunks = [chunks[i] for i in indices[0]] <---
            # context = "\n\n".join(retrieved_chunks) <---

        # distance --> similarity score
        # indices --> position of best chunk
        # Combine multiple chunks
    # Retrieve context
        context = chunks[indices[0][0]] # <--->

        # Prompt  -----> Controls LLM behavior
        prompt = f""" 
You are a knowledgeable story assistant.

Your task is to answer the user's question using ONLY the information
present in the provided story context.

Context:
{context}

User Question:
{query}

Instructions:
- Only use the story text.
- Keep the answer short and clear.
- Give 3–5 simple lines.
- If the answer isn’t in the story, say: "The story does not clearly mention this information."


Answer:
"""
        #LLM Calling
        chat_completion = client.chat.completions.create(  # ---> LLM calling via GROQ
            model = "llama-3.1-8b-instant", # ---> Uses llama3 model
            messages = [{"role":"user", "content":prompt}]  # ---> Sends the prompt as a user message and Receives the generated response from the model
        )
        answer = chat_completion.choices[0].message.content

        st.subheader("Answer")
        st.write(answer)
