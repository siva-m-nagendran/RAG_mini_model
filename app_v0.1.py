import streamlit as st # create the web UI (buttons, inputs, text) and etc
import pdfplumber # Library used to read and extract text from PDF files
from chunk_text import split_text # Breaks large text into smaller overlapping chunks
from sentence_transformers import SentenceTransformer # Loads the embedding model that converts text → numbers (vectors)
import faiss # FAISS is used for fast similarity search between embeddings.
import numpy as np # NumPy handles numerical arrays (required for FAISS & embeddings)
from groq import Groq # Groq SDK to call the LLM (LLaMA 3) for generating answers

# Initialize Groq client
client = Groq(api_key = st.secrets["GROQ_API_KEY"]) # API key is stored in secrets.toml

# -------- STREAMLIT PAGE SETTINGS --------
st.set_page_config(page_title="Mini RAG", layout = "centered") # Browser tab title -> Mini RAG and Page layout -> centered
st.title("(RAG_Model)_ Document Q&A") # Shows a big heading on the webpage.

# -------- File Upload --------
uploaded_file = st.file_uploader(
    "Upload a PDF or TXT file",
    type=["pdf", "txt"]
) # Accepts PDF or TXT and Stores the uploaded file in uploaded_file

# -------- Utility Functions --------
def clean_text(text): # Function to normalize text
    text = text.lower() # Converts all text to lowercase (better embeddings).
    text = text.replace("\n", " ") # Removes newlines and tabs.
    text = text.replace("\t", " ") # Removes newlines and tabs.
    return " ".join(text.split())  # Removes extra spaces and returns clean text.


# -------- PDF TEXT EXTRACTION --------
def extract_text_from_pdf(uploaded_file): # Function to extract text page-by-page from PDF.
    all_text = "" # Empty string to collect all pages’ text.
    with pdfplumber.open(uploaded_file) as pdf: # Opens the uploaded PDF safely.
        for page in pdf.pages: # Loops through each page.
            page_text = page.extract_text() # Extracts readable text from the page.
            if page_text: # Checks if the page actually contains text.
                all_text += clean_text(page_text) + "\n" # Cleans page text and appends it.
    return all_text # Returns full cleaned PDF text.


# -------- Load Model (Cached) --------
@st.cache_resource # Streamlit cache decorator: Model loads only once and Improves performance
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2") # Loads a free & fast embedding model.

model = load_model() # Calls the function and stores the model.

# -------- Process File --------
if uploaded_file: # Runs only after user uploads a file.
    st.success("File uploaded successfully!") # Shows green success message.

    if uploaded_file.type == "application/pdf": # Checks if the file is a PDF.
        text = extract_text_from_pdf(uploaded_file) # Extracts text from PDF.
    else:                                                                   #<-|
        text = clean_text(uploaded_file.read().decode("utf-8")) # Reads and cleans TXT file.
# TEXT CHUNKING
    chunks = split_text(text) # Breaks large text into small overlapping chunks.
    st.write(f"Total Chunks: {len(chunks)}") # Displays number of chunks in UI.
# CREATE EMBEDDINGS
    embeddings = model.encode(chunks, convert_to_numpy=True) # Converts each chunk into vector embeddings.

    dimension = embeddings.shape[1] # Gets vector size (e.g., 384).
    index = faiss.IndexFlatL2(dimension) # Creates FAISS index using Euclidean distance.
    index.add(embeddings) # Stores all embeddings inside FAISS.

    # -------- User Question --------
    query = st.text_input("Ask your question:") # Creates a question input box.
    # Creates a button labeled "Get Answer" in the Streamlit UI
    # The button returns True only when it is clicked
    ask_btn = st.button("Get Answer")

    if ask_btn and query:
        with st.spinner("Searching and generating answer..."):
            query_embedding = model.encode([query], convert_to_numpy=True)  # Converts the user's question into a numerical vector (embedding)
            # Performs similarity search using FAISS
            # k=1 means retrieve the single most relevant chunk
            # distances -> similarity score
            # indices -> index position of the best matching chunk
            distances, indices = index.search(query_embedding, k=1)
            # Retrieves the most relevant text chunk using the FAISS result index
            context = chunks[indices[0][0]]

# PROMPT CREATION
        prompt = f"""
You are a knowledgeable AI assistant designed to answer questions strictly
based on the provided document content.

Context:
{context}

User Question:
{query}

Guidelines:
- Use ONLY the information available in the context.
- Provide a clear, concise, and professional answer.
- Limit the response to 3–5 well-structured sentences.
- Do NOT add assumptions or external knowledge.
- If the answer is not found in the context, reply exactly:
  "The document does not clearly mention this information."

Answer:
"""

        response = client.chat.completions.create( # Sends prompt to Groq LLM.
            model="llama-3.1-8b-instant", # Uses LLaMA-3 (8B) model.
            messages=[{"role": "user", "content": prompt}], #Sends prompt as user message.
            temperature=0.3 # Low creativity -> factual answers.
        )
# DISPLAY ANSWER
        st.subheader("Answer") # Displays heading.
        st.write(response.choices[0].message.content) # Shows LLM’s final answer.
