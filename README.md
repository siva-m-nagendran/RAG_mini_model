# RAG_mini_model
Mini RAG project to query 'The Jungle Book' using embeddings and FAISS.

Mini_RAG - Story Query

A mini RAG (Retrieve and Generate) system that allows users to ask questions about The Jungle Book and get answers generated from the book text using embeddings, FAISS, and a language model.

Quick Demo
<img width="1493" height="762" alt="&#39;The Jungle Book&#39; - Story Query" src="https://github.com/user-attachments/assets/e023e194-4d0b-4b12-9d7a-84538bf79218" />

Quick Start (one-liner):

https://ragminimodel-dpw4bn8yefguvhhztezldd.streamlit.app/

Features
* Extracts text from PDF and cleans it.
* Splits the text into overlapping chunks for embeddings.
* Converts text chunks into vector embeddings using SentenceTransformers.
* Stores embeddings in FAISS for efficient similarity search.
* Streamlit web app for querying the book.
* Uses Groq LLM to generate answers from retrieved context.
* Provides short, concise answers (3–5 lines) only using the story content.

Folder / File Structure

Mini_RAG/
│
├── extract_text.py
├── chunk_text.py
├── embeddings_store.py
├── app.py
├── query_book.py      # Optional / alternative Streamlit app
├── The-Jungle-Books-text.pdf
├── book_index.faiss
├── chunks.npy
├── requirements.txt
└── README.md

Setup Instructions
1. Clone the repository

git clone <your-repo-url>
cd Mini_RAG
1. Create and activate Conda environment

conda create -n mini_rag python=3.9 -y
conda activate mini_rag
1. Install dependencies

pip install -r requirements.txt
1. Add Groq API key
* Create secrets.toml in project folder:

[GROQ_API_KEY]
value = "your_groq_api_key_here"

Usage Steps
1. Extract text from PDF

python extract_text.py
2. Split text into chunks

python chunk_text.py
3. Generate embeddings & build FAISS index

python embeddings_store.py
4. Run Streamlit app

streamlit run app.py
* Open the browser, type your question, and get an answer.

Example Query
Question: "Who teaches Mowgli the ways of the jungle?"
Answer: "Bagheera, the black panther, teaches Mowgli the ways of the jungle and keeps him safe from danger."

Notes
* Retrieves the most relevant text chunk from the book.
* Answers are short and context-specific.
* Responds with:

"The story does not clearly mention this information."
if the answer isn’t present in the book.

Dependencies
* Python 3.9+
* Streamlit
* SentenceTransformers
* PyTorch
* FAISS
* NumPy
* pdfplumber
- [ ] Groq Python SDK
