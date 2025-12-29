# RAG_mini_model

        Mini RAG project to query 'The Jungle Book' using embeddings and FAISS.

A mini **RAG (Retrieve and Generate)** system that allows users to ask questions about The Jungle Book and get answers generated from the book text using embeddings, FAISS, and a language model.

**Quick Start (one-liner):**

_📄🤖 Old_Version:_

        https://ragminimodel-dwoq5cdwggvrparzstxtgc.streamlit.app/

_🚀 New_version_0.1 (# Live Demo)_

        https://ragminimodel-5biwrbtnflgtyfvcsyukni.streamlit.app/

        
**Quick Demo**

<img width="1493" height="813" alt="Screenshot 2025-12-29 at 6 49 27 PM" src="https://github.com/user-attachments/assets/3f960882-9186-4b84-bfbf-a4c88dabadd2" />


**Features**

        * Extracts text from PDF and cleans it.
        * Splits the text into overlapping chunks for embeddings.
        * Converts text chunks into vector embeddings using SentenceTransformers.
        * Stores embeddings in FAISS for efficient similarity search.
        * Streamlit web app for querying the book.
        * Uses Groq LLM to generate answers from retrieved context.
        * Provides short, concise answers (3–5 lines) only using the story content.

## 📂 Project Structure

        - `app.py` – Initial version (basic RAG flow)
        - `app_v0.1.py` – Updated version User Input and ask query from the given documents (button-based query handling, improved UX)
        - `chunk_text.py` – Text chunking logic
        - `requirements.txt` – Project dependencies


**Folder / File Structure**

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

**Setup Instructions**

**_1. Create and activate Conda environment_**

        conda create -n mini_rag python=3.9 -y
        conda activate mini_rag

**_2. Install dependencies_**

        pip install -r requirements.txt

**_3. Add Groq API key_**

* Create secrets.toml in project folder:
        
        [GROQ_API_KEY]
        value = "your_groq_api_key_here"

**Usage Steps**

**_1. Extract text from PDF_**

      python extract_text.py

**_2. Split text into chunks_**

      python chunk_text.py

**_3. Generate embeddings & build FAISS index_**

      python embeddings_store.py

**_4. Run Streamlit app_**

      streamlit run app.py

* Open the browser, type your question, and get an answer.

**Example Query**

  _**Question:**_
  
              "Who is teacher of  Mowgli?"
              
  _**Answer:**_
  
              "Bagheera is Mowgli's teacher. He teaches Mowgli and is very protective of him. He warns Mowgli many times about his enemy, Shere Khan. Bagheera is like a parent to Mowgli, and Mowgli listens to him and respects him."

**Notes**

  * Retrieves the most relevant text chunk from the book.
  * Answers are short and context-specific.
  * Responds with:
        
        "The story does not clearly mention this information."
  
    if the answer isn’t present in the book.

**Dependencies**

        * Python 3.9+
        * Streamlit
        * SentenceTransformers
        * PyTorch
        * FAISS
        * NumPy
        * pdfplumber
        * Groq Python SDK
