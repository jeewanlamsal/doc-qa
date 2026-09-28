# Document Q&A — RAG System

A Retrieval-Augmented Generation (RAG) application that lets you 
upload any PDF and ask questions about it in plain English. Built 
with LangChain, ChromaDB, Google Gemini, and Streamlit.

## Demo

Upload a PDF → Ask questions → Get grounded answers with sources

## What it does

- Upload any PDF through a clean web interface
- Automatically reads, chunks, and embeds the document
- Ask questions in natural language
- Get accurate answers grounded in the document content
- Shows source chunks used to generate each answer
- Refuses to hallucinate — says "I don't know" when the 
  answer isn't in the document
- Each document gets its own isolated vector store — 
  no cross-contamination between uploads

## Architecture
PDF → Text Extraction → TOC Filtering → Chunking → Deduplication
↓
Gemini Embeddings
↓
ChromaDB
↓
Question → Multi-Query Retrieval → MMR Search → Top Chunks
↓
Prompt Assembly
↓
Gemini LLM → Answer + Sources


## Tech Stack

| Component | Technology |
|---|---|
| RAG Pipeline | LangChain |
| LLM | Google Gemini (free tier) |
| Embeddings | Google Gemini Embeddings |
| Vector Store | ChromaDB (local) |
| UI | Streamlit |
| PDF Parsing | PyPDF |

## Project Structure
doc-qa/
├── app.py # Streamlit UI
├── rag/
│ ├── loader.py # PDF loading and validation
│ ├── embedder.py # Chunking, embedding, ChromaDB
│ └── chain.py # LangChain Q&A chain
├── uploads/ # PDF storage (git-ignored)
├── chroma_db/ # Vector database (git-ignored)
├── .env # API keys (git-ignored)
└── requirements.txt


## Setup

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/doc-qa.git
cd doc-qa
```

**2. Create and activate virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Get a free Gemini API key**

Go to [aistudio.google.com](https://aistudio.google.com), sign in 
with Google, click **Get API Key**.

**5. Create `.env` file**
GOOGLE_API_KEY=your_key_here


**6. Create required folders**
```bash
mkdir uploads chroma_db
```

**7. Run the app**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Key Features Explained

**Rate-limit-aware embedding** — Embeds in batches of 20 with 
15-second pauses to stay within Gemini's free tier limits.

**Per-document vector isolation** — Each PDF gets its own 
ChromaDB collection identified by an MD5 hash of the filename. 
Switching documents never contaminates results.

**TOC filtering** — Automatically detects and removes table of 
contents pages that would otherwise dominate similarity searches.

**Multi-query retrieval** — Generates multiple phrasings of 
each question to improve recall for indirect or vague queries.

**Model fallback chain** — If the primary Gemini model is 
unavailable or rate-limited, automatically falls back through 
a chain of alternative models.

