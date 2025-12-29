# 🚗 Car Manual RAG Assistant

This is a specialized **Retrieval Augmented Generation (RAG)** system designed specifically for automotive owner's manuals. Unlike generic RAG pipelines, this project is engineered to handle the complex layout of car manuals—including dense specification tables, dashboard diagrams, and technical instructions.

It features a **Streamlit UI**, a custom **Evaluation Pipeline**, and a **Hybrid Search** engine that combines semantic understanding with keyword precision.

---

## ⚡ Key Features

*   **🧩 Multimodal Ingestion**  
    Extracts not just text, but also **Tables** (using Camelot) and **Images** (using PyMuPDF) to capture every detail.

*   **📊 Smart Table Chunking**  
    Uses a specialized chunking strategy for tables (smaller, context-rich chunks) vs. narrative text (larger chunks) to ensure specs like "Fuse Ratings" or "Tyre Pressures" are retrieved accurately.

*   **🖼️ Context-Aware Image Tagging**  
    Uses an LLM to generate descriptive functional tags for images (e.g., *"diagnostic connector location dashboard"*), making diagrams searchable.

*   **🔍 Hybrid Reranking**  
    Combines **Vector Search** (Bi-Encoder) with **Keyword Matching** (BM25-style boost) and a **Cross-Encoder Reranker** (`bge-reranker-base`). *.

*   **⚖️ LLM-as-a-Judge Evaluation**  
    Includes a full evaluation suite (`evaluate.py`) that generates synthetic ground-truth questions from your PDF and calculates **Hit Rate** and **MRR** (Mean Reciprocal Rank).

---

## 🛠️ Tech Stack

| Component | Technology                                         |
| :--- |:---------------------------------------------------|
| **UI** | Streamlit                                          |
| **LLM** | gpt-4o-mini (via  OpenAI LLM)                      |
| **Embeddings** | `BAAI/bge-small-en-v1.5` (384d)                    |
| **Reranker** | `BAAI/bge-reranker-base`                           |
| **Vector Store** | FAISS (IndexFlatIP)                                |
| **PDF Processing** | `pypdf`, `camelot-py` (tables), `pymupdf` (images) |

---

## 🚀 Getting Started

### 1. Prerequisites
You need **Python 3.10+**.

**⚠️ Important:** You must install **Ghostscript** for the table extraction (Camelot) to work.

*   **Mac**: `brew install ghostscript`
*   **Linux**: `sudo apt-get install ghostscript`
*   **Windows**: [Download Installer](https://www.ghostscript.com/download/gsdnld.html)

### 2. Installation

```bash
# Unzip the codebase
unzip car_manual_rag.zip

# Activate virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

3. Configuration
Set your API keys. You can do this in a .env file or export them in your terminal.
#  OpenAI Configuration
export OPENAI_API_KEY=""

4. Local Model Setup (Recommended)
To ensure stability and speed, it is recommended to download the embedding and reranking models locally instead of downloading them at runtime.
Install Git LFS (if not already installed):
git lfs install
Clone the models:

# Download Embedding Model
git lfs clone https://huggingface.co/BAAI/bge-small-en-v1.5

# Download Reranking Model
git lfs clone https://huggingface.co/BAAI/bge-reranker-base

Update config.py:
Change the model paths to point to your local directories:

# In config.py
EMBEDDING_MODEL = "./bge-small-en-v1.5"   # Path to local folder
RERANKER_MODEL = "./bge-reranker-base"    # Path to local folder

5. Final Configuration
Use config.py to change any other configuration values (such as CHUNK_SIZE, TOP_K_RESULTS, or KEYWORD_BOOST) to suit your hardware or specific manual requirements.

6. Run the App
streamlit run app.py
Open your browser to http://localhost:8501.