# 🇮🇳 Optimized Legal RAG System (Phi-2)

This project implements a highly efficient Retrieval-Augmented Generation (RAG) system designed to answer complex legal queries based on external document context (specifically, Indian law documents like the Contract Act and Company Law).

The system is engineered for maximum performance and stability on resource-constrained hardware, such as the standard **NVIDIA T4 GPU** available in Colab environments.

## 🌟 Key Features

* **RAG Architecture:** Uses the LangChain framework to retrieve relevant snippets from a local vector store before generating an answer.

* **Cited Answers:** The LLM is instructed to generate citations (e.g., `[Snippet 1]`) ensuring all facts are traceable back to the source documents.

* **Memory Optimization:** Uses the smallest viable model (Microsoft Phi-2) and offloads the BGE embedding model to the CPU to prevent CUDA Out of Memory (OOM) errors.

## 💻 Technology Stack

| Component | Technology | File Reference | Purpose | 
 | ----- | ----- | ----- | ----- | 
| **Large Language Model** | **Microsoft Phi-2** (2.7B) | `app.py` | Highly optimized, efficient model for fast, reliable inference on a T4 GPU. | 
| **Embeddings** | **BAAI/bge-m3** | `ingest.py` | State-of-the-art embedding model for high-quality semantic search. **(Critical: Loaded onto CPU)** | 
| **Vector Database** | **FAISS** | `ingest.py` | Local, fast, and CPU-efficient similarity search index. | 
| **Framework** | **LangChain** | `app.py`, `ingest.py` | Orchestrates the RAG pipeline. | 
| **Interface** | **Gradio** | `app.py` | Creates the shareable, user-friendly web interface. | 

## 🛠️ Setup and Installation

### Prerequisites

1. Access to a cloud environment (e.g., Google Colab) with a **GPU Runtime (T4 recommended)**.

2. Your legal PDF documents inside a directory named `data/` in the project root.

### Installation Steps

1. **Clone the Repository** and navigate to the project directory.

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
