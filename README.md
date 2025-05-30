# 🧠 Intelligent Virtual Assistant for Alzheimer's Disease (RAG-based)

An AI-powered virtual assistant designed to provide medical insights for Alzheimer's Disease using **Retrieval-Augmented Generation (RAG)**. This system fetches medical articles from **PubMed**, stores them in a vector database, and generates responses using a powerful LLM — **LLaMA 8B (via Hugging Face)**.

---

## 🚀 Features

- 🔍 **Dynamic Article Retrieval** from [PubMed](https://pubmed.ncbi.nlm.nih.gov/)
- 🧠 **RAG Architecture**: combines retrieval + generation for contextual responses
- 🧬 Uses **Chroma DB** as the vector store
- 🤖 Integrated with **LLaMA 8B** model (Hugging Face) for advanced natural language responses
- 🩺 Tailored to Alzheimer’s-related research and user queries
- ⚙️ CLI or API-based interaction with the assistant

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Language Model | [`LLaMA 8B`](https://huggingface.co/) (Hugging Face) |
| Vector DB | [`Chroma`](https://www.trychroma.com/) |
| Retrieval | Custom PubMed scraper & parser |
| Architecture | Retrieval-Augmented Generation (RAG) |
| Backend | Python, LangChain (if applicable) |

---

## ⚠️ Current Limitations

1. **Contextual Learning Issues**  
   → The system struggles to maintain deep context across multiple queries, especially in multi-turn interactions.

2. **Performance Trade-Off**  
   - Increasing the number of PubMed articles improves context, but **drastically slows down** the system.  
   - Reducing articles improves performance but **weakens answer quality**.

3. **Memory & Latency**  
   - Loading large datasets into memory for LLaMA 8B and embeddings can significantly increase latency.

---

## 📂 Project Structure

```bash
├── pubmed_scraper/
│   └── fetch_articles.py       # Fetches and parses articles from PubMed
├── embedding/
│   └── vector_store.py         # Chroma DB setup and ingestion
├── llm/
│   └── llama_interface.py      # Hugging Face integration with LLaMA 8B
├── app.py                      # Main app logic for query processing
├── requirements.txt
└── README.md
