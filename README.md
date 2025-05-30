# 🧠 Intelligent Virtual Assistant for Alzheimer's Disease (Powered by RAG + LLaMA 8B)

This repository contains a smart, research-backed **virtual assistant** designed to help users understand and explore topics related to **Alzheimer's Disease**. Built using **Retrieval-Augmented Generation (RAG)** and the **LLaMA 8B** language model, this system delivers high-quality responses derived from **real medical research articles** sourced from **PubMed** — one of the largest biomedical literature platforms.

---

## 📌 What This Project Does

This project creates an **AI assistant** that:
- Responds to questions about **Alzheimer’s Disease**.
- **Retrieves articles** from the **PubMed platform**, which is a publicly available medical research database managed by the U.S. National Library of Medicine.
- Uses a technique called **RAG (Retrieval-Augmented Generation)** — a powerful AI framework that **retrieves documents** first, then **generates answers** based on them.
- Stores the retrieved articles in **Chroma DB**, a vector database optimized for storing and searching text in the form of embeddings.
- Uses the **LLaMA 8B** model from Hugging Face — a **large language model with 8 billion parameters**, capable of understanding and generating human-like text.

---

## 📚 Key Technologies (Explained)

### 🧠 LLaMA 8B (Meta AI / Hugging Face)
A powerful transformer-based **Large Language Model (LLM)** that can answer complex questions. In this project, it is accessed via [Hugging Face](https://huggingface.co), a platform for hosting and running machine learning models.

> **8B** refers to the number of parameters — this is the internal knowledge scale of the model. More parameters = deeper understanding, but also more computing power required.

---

### 🔍 RAG – Retrieval-Augmented Generation

Instead of generating answers based purely on model training, RAG works in **two steps**:
1. **Retrieve**: First, it fetches relevant documents (articles from PubMed in this case).
2. **Generate**: Then, it feeds those documents into the language model to generate a response.

This allows the model to be **accurate, updated, and explainable**, especially when answering niche questions like those in medical domains.

---

### 💾 Chroma DB (Vector Database)

To allow fast searching through large text documents, the project uses **Chroma DB**, a specialized **vector store**. Each article is converted into a **numeric vector** (embedding), which allows for **semantic search** — finding documents based on meaning, not just keywords.

---

### 🛡️ Custom Guardrails

Custom **guardrails** have been implemented to:
- Filter inappropriate or medically dangerous responses
- Validate query intent before passing to the LLM
- Set boundaries on the topics the assistant will answer
- Prevent hallucinations and unverified content leakage

These safety checks ensure the assistant is **secure, domain-relevant, and trustworthy**.

---

### ✅ Testing: Frontend + Backend

The project includes complete **automated testing coverage**:

#### ✅ Backend Testing:
- Ensures that PubMed article scraping works correctly
- Verifies correct storage and retrieval from Chroma DB
- Validates interaction with LLaMA 8B for consistency

#### ✅ Frontend Testing:
- Checks user interface response accuracy
- Validates form inputs and query behavior
- Simulates real user conversations for UX improvement

These tests help ensure that the system behaves correctly, is stable, and can scale.

---

## ⚠️ Known Limitations

1. **Contextual Inaccuracy**  
   The system struggles to maintain deep understanding in longer or multi-turn conversations. This may be due to limitations in how much context LLaMA 8B can handle at once.

2. **Speed vs Quality Trade-off**  
   - Fetching a **large number of articles** improves answer quality but **slows the system dramatically** due to processing and embedding overhead.
   - Fetching **fewer articles** speeds things up but often results in **shallow, inaccurate answers**.

3. **Performance Bottlenecks**  
   LLaMA 8B is a **heavy model**, and without GPU acceleration or model quantization, it may take time to respond on standard systems.

---

## 🛠️ Installation & Usage Guide

### 🔧 Prerequisites

- Python 3.9+
- Hugging Face account & access token
- Chroma DB installed (`pip install chromadb`)
- Other Python libraries: `transformers`, `torch`, `requests`, `beautifulsoup4`, `langchain`, etc.

### 📦 Installation

```bash
git clone https://github.com/your-username/alzheimers-rag-assistant.git
cd alzheimers-rag-assistant
pip install -r requirements.txt
