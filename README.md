# Airtel DTH RAG Chatbot

This is a cut-down version of the chatbot developed during my internship at **Airtel, Gurgaon**. This version runs through a **Command Line Interface (CLI)** instead of being deployed as a production application.

## RAG — Retrieval Augmented Generation

**Retrieval Augmented Generation (RAG)** is an approach that combines an organization's knowledge base with the capabilities of a Large Language Model (LLM). It can be used to automate repetitive information-based tasks such as customer support, where the model needs to respond based on a defined set of information.

In this project, a user enters a query, which is converted into a vector representation using **Sentence Transformers from Hugging Face**.

The knowledge base consists of information collected from various **Airtel DTH FAQ pages and documents**. This information is divided into smaller chunks, converted into vector embeddings, and stored in a **FAISS vector database**.

When a user submits a query, its vector representation is compared against the vectors stored in the knowledge base using **similarity search**. The most relevant chunks of information are retrieved and passed as context to the **Gemini 1.5 Flash** model.

Gemini then uses the retrieved information to generate a user-friendly response to the query.

**Redis** is used to store conversation context, allowing the chatbot to retain recent interactions with the user and respond to follow-up questions based on the ongoing conversation.

## Tech Stack

- **Python**
- **Google Gemini 1.5 Flash** — Response generation
- **Sentence Transformers** — Text embeddings
- **FAISS** — Vector database and similarity search
- **Redis** — Conversation context storage
- **PyMuPDF** — PDF text extraction

## Workflow

```text
User Query
    ↓
Sentence Transformer
    ↓
Query Embedding
    ↓
FAISS Similarity Search
    ↓
Relevant Knowledge Base Chunks
    ↓
Gemini 1.5 Flash
    ↓
User-Friendly Response
    ↓
Redis
    ↓
Conversation Context
```

## Knowledge Base

The knowledge base is built from Airtel DTH-related documents and FAQ data. The `build_kb.py` script extracts the content, divides it into chunks, generates embeddings, and creates the FAISS index.

The resulting knowledge base can then be loaded by the chatbot for retrieval.

## Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure the Gemini API key

Create a `.env` file in the project directory:

```env
GEMINI_API_KEY=your_api_key_here
```

### 3. Build the knowledge base

Make sure the required PDF and JSON files are present in their respective directories and run:

```bash
python build_kb.py
```

This generates the FAISS knowledge-base index.

### 4. Run the chatbot

```bash
python chatbot.py
```

The chatbot can then be used directly through the command line.

## Project Structure

```text
airtel-dth-rag-chatbot/
│
├── chatbot.py
├── build_kb.py
├── requirements.txt
├── .env.example
├── .gitignore
│
├── pdfs/
│   └── Airtel DTH knowledge-base documents
│
└── json/
    └── Airtel DTH FAQ data
```
