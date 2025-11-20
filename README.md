# Wayang GraphRAG Hybrid Chatbot

This project implements a **Hybrid Retrieval-Augmented Generation (RAG)** system to answer questions about Wayang stories (specifically *Sakri Lahir* and *Hastimurti Gugur*).

It combines two powerful retrieval methods:

1.  **Knowledge Graph (Structured):** Uses **Neo4j** to store entities (People, Locations, Weapons) and their relationships (e.g., `SON_OF`, `KILLED_BY`, `USES_WEAPON`). This allows for precise answers to genealogy and factual questions.
2.  **Semantic Vector Search (Unstructured):** Uses **Gemini Embeddings** to index the narrative text. This allows the bot to answer fuzzy or narrative-based questions (e.g., *"How did the battle happen?"*).

## 🏗️ Architecture

The system is containerized using Docker and consists of two main services:

* **Neo4j Database:** Stores the knowledge graph.
* **ETL & Chat Service (builder):** A Python container that handles:
    * **NER (Named Entity Recognition):** Uses `cahya/bert-base-indonesian-ner` to extract names.
    * **Relationship Extraction:** Uses **Google Gemini 1.5 Flash** to find connections between entities.
    * **Vector Embedding:** Uses `text-embedding-004` for semantic search.
    * **Chat Interface:** A terminal-based chatbot that queries both the Graph and Vector DB.

## 🚀 Prerequisites

* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
* [Git](https://git-scm.com/) (to clone the repository).
* **Google Gemini API Key** (Get one from [Google AI Studio](https://aistudio.google.com/)).

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/RaihanSavana/graphrag-ner-chatbot.git](https://github.com/RaihanSavana/graphrag-ner-chatbot.git)
cd graphrag-ner-chatbot
```
### Configure Secrets
Create a file named `.env` in the root directory (next to docker-compose.yml) and add your API keys:\\

```bash
# .env file
GEMINI_API_KEY=your_actual_google_api_key_here
NEO4J_PASSWORD=password
```

### 3. Build the Docker Containers

This will download the Neo4j image and install Python dependencies (Pandas, Torch, Neo4j Driver, etc.).

```bash
docker-compose build
```

## 🏃‍♂️ How to Run

### Step 1: Start the Database
Start the Neo4j container first and wait for it to initialize.
```bash
docker-compose up -d neo4j
```
**Note**: Wait about **20-30 minutes**. You can check if it's ready by running `docker logs neo4j_wayang_db` and looking for the "Started" message.

### Step 2: Launch the Chatbot
Run the interactive terminal chatbot.
```bash
docker-compose run --rm builder python terminal_chat.py
```
