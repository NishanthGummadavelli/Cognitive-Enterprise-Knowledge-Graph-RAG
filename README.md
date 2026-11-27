# Cognitive Enterprise Knowledge Graph and RAG System

An AI-powered enterprise knowledge assistant that combines **Retrieval-Augmented Generation (RAG), Hybrid Information Retrieval, and Knowledge Graphs** to provide context-aware and document-grounded answers from enterprise documents.

> **Project Status: Work in Progress**  
> Core components and system modules have been developed. However, API integration, end-to-end workflows, and complete system-level verification are still in progress. Some functionality may require further testing, debugging, and refinement.

---

## 1. Project Overview

Organizations generate and maintain large volumes of documents containing valuable information. Retrieving relevant information from these documents can be challenging when traditional search methods rely primarily on keywords or when basic RAG systems retrieve information using semantic similarity alone.

The **Cognitive Enterprise Knowledge Graph and RAG System** addresses this challenge by combining document processing, semantic retrieval, keyword-based retrieval, and knowledge graph-based information retrieval.

The system processes uploaded documents, extracts and organizes their content, builds searchable indexes, and uses a Large Language Model (LLM) to generate responses based on retrieved information.

Our approach aims to improve information discovery by combining multiple retrieval techniques rather than depending on a single search method.

## 2. Problem Statement

Traditional document search systems often face the following limitations:

- Difficulty retrieving relevant information from large document collections.
- Keyword-based search may fail to identify contextually relevant information.
- Semantic search alone may overlook important exact terms and identifiers.
- Conventional RAG systems may not adequately represent relationships between entities across documents.
- Maintaining document context and organizing enterprise knowledge can become difficult as document collections grow.

Our project addresses these challenges through a hybrid retrieval architecture integrated with a knowledge graph.

## 3. Project Objectives

- Develop a document ingestion pipeline supporting multiple file formats.
- Extract, clean, classify, and organize document content.
- Implement structure-aware text chunking with metadata preservation.
- Generate semantic embeddings for document chunks.
- Combine semantic and keyword-based retrieval.
- Integrate knowledge graph-based entity and relationship retrieval.
- Use Reciprocal Rank Fusion (RRF) to combine retrieval results.
- Generate context-aware responses using a locally hosted LLM.
- Provide document management, authentication, and conversation management capabilities.

## 4. Proposed System Architecture

The system follows a modular architecture consisting of the following stages:

### 4.1 Document Ingestion and Preprocessing

Users upload documents in supported formats:

- PDF
- DOCX
- TXT

The ingestion pipeline performs:

1. **Document Loading:** Extracts document contents using format-specific loaders, including PyMuPDF for PDF processing.
2. **Text Cleaning:** Removes unnecessary formatting artifacts and prepares extracted text for processing.
3. **Document Classification:** Uses a machine-learning-based classification component to categorize documents.
4. **Text Chunking:** Divides document content into manageable sections while considering paragraph and heading boundaries.
5. **Metadata Preservation:** Maintains relevant information such as source documents, page numbers, section headings, and chunk relationships.

### 4.2 Embedding Generation and Indexing

After preprocessing, document chunks are converted into numerical vector representations using a pretrained BGE embedding model.

The resulting embeddings are stored in ChromaDB to support semantic similarity search.

The system also builds a keyword-based retrieval index using BM25, allowing relevant information to be retrieved through lexical matching.

This creates two complementary retrieval paths:

- **Dense Retrieval:** Identifies semantically related content using vector embeddings.
- **Sparse Retrieval:** Identifies relevant content using keyword matching through BM25.

### 4.3 Knowledge Graph Construction

The system incorporates a knowledge graph to represent entities and their relationships extracted from documents.

The extraction process uses:

- spaCy for initial named entity recognition.
- Qwen3 for contextual entity and relationship extraction and refinement.
- Neo4j AuraDB for knowledge graph storage.

Extracted entities and relationships are organized into a graph structure, enabling relationship-oriented information retrieval.

### 4.4 Hybrid Retrieval

When a user submits a question, the retrieval pipeline searches for relevant information through multiple approaches:

- Semantic search using BGE embeddings and ChromaDB.
- Keyword-based retrieval using BM25.
- Knowledge graph retrieval using Neo4j.

The retrieval results are combined using **Reciprocal Rank Fusion (RRF)**.

RRF combines ranked results from multiple retrieval methods to produce a consolidated ranking of relevant information.

### 4.5 Response Generation

The retrieved information is passed to the language model along with the user's question.

The system uses:

- **Ollama** for local language model execution.
- **Qwen3:4b** for contextual understanding and response generation.

The model generates responses using the retrieved context, helping ground the answers in the available document information.

---

## 5. Key Features and Developed Modules

| Module | Description |
|---|---|
| Multi-format document ingestion | Supports PDF, DOCX, and TXT document processing. |
| Document preprocessing | Performs text extraction, cleaning, and preparation. |
| Document classification | Integrates a machine-learning-based document classification component. |
| Structure-aware chunking | Divides content into manageable chunks while preserving document structure and metadata. |
| Semantic retrieval | Uses BGE embeddings and ChromaDB for similarity-based retrieval. |
| Keyword retrieval | Uses BM25 for lexical information retrieval. |
| Knowledge graph | Represents extracted entities and relationships using Neo4j. |
| Hybrid retrieval | Combines semantic, keyword, and graph-based retrieval. |
| Reciprocal Rank Fusion | Combines ranked retrieval results. |
| LLM integration | Uses Ollama with Qwen3:4b for response generation. |
| Document management | Supports document records, version management, and knowledge spaces. |
| Authentication | Implements registration, login, JWT authentication, refresh-token handling, and logout. |
| Conversation management | Supports conversation storage and management. |

### Supported RAG Modes

The system is designed to support three modes of information interaction:

1. **Document Question Answering:** Retrieves relevant document content to answer user questions.
2. **Document Summarization:** Generates summaries of uploaded document content.
3. **Knowledge-Augmented Question Answering:** Uses document retrieval and knowledge graph information to provide context-enriched responses.

---

## 6. Technology Stack

### Backend and API
- Python 3.12
- FastAPI
- Uvicorn
- SQLAlchemy
- Pydantic

### Frontend
- React
- TypeScript
- Vite

### Databases and Storage
- PostgreSQL – Relational data management.
- ChromaDB – Vector storage and semantic retrieval.
- Neo4j AuraDB – Knowledge graph storage.

### Artificial Intelligence and Information Retrieval
- Ollama
- Qwen3:4b
- BGE – Pretrained embedding model
- spaCy – Named entity recognition
- BM25 – Keyword-based retrieval
- Scikit-learn – Machine learning and document classification
- NumPy and Pandas – Data processing
- Joblib – Model artifact management

### Development and Integration
- Git and GitHub
- REST APIs
- JWT Authentication
- Reciprocal Rank Fusion (RRF)

---

## 7. Database Architecture

The project uses multiple storage technologies because different types of information require different storage and retrieval approaches.

| Technology | Purpose |
|---|---|
| PostgreSQL | Stores users, knowledge spaces, memberships, document records, document versions, refresh tokens, and conversations. |
| ChromaDB | Stores document chunk embeddings for semantic retrieval. |
| Neo4j AuraDB | Stores entities and relationships extracted from documents. |

This multi-database architecture separates relational application data, vector representations, and graph-based knowledge.

---

## 8. Development Approach

The project follows a modular development approach.

**Phase 1 – Requirement Analysis and Research**

- Studied existing RAG-based document question-answering systems.
- Identified limitations of relying only on semantic search.
- Explored hybrid retrieval and knowledge graph integration.
- Defined the proposed architecture and system modules.

**Phase 2 – Document Processing Pipeline**

- Implemented document loaders for supported file formats.
- Developed text cleaning and preprocessing components.
- Integrated document classification.
- Implemented structure-aware chunking and metadata preservation.

**Phase 3 – Retrieval and Knowledge Representation**

- Integrated BGE-based embedding generation.
- Implemented vector storage using ChromaDB.
- Developed BM25-based keyword retrieval.
- Integrated entity and relationship extraction.
- Implemented Neo4j knowledge graph storage.
- Developed hybrid retrieval and RRF-based result fusion.

**Phase 4 – Backend and Frontend Development**

- Developed backend services using FastAPI.
- Integrated PostgreSQL for application data.
- Implemented authentication and document management services.
- Developed the React and TypeScript frontend.
- Worked on frontend and backend API integration.

**Phase 5 – Integration and Testing**

- Worked on integrating the individual modules.
- Tested document ingestion and retrieval components.
- Investigated API integration and runtime issues.
- Continued debugging and validating the complete application workflow.

---

## 9. Project Directory Structure

The project is organized into separate frontend, backend, ingestion, retrieval, extraction, and storage modules.

```text
Cognitive-Enterprise-Knowledge-Graph-and-RAG/
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── ...
│   └── package.json
│
├── src/
│   ├── api/
│   │   └── routers/
│   │
│   ├── ingestion/
│   │   ├── loaders.py
│   │   ├── cleaner.py
│   │   └── chunker.py
│   │
│   ├── extraction/
│   │
│   ├── retrieval/
│   │
│   ├── knowledge_graph/
│   │
│   ├── storage/
│   │
│   └── ...
│
├── models/
│   └── classification/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── requirements.txt
├── README.md
└── .gitignore
```

*Note: This is a high-level representation of the project organization; individual files and folders may vary with the latest implementation.*

---

## 10. Running the Application

### Prerequisites

Ensure that the following are installed and configured:

- Python 3.12
- Node.js and npm
- PostgreSQL
- Neo4j AuraDB or a compatible Neo4j instance
- Ollama with the required Qwen3 model

### Backend Setup

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <PROJECT_FOLDER>
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install backend dependencies:

```bash
pip install -r requirements.txt
```

Configure the required database credentials, connection strings, model settings, and environment variables according to the project configuration.

Start the backend:

```bash
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend Setup

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

### Application URLs

| Service | URL |
|---|---|
| Frontend | http://127.0.0.1:3000 |
| Backend | http://127.0.0.1:8000 |
| API Documentation (Swagger) | http://127.0.0.1:8000/docs |
| Backend Health Check | http://127.0.0.1:8000/health |

*The frontend URL assumes the Vite development server is configured to use port 3000.*

---

## 11. Current Development Status

The project has progressed through the development of its major architectural components, including document processing, semantic and keyword retrieval, knowledge graph integration, backend services, and frontend development.

The application is currently undergoing integration and verification.

**Work currently in progress:**

- Verification of frontend and backend API communication.
- End-to-end validation of document upload and processing.
- Testing the complete hybrid retrieval workflow.
- Verification of knowledge graph retrieval and response generation.
- Debugging and resolving any remaining integration issues.
- Performance, reliability, and functional testing.

The project should therefore be considered a work in progress rather than a fully verified production-ready application.

---

## 12. Future Enhancements

Potential improvements include:

- Improved retrieval ranking and relevance evaluation.
- Advanced document classification and extraction.
- More robust entity resolution and knowledge graph maintenance.
- Better document citation and source traceability.
- Evaluation using retrieval and response-quality metrics.
- Performance optimization for large document collections.
- Improved API reliability and asynchronous document processing.
- Enhanced access control and enterprise-level security.

---

## 13. Team

**CodeCrafters**

This project is developed as a collaborative effort involving research, system design, AI integration, backend development, frontend development, database integration, and testing.

---

## 14. Conclusion

The Cognitive Enterprise Knowledge Graph and RAG System explores an integrated approach to enterprise knowledge retrieval by combining semantic search, keyword-based retrieval, knowledge graphs, and Large Language Models.

By bringing together multiple information retrieval techniques, the system aims to provide more context-aware and meaningful answers from enterprise documents.

The project demonstrates the application of modern AI technologies and modular software architecture to the development of an intelligent document knowledge assistant.

**Project Status: Still Work in Progress.**
