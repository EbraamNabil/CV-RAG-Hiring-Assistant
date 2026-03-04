# CV-RAG Hiring Assistant

AI system for evaluating job candidates using **Retrieval-Augmented Generation (RAG)**.

The application analyzes multiple CVs, retrieves relevant evidence from them, and produces structured hiring recommendations using a large language model.

The goal of the system is to simulate the reasoning process of a **strict HR recruitment specialist**, ensuring that decisions are based only on explicit information found in the resumes.

---

# Project Overview

This project demonstrates how modern **RAG pipelines** can be used to build intelligent systems that reason over documents.

The system allows an HR user to upload multiple CVs and ask questions such as:

* Which candidate is best suited for an AI Engineer role?
* Compare candidates for a Machine Learning position.
* Which candidate has the strongest experience with RAG systems?

Instead of relying on keyword matching alone, the system retrieves relevant context from the CVs and sends that information to an LLM to produce a reasoned answer supported by evidence.

---

# Core Concepts Implemented

The project demonstrates several advanced RAG techniques commonly used in production AI systems.

These include:

* Multiple retrieval strategies
* Hybrid search
* Adaptive retrieval
* Document-aware chunking
* Prompt injection protection
* Evidence-based reasoning

---

# Retrieval Strategies Implemented

The system supports **four different RAG retrieval approaches**.
The user can choose between them from the Streamlit sidebar.

## 1. Multi Query Retrieval

Multi Query retrieval improves recall by generating several alternative search queries.

Instead of retrieving documents using the original question only, the system asks the language model to rewrite the query into multiple variations.

Example:

Original question:

```
Which candidate is best for an AI engineer role?
```

Generated queries might be:

```
AI engineer experience
machine learning engineer skills
deep learning engineer background
```

Each query retrieves relevant chunks, and all results are combined before sending them to the model.

This approach helps retrieve documents that might not match the exact wording of the original question.

---

## 2. Hybrid RAG (Semantic + Keyword Retrieval)

Hybrid retrieval combines two different search techniques.

### Semantic Search

Semantic search uses **vector embeddings** to capture the meaning of text.

The system converts each document chunk into a numerical vector using a transformer model:

```
sentence-transformers/all-MiniLM-L6-v2
```

The vectors are stored in **ChromaDB**, which allows similarity search.

When a query is asked, the system finds the chunks with the most similar semantic meaning.

---

### Keyword Search

Keyword search is implemented using **BM25 retrieval**.

BM25 is a classical information retrieval algorithm used in search engines.

It works by matching exact words and ranking documents based on term frequency and document length.

---

### Why Hybrid Retrieval?

Semantic search is good at understanding meaning but sometimes misses exact keywords.

Keyword search captures exact terms but cannot understand context.

By combining both:

```
semantic results + keyword results
```

the system retrieves a more complete set of relevant documents.

---

## 3. Adaptive RAG (Relative Threshold)

Adaptive retrieval dynamically decides how many chunks to use.

The system retrieves many candidate chunks and calculates their similarity scores.

Then it observes how the scores change.

If the similarity score suddenly drops significantly, the system assumes that the remaining chunks are no longer relevant.

Retrieval stops automatically at that point.

This prevents irrelevant information from being sent to the LLM.

---

## 4. Adaptive RAG (Biggest Jump)

This strategy also analyzes similarity scores.

Instead of using an average threshold, it identifies the **largest gap between consecutive similarity scores**.

Example score distribution:

```
0.92
0.90
0.88
0.87
0.40
0.39
```

There is a large gap between **0.87 and 0.40**.

The system keeps only the chunks above that gap.

This helps isolate the most relevant pieces of information.

---

# Document Chunking Strategies

Before retrieval can happen, documents must be divided into smaller pieces called **chunks**.

The project implements **two chunking strategies**.

Users can choose between them in the Streamlit interface.

---

## 1. Recursive Character Chunking

This method splits text based on a hierarchy of separators.

The splitting order typically follows:

```
paragraphs
lines
spaces
```

Chunk parameters used in this project:

```
chunk_size = 1000 characters
chunk_overlap = 150 characters
```

Overlap ensures that important context is not lost between chunks.

This method is fast and works well for most documents.

---

## 2. Document-Aware Structural Chunking

This strategy uses the **Unstructured library** to parse the PDF.

Instead of splitting blindly by characters, the system identifies document elements such as:

* Titles
* Headers
* Paragraphs
* List items
* Sections

Each structural element becomes a chunk.

Example elements detected in CVs:

```
Experience
Projects
Education
Skills
Certifications
```

This preserves the logical structure of the document and improves retrieval quality.

---

# System Architecture

```
User Question
        ↓
Query Processing
        ↓
Retrieval Layer
    ├─ Semantic Search (ChromaDB + Embeddings)
    └─ Keyword Search (BM25)
        ↓
Hybrid Retrieval
        ↓
Context Construction
        ↓
Large Language Model
(Llama 3.3 via Groq API)
        ↓
Structured HR Evaluation
        ↓
Evidence Display
```

---

# Streamlit Interface

The system includes an interactive interface built with **Streamlit**.

The interface is divided into two main sections.

---

## Sidebar

The sidebar allows the user to configure the system.

The user can:

* Upload exactly **five CV files**
* Select the **retrieval strategy**
* Select the **chunking strategy**

Available retrieval options:

```
Multi Query
Hybrid RAG
Adaptive RAG (Relative Threshold)
Adaptive RAG (Biggest Jump)
```

Available chunking options:

```
Recursive Chunking
Document Aware Chunking
```

---

## Chat Interface

The main interface behaves like a chat assistant.

The HR user can ask questions such as:

```
Which candidate is the best AI engineer?
Compare the candidates for a data scientist role.
Which candidate has RAG experience?
```

The system responds with three sections.

---

### Quick Answer

A short summary of the hiring recommendation.

---

### Detailed Analysis

The system explains its reasoning using only retrieved CV information.

If the user asks for comparison, the system generates a structured table.

Example:

```
| Candidate | Strengths | Missing Skills | Verdict |
```

---

### Final Conclusion

A final hiring recommendation based on the analysis.

---

### Evidence Section

Below the answer, the system shows the **actual CV text used as evidence**.

Each piece of evidence is grouped by candidate and displayed inside expandable sections.

This ensures transparency and allows the HR user to verify the reasoning.

---

# Security

The system includes **prompt injection detection**.

Certain malicious instructions are blocked, such as:

* ignore previous instructions
* bypass rules
* output unrelated content

If such patterns are detected, the assistant refuses the request.

---

# Technology Stack

Python
Streamlit
LangChain
ChromaDB
Sentence Transformers
BM25 Retrieval
Groq API
Llama 3.3

---

# Installation

Clone the repository:

```
git clone https://github.com/EbraamNabil/CV-RAG-Hiring-Assistant.git
```

Enter the project directory:

```
cd CV-RAG-Hiring-Assistant
```

Install dependencies:

```
pip install -r requirements.txt
```

Create environment variables:

```
GROQ_API_KEY=your_api_key_here
```

Run the application:

```
streamlit run app2.py
```

---

# Future Improvements

Possible extensions for the system include:

* Cross-encoder reranking
* Candidate scoring
* automatic shortlist generation
* support for larger document collections
* ATS integration
