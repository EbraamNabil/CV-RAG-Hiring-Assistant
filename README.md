# 📄 RAG HR Candidate Evaluator

An AI-powered HR assistant that evaluates job candidates by analyzing multiple CVs using **Retrieval-Augmented Generation (RAG)**.

The system retrieves relevant information from candidate resumes and reasons like a **senior HR recruiter** to recommend the best candidates for a role.

---

# 🚀 Features

- Multi-CV analysis
- Semantic search using embeddings
- Keyword search using BM25
- Hybrid RAG (Semantic + Keyword retrieval)
- Multi-Query retrieval
- Adaptive RAG retrieval strategies
- Prompt Injection protection
- Evidence-based candidate evaluation
- Interactive Streamlit interface

---

# 🧠 RAG Techniques Implemented

The project demonstrates multiple advanced RAG techniques:

| Technique | Description |
|--------|--------|
| Multi Query | Generates multiple search queries for better retrieval |
| Hybrid RAG | Combines semantic search and keyword search |
| Adaptive RAG (Relative Threshold) | Dynamically selects chunks based on score drops |
| Adaptive RAG (Biggest Jump) | Detects the largest similarity score gap |
| Structural Chunking | Uses document structure to build chunks |

# 🏗 System Architecture
User Question
↓
Query Processing
↓
Retrieval Layer

• Semantic Search (Embeddings + Chroma)
• Keyword Search (BM25)

↓

Hybrid Retrieval

↓

Context Construction

↓

LLM Reasoning (Llama 3.3 via Groq)

↓

HR Decision + Evidence

# 🏗 System Architecture
cv-rag-project

app.py
requirements.txt
.gitignore
.env.example

temp/
chroma_db/


---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/cv-rag-project.git
cd cv-rag-project

install dependencies:
pip install -r requirements.txt

Create .env file:
GROQ_API_KEY=your_api_key_here

Run the app:
streamlit run app.py





---

# 🏗 System Architecture
