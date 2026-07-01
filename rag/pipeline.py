from rag.loader import load_pdf
from rag.splitter import split_documents
from rag.vectordb import create_vectordb
from rag.retriever import create_retriever


def build_rag(pdf_paths):

    documents = []

    for pdf in pdf_paths:

        docs = load_pdf(pdf)

        documents.extend(docs)

    chunks = split_documents(documents)

    vectordb = create_vectordb(chunks)

    retriever = create_retriever(
        vectordb,
        chunks
    )

    return retriever