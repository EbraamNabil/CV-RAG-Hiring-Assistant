import streamlit as st

from langchain_chroma import Chroma

from rag.embedding import get_embedding_model


@st.cache_resource
def create_vectordb(chunks):

    embeddings = get_embedding_model()

    vectordb = Chroma.from_documents(

        documents=chunks,

        embedding=embeddings

    )

    return vectordb