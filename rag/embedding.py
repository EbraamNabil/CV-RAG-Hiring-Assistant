import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "BAAI/bge-small-en-v1.5"


@st.cache_resource

#rhe difference between @st.cache_data and @st.cache_resource is that @st.cache_data is used to cache data that is expensive to compute or load, while @st.cache_resource is used to cache resources that are expensive to create or initialize. In this case, we are caching the embedding model, which is a resource that is expensive to create, so we use @st.cache_resource.
def get_embedding_model():

    return HuggingFaceEmbeddings(

        model_name=MODEL_NAME,

        model_kwargs={
            "device": "cpu"
        },

        encode_kwargs={
            "normalize_embeddings": True
        }

    )