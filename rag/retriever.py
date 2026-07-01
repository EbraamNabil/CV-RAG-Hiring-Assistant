from langchain_community.retrievers import BM25Retriever


class HybridRetriever:

    def __init__(self, vectordb, chunks):

        self.dense = vectordb.as_retriever(
            search_kwargs={"k": 5}
        )

        self.bm25 = BM25Retriever.from_documents(chunks)
        self.bm25.k = 5

    def invoke(self, query):

        dense_docs = self.dense.invoke(query)

        bm25_docs = self.bm25.invoke(query)

        merged = []

        seen = set()

        for doc in dense_docs + bm25_docs:

            key = (
                doc.metadata.get("source", "")
                + "_"
                + str(doc.metadata.get("page", 0))
                + "_"
                + doc.page_content[:100]
            )

            if key not in seen:

                seen.add(key)

                merged.append(doc)

        return merged[:5]


def create_retriever(vectordb, chunks):

    return HybridRetriever(
        vectordb,
        chunks
    )