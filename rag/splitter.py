from langchain_text_splitters import RecursiveCharacterTextSplitter

# we make create_splitter  function to make Loose Coupling which means that the function is independent of other parts of the code and can be easily modified or replaced without affecting the rest of the codebase. This makes the code more flexible and easier to maintain.

def create_splitter():
    
    

    return RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=200,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )


def split_documents(documents):

    splitter = create_splitter()

    return splitter.split_documents(
        documents
    )