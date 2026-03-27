from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def get_db():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Chroma(
        collection_name="aws_docs",
        embedding_function=embeddings,
        persist_directory="./chroma",
        collection_metadata={
            "hnsw:space": "cosine"
        }
    )
    return vector_store
