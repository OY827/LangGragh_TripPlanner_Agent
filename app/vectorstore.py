import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

def get_qdrant_client():
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", 6333))

    return QdrantClient(host=host, port=port)

def get_vectorstore(collection_name: str = "langchain_collection"):
    embeddings = OpenAIEmbeddings()

    qdrant_client = get_qdrant_client()

    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embeddings=embeddings
    )
