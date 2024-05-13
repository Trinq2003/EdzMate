from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from dotenv import load_dotenv
from transformers import AutoTokenizer
import os

# from modules.data_loader import MARKDOWN_SEPARATORS
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

class QdrantCloudServer(QdrantClient):
    def __init__(self, url, api_key):
        super().__init__(url=url, api_key=api_key)

if __name__ == "__main__":
    # Environemnt variables loading
    load_dotenv()
    QDRANT_HOST = os.getenv("QDRANT_HOST")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
    EMBEDDING_DIMENSION = os.getenv("EMBEDDING_DIMENSION")
    CHUNK_SIZE = os.getenv("CHUNK_SIZE")

    print(f"[INFO] QDRANT_HOST: {QDRANT_HOST}")
    print(f"[INFO] QDRANT_API_KEY: {QDRANT_API_KEY}")
    print(f"[INFO] QDRANT_COLLECTION_NAME: {QDRANT_COLLECTION_NAME}")
    print(f"[INFO] EMBEDDING_MODEL_NAME: {EMBEDDING_MODEL_NAME}")
    print(f"[INFO] EMBEDDING_DIMENSION: {EMBEDDING_DIMENSION}")
    print(f"[INFO] CHUNK_SIZE: {CHUNK_SIZE}")

    # Qdrant cloud server
    client = QdrantClient(
        url= QDRANT_HOST, 
        api_key= QDRANT_API_KEY,
    )

    # Load embedding model
    embedding_model_name = EMBEDDING_MODEL_NAME
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # Create collection
    vectors_config = VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE)
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=vectors_config
    )

    # Create vectorstore
    vectorstore = Qdrant(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embeddings=embeddings
    )

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME),
        chunk_size = int(CHUNK_SIZE),
        chunk_overlap = int(int(CHUNK_SIZE)/10),
        add_start_index = True,
        strip_whitespace = True,
        separators = MARKDOWN_SEPARATORS,
    )

    with open("./knowledge/md/dive_into_deep_learning.md", "r") as f:
        text = f.read()

    texts = text_splitter.split_text(text)

    vectorstore.add_texts(texts)

