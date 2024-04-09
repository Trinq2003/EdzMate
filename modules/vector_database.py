from langchain.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from modules.embeddings import EmbeddingsModel
from langchain.docstore.document import Document as LangchainDocument
from typing import List

from modules.data_loader import KnowledgeBase

class KnowledgeVectorDatabase:
    def __init__(self, knowledge_base: KnowledgeBase, embedding_model: EmbeddingsModel) -> None:
        self.knowledge_base = knowledge_base
        self.knowledge_vector_database = FAISS.from_documents(
            self.knowledge_base.processed_knowledge_database, embedding_model, distance_strategy=DistanceStrategy.COSINE
        )

    def knowledge_vector_database(self):
        return self.knowledge_vector_database

    def get_knowledge_vector(self, query: str):
        return self.knowledge_base.processed_knowledge_database[query]
    
    def retrieve_knowledge(self, query: str, top_k: int) -> List[LangchainDocument]:
        top_k_relevant_docs = self.knowledge_vector_database.similarity_search(query, top_k)
        return top_k_relevant_docs
        
        
        
        