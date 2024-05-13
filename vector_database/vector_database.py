from langchain.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.document import Document as LangchainDocument
from typing import List

from modules.embeddings import EmbeddingsModel
# from embeddings import EmbeddingsModel
from modules.data_loader import KnowledgeBase
# from data_loader import KnowledgeBase

class KnowledgeVectorDatabase:
    def __init__(self, knowledge_base: KnowledgeBase, embedding_model: EmbeddingsModel) -> None:
        self.knowledge_base = knowledge_base
        self.knowledge_base.split_documents(chunk_size=1024, tokenizer_name=embedding_model.get_model_name())
        print(f"[INFO] Type of knowledge_base.processed_knowledge_database: {type(self.knowledge_base.processed_knowledge_database[0])}")
        self.knowledge_vector_database = FAISS.from_documents(
            self.knowledge_base.processed_knowledge_database, embedding_model.embedding_model, distance_strategy=DistanceStrategy.COSINE
        )

    def knowledge_vector_database(self):
        return self.knowledge_vector_database

    def get_knowledge_vector(self, query: str):
        return self.knowledge_base.processed_knowledge_database[query]
    
    def retrieve_knowledge(self, query: str, top_k: int) -> List[LangchainDocument]:
        top_k_relevant_docs = self.knowledge_vector_database.similarity_search(query, top_k)
        return top_k_relevant_docs
        
if __name__ == "__main__":
    knowledge_base = KnowledgeBase("./knowledge/huggingface_doc.csv")
    embeddings = EmbeddingsModel(embedding_model_name="thenlper/gte-small")
    knowledge_vector_database = KnowledgeVectorDatabase(knowledge_base= knowledge_base, embedding_model=embeddings)
    
    user_query = "How to create a pipeline?"
    print("[INFO] Retrieving documents...")
    retrieved_docs = knowledge_vector_database.retrieve_knowledge(query=user_query, top_k=3)
    print(retrieved_docs)
    print("[INFO] Done")
