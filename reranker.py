from ragatouille import RAGPretrainedModel
from typing import List
from langchain.docstore.document import Document as LangchainDocument

class Reranker:
    def __init__(self, name_of_model):
        self.reranker = RAGPretrainedModel.from_pretrained(name_of_model)
        
    def reranking(self, user_query:str, retrieved_docs: List[LangchainDocument], k:int) -> List[LangchainDocument]:
        reranked_docs = self.reranker.rerank(user_query, retrieved_docs, k=k)
        return reranked_docs