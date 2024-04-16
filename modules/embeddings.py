from langchain_community.embeddings import HuggingFaceEmbeddings

class EmbeddingsModel:
    def __init__(self, embedding_model_name) -> None:
        self.model_name = embedding_model_name
        self.embedding_model = HuggingFaceEmbeddings(
            model_name = self.model_name,
            multi_process = True,
            model_kwargs = {"device": "cuda"},
            encode_kwargs = {"normalize_embeddings": True}
        )
    def get_model_name(self):
        return self.model_name

if __name__ == "__main__":
    EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    
    user_query = "How to create a pipeline in transformer library?"
    embeddings = EmbeddingsModel(EMBEDDING_MODEL_NAME)
    query_vector = embeddings.embedding_model.embed_query(user_query)
    
    print(query_vector)