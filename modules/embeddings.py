from langchain_community.embeddings import HuggingFaceEmbeddings

class EmbeddingsModel:
    def __init__(self, embedding_model_name) -> None:
        self.embedding_model = HuggingFaceEmbeddings(
            model_name = embedding_model_name,
            multi_process = True,
            model_kwargs = {"device": "cuda"},
            encode_kwargs = {"normalize_embeddings": True}
        )

if __name__ == "__main__":
    user_query = "How to create a pipeline in transformer library?"
    embeddings = EmbeddingsModel("bert-base-uncased")
    query_vector = embeddings.embedding_model.embed_query(user_query)
    
    print(query_vector)