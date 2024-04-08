from langchain_community.embeddings import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "thenlper/gte-small"

def embedding_model(embedding_model_name:str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
    )

if __name__ == "__main__":
    import pacmap
    import numpy as np
    import pandas as pd
    import plotly.express as px
    from indexing import split_documents, TOKENIZER_MODEL_NAME
    from vector_database import knowledge_vector_database
    from data_loader import load_huggingface_dataset
    
    embedding_model = embedding_model(EMBEDDING_MODEL_NAME)
    user_query = "How to create a pipeline object?"
    query_vector = embedding_model.embed_query(user_query)
    
    raw_knowledge_base = load_huggingface_dataset("m-ric/huggingface_doc", split="train")
    
    docs_processed = split_documents(
        512,  # We choose a chunk size adapted to our model
        raw_knowledge_base,
        tokenizer_name=TOKENIZER_MODEL_NAME,
    )
    
    knowledge_vector_database = knowledge_vector_database(docs_processed, embedding_model)

    embedding_projector = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=1)

    embeddings_2d = [
        list(knowledge_vector_database.index.reconstruct_n(idx, 1)[0]) for idx in range(len(docs_processed))
    ] + [query_vector]
    
    documents_projected = embedding_projector.fit_transform(np.array(embeddings_2d), init="pca")
        
    df = pd.DataFrame.from_dict(
        [
            {
                "x": documents_projected[i, 0],
                "y": documents_projected[i, 1],
                "source": docs_processed[i].metadata["source"].split("/")[1],
                "extract": docs_processed[i].page_content[:100] + "...",
                "symbol": "circle",
                "size_col": 4,
            }
            for i in range(len(docs_processed))
        ]
        + [
            {
                "x": documents_projected[-1, 0],
                "y": documents_projected[-1, 1],
                "source": "User query",
                "extract": user_query,
                "size_col": 100,
                "symbol": "star",
            }
        ]
    )

    # visualize the embedding
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="source",
        hover_data="extract",
        size="size_col",
        symbol="symbol",
        color_discrete_map={"User query": "black"},
        width=1000,
        height=700,
    )
    fig.update_traces(
        marker=dict(opacity=1, line=dict(width=0, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )
    fig.update_layout(
        legend_title_text="<b>Chunk source</b>",
        title="<b>2D Projection of Chunk Embeddings via PaCMAP</b>",
    )
    fig.show()