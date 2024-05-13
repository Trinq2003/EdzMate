from transformers import Pipeline

from modules.prompt import Prompt
from modules.data_loader import KnowledgeBase
from vector_database.vector_database import KnowledgeVectorDatabase
from modules.embeddings import EmbeddingsModel
from modules.reader import ReaderLLM
# from modules.reranker import Reranker

if __name__ == "__main__":
    TOP_K = 3
    reader = ReaderLLM(model_name="HuggingFaceH4/zephyr-7b-beta")
    raw_prompt_template = [
        {
            "role": "system",
            "content": """Using the information contained in the context,
    give a comprehensive answer to the question.
    Respond only to the question asked, response should be concise and relevant to the question.
    Provide the number of the source document when relevant.
    If the answer cannot be deduced from the context, do not give an answer.""",
        },
        {
            "role": "user",
            "content": """Context:
    {context}
    --- 
    Now here is the question you need to answer.

    Question: {question}""",
        },
    ]
    prompt_template = Prompt(prompt_template=raw_prompt_template, tokenizer=reader.tokenizer)
    
    
    knowledge_base = KnowledgeBase("./knowledge/huggingface_doc.csv")
    embeddings = EmbeddingsModel(embedding_model_name="thenlper/gte-small")
    knowledge_vector_database = KnowledgeVectorDatabase(knowledge_base= knowledge_base, embedding_model=embeddings
                                                        )
    # reranker = Reranker("colbert-ir/colbertv2.0")
    
    
    user_query = "How to create a pipeline?"
    print("[INFO] Retrieving documents...")
    retrieved_docs = knowledge_vector_database.retrieve_knowledge(query=user_query, top_k=TOP_K)
    print("[INFO] Reranking documents...")
    # reranked_docs = reranker.reranking(user_query=user_query, retrieved_docs=retrieved_docs, k=TOP_K)
    # reranked_docs = [doc["content"] for doc in reranked_docs]
    
    context = "\nContext document:\n"
    context += "".join(
        [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs)]
    )
    prompt = prompt_template.format(question=user_query, context=context)
    
    print("[INFO] Generating answer...")
    answer = reader(prompt)[0]["generated_text"]
    
    print(answer)