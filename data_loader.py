import datasets
import tqdm
from langchain.docstore.document import Document as LangchainDocument

def load_huggingface_dataset(name_of_dataset: str, split: str = "train"):
    ds = datasets.load_dataset(name_of_dataset, split=split)
    RAW_KNOWLEDGE_BASE = [
        LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(ds)
    ]
    
    return RAW_KNOWLEDGE_BASE

if __name__ == "__main__":
    RAW_KNOWLEDGE_BASE = load_huggingface_dataset("wikipedia", split="train")
    print(RAW_KNOWLEDGE_BASE[0].page_content)
    print(RAW_KNOWLEDGE_BASE[0].metadata)