import datasets
from tqdm import tqdm
import os
import pandas as pd
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from typing import Optional

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

class KnowledgeBase:
    def __init__(self, path_to_dataset: str, split: str = "train"):
        if os.path.exists(path_to_dataset):
            self.ds = pd.read_csv(path_to_dataset)
            
            print("[INFO] Local dataset loaded successfully")
        else:
            self.ds = datasets.load_dataset(path_to_dataset, split=split)
        
        self.raw_knowledge_base = [
            LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) 
            for doc in tqdm(self.ds.to_dict(orient="records"))
        ]
        
        self.processed_knowledge_database = []
        
        
    def split_documents(self, chunk_size:int, tokenizer_name:Optional[str]):
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(tokenizer_name),
            chunk_size = chunk_size,
            chunk_overlap = int(chunk_size/10),
            add_start_index = True,
            strip_whitespace = True,
            separators = self.MARKDOWN_SEPARATORS,
        )
        
        processing_knowledge_database = []
        for doc in self.raw_knowledge_base:
            processing_knowledge_database += text_splitter.split_documents([doc])
            
        unique_texts = {}
        for doc in processing_knowledge_database:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                self.processed_knowledge_database.append(doc)
        



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    
    
    knowledge_base = KnowledgeBase("./knowledge/huggingface_doc.csv")
    knowledge_base.split_documents(chunk_size=1000, tokenizer_name=EMBEDDING_MODEL_NAME)
    
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(knowledge_base.processed_knowledge_database)]
    
    fig = pd.Series(lengths).hist()
    plt.title("Distribution of document lengths in the knowledge base (in count of tokens)")
    plt.show()
    
    
    