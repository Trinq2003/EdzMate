import datasets
import tqdm
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from typing import Optional

class KnowledgeBase:
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
    def __init__(self, path_to_dataset: str, split: str = "train"):
        self.ds = datasets.load_dataset(path_to_dataset, split=split)
        self.load_huggingface_dataset(path_to_dataset=path_to_dataset, split=split)
        
        self.processed_knowledge_database = []
        
    def load_huggingface_dataset(self, path_to_dataset: str, split: str = "train"):
        ds = datasets.load_dataset(path_to_dataset, split=split)
        raw_knowledge_base = [
            LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(ds)
        ]
        
    def split_documents(self, chunk_size:int, tokenizer_name:Optional[str]):
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(tokenizer_name),
            chunk_size = chunk_size,
            chunk_overlap = int(chunk_size/10),
            add_start_index = True,
            strip_whitespace = True,
            seperators = self.MARKDOWN_SEPARATORS,
        )
        
        processing_knowledge_database = []
        for doc in self.raw_knowledge_base:
            processing_knowledge_database += text_splitter.split_documents([doc])
            
        unique_texts = {}
        for doc in processing_knowledge_database:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                self.processed_knowledge_database.append(doc)
        



# if __name__ == "__main__":
