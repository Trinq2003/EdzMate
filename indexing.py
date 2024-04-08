import np
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from transformers import AutoTokenizer

# We use a hierarchical list of separators specifically tailored for splitting Markdown documents
# This list is taken from LangChain's MarkdownTextSplitter class.
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
TOKENIZER_MODEL_NAME = "thenlper/gte-small"

def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = TOKENIZER_MODEL_NAME,
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique

if __name__ == "__main__":
    from data_loader import load_huggingface_dataset
    from transformers import AutoTokenizer
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    
    raw_knowledge_base = load_huggingface_dataset("m-ric/huggingface_doc", split="train")
    
    docs_processed = split_documents(
        512,  # We choose a chunk size adapted to our model
        raw_knowledge_base,
        tokenizer_name=TOKENIZER_MODEL_NAME,
    )

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME)
    lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)]
    fig = pd.Series(lengths).hist()
    plt.title("Distribution of document lengths in the knowledge base (in count of tokens)")
    plt.show()