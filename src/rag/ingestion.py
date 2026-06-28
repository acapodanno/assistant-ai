from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

LOADER_MAP = {
    ".md":   lambda path: TextLoader(file_path=path, encoding="utf-8"),
    ".txt":  lambda path: TextLoader(file_path=path, encoding="utf-8"),
    ".pdf":  lambda path: PyPDFLoader(file_path=path),
}

def retrieve_documents() -> list[Document]:
    kb_root = "./data/knowledge_base"
    documents = []
    for root, _dirs, files in os.walk(kb_root):
        for filename in files:
            filepath = os.path.join(root, filename)
            ext = os.path.splitext(filename)[1].lower()
            loader_factory = LOADER_MAP.get(ext)
            if loader_factory is None:
                continue
            loader = loader_factory(filepath)
            docs = loader.load()
            category = os.path.basename(root)
            for doc in docs:
                doc.metadata["file_type"] = ext.lstrip(".")
                doc.metadata["source"] = filepath
                doc.metadata["category"] = category
            documents.extend(docs)
    return documents

SEPARATORS_MAP = {
    "md":   ["\n# ", "\n## ", "\n### ", "\n\n", "\n", " ", ""],
    "pdf":  ["\n\n", "\n", ". ", " ", ""],
    "txt":  ["\n\n", "\n", " ", ""],
}

def chunk_documents(documents: list[Document]) -> list[Document]:
    chunked = []
    for doc in documents:
        file_type = doc.metadata.get("file_type", "txt")
        separators = SEPARATORS_MAP.get(file_type, ["\n\n", "\n", " ", ""])
        splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=800,
            chunk_overlap=120,
            length_function=len,
        )
        chunked.extend(splitter.split_documents([doc]))
    return chunked
