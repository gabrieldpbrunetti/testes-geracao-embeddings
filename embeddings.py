import os
from langgraph.graph import StateGraph
from typing import TypedDict
from langchain_chroma import Chroma
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_core.documents.base import Document

def process_docs(path: str) -> list[Document]:
    dir_loader = DirectoryLoader(
        path=path,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader, #type: ignore
        # loader_kwargs={"extract_tables": "html"},
        use_multithreading=True
    )
    text_splitter = CharacterTextSplitter(r"\n\n\d+\.\s*[^\n]+", True, chunk_size=500, chunk_overlap=100)

    docs = dir_loader.load_and_split(text_splitter)
    for doc in docs:
        file_path = doc.metadata["file_path"].lower()
        if "edital" in file_path:
            doc.metadata["tipo_documento"] = "edital"
        elif "chamada" in file_path:
            doc.metadata["tipo_documento"] = "chamada"
        else:
            doc.metadata["tipo_documento"] = "outros"   
    return docs

path = "documentos fapes"
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="semantic_similarity")

if os.path.exists("db"):
    db = Chroma(embedding_function=embeddings, persist_directory="db")
else:
    docs = process_docs(path)
    db = Chroma.from_documents(docs, embeddings, persist_directory="db")

results = db.similarity_search("Qual o prazo de submissão de uma proposta", 3, {"file_path": "documentos fapes/Edital Fapes 21.2024 - Visita Técnico-Científica 2025-1.pdf"})
print(results)