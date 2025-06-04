import os
from langchain_chroma import Chroma
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_core.documents.base import Document

def process_pdfs(path: str) -> list[Document]:
    dir_loader = DirectoryLoader(
        path=path,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader, #type: ignore
        loader_kwargs={"extract_tables": "html"},
        use_multithreading=True
    )
    text_splitter = CharacterTextSplitter(r"\n\n\d+\.\s*[^\n]+", True)

    docs = dir_loader.load_and_split(text_splitter)
    with open("output.txt", "w", encoding="utf-8") as output_file:
        for i, doc in enumerate(docs):
            file_path = doc.metadata["file_path"].lower()
            if "edital" in file_path:
                doc.metadata["tipo_documento"] = "edital"
            elif "chamada" in file_path:
                doc.metadata["tipo_documento"] = "chamada"
            else:
                doc.metadata["tipo_documento"] = "outros"
            print(f"Chunk: {i}", doc, sep='\n', end='\n\n\n', file=output_file)
    return docs

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
if os.path.exists("./db"):
    db = Chroma(embedding_function=embeddings, persist_directory="./db")
else:
    docs = process_pdfs("documentos fapes")
    db = Chroma.from_documents(docs, persist_directory="./db", embedding=embeddings)

query = """1.1. Ações Estratégicas 
a) estimular o intercâmbio de pesquisadores e estudantes de pós-graduação das instituições 
capixabas; 
b) estimular a formação de estudantes de pós-graduação; 
c) apoiar os Programas de Pós-graduação do Espírito Santo; 
d) despertar o interesse de estudantes e profissionais em P,D&I; 
e) estimular a atualização de conhecimentos e o debate de temas específicos e de interesse para o 
desenvolvimento científico e tecnológico do estado do Espírito Santo;   
f) contribuir para o desenvolvimento científico e tecnológico do estado do Espírito Santo. """
results = db.similarity_search("Ações estratégicas", 1, {"file_path": "documentos fapes/Edital Fapes 22.2024 - Estagio Técnico-Científico 2025-2.pdf"})
print(results[0].page_content)