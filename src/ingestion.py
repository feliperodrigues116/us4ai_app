import json
import os
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from src.config import CHROMA_PATH

CATALOGS_DIR = "./catalogs"

def load_json_catalog(filepath: str) -> list[Document]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    docs = []
    for item in data:
        # Concatenar problema e controle para facilitar a busca híbrida
        content = f"Problem: {item.get('problem', '')}\nControl: {item.get('control', '')}"
        metadata = {
            "knowledge_item_id": item.get("id", ""),
            "source_name": item.get("source_name", ""),
            "source_section": item.get("source_section", ""),
            "category": item.get("category", "")
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

def check_and_ingest_catalogs():
    """Lê os JSONs da pasta catalogs/ e ingere no ChromaDB local se existirem e não estiverem indexados."""
    if not os.path.exists(CATALOGS_DIR):
        os.makedirs(CATALOGS_DIR)
        return False, "Pasta catalogs/ vazia ou não encontrada. Coloque os arquivos JSON lá."

    files = [f for f in os.listdir(CATALOGS_DIR) if f.endswith('.json')]
    if not files:
         return False, "Nenhum JSON encontrado na pasta catalogs/. A busca não funcionará."

    all_docs = []
    for f in files:
        filepath = os.path.join(CATALOGS_DIR, f)
        docs = load_json_catalog(filepath)
        all_docs.extend(docs)

    if all_docs:
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        
        # Simples check para não duplicar dados
        current_docs = vectorstore._collection.count()
        if current_docs > 0:
            return True, f"Base já indexada. ({current_docs} itens encontrados)."

        # Adiciona no banco e persiste
        vectorstore.add_documents(documents=all_docs)
        return True, f"Sucesso! {len(all_docs)} itens indexados no ChromaDB."

    return False, "Falha na leitura dos documentos JSON."
