from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from fastembed.rerank.cross_encoder import TextCrossEncoder
from src.config import CHROMA_PATH

class HybridRetriever:
    def __init__(self):
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings
        )
        #self.reranker = TextCrossEncoder(model_name="Xenova/bge-reranker-base")
        self.reranker = TextCrossEncoder(model_name="BAAI/bge-reranker-base")

    def get_relevant_knowledge(self, query: str, top_k: int = 5) -> list[dict]:
        # Busca semântica inicial no ChromaDB
        initial_docs = self.vector_store.similarity_search(query, k=15)
        
        if not initial_docs:
            return []

        # Reranking com FastEmbed para melhorar a precisão
        passages = [doc.page_content for doc in initial_docs]
        scores = list(self.reranker.rerank(query, passages))
        
        # Junta o score com o documento, ordena e pega o top_k
        ranked_results = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)[:top_k]
        
        # Extrai os metadados (IDs) para a matriz de rastreabilidade
        return [doc.metadata for _, doc in ranked_results]
