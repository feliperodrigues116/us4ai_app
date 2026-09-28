import streamlit as st
import pandas as pd
import sys
import os
import pysqlite3

sys.modules["sqlite3"] = pysqlite3

# Adiciona o diretório base no path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.graph.workflow import build_us4ai_graph
from src.ingestion import prepare_nist_documents

# --- LLM Guard Configuration ---
try:
    from llm_guard.input_scanners import PromptInjection
    from llm_guard.input_scanners.prompt_injection import MatchType
    # Inicializa o scanner. Usa validação rigorosa (FULL) e um threshold de 0.5.
    scanner = PromptInjection(threshold=0.5, match_type=MatchType.FULL)
    scanner_available = True
except ImportError:
    scanner_available = False
    st.warning("LLM Guard não está instalado. A validação de Prompt Injection está desabilitada.")

st.set_page_config(page_title="us4ai - RE4AI Assistant", page_icon="🤖", layout="wide")

st.title("🤖 us4ai - Engenharia de Requisitos para IA")

# Prepare NIST documents without creating a vector index.
with st.sidebar:
    st.header("NIST Playbook Knowledge")
    if st.button("Prepare NIST documents", type="primary"):
        with st.spinner("Validating the authoritative NIST Playbook..."):
            documents = prepare_nist_documents()
            st.success(f"Prepared {len(documents)} NIST documents. No vector index was written.")
    st.info("Source: catalogs/nist_ai_rmf_playbook.json. NIST vector indexing is not implemented yet.")

# Main: Input do Usuário
st.subheader("📝 Formulário de User Story")
user_story = st.text_area("Descrição da User Story:", "As a customer, I want to issue my water bill via AI chatbot...", height=100)
criteria = st.text_area("Critérios de Aceite (um por linha):", "The chatbot shall verify customer identity before issuing details.", height=100)

if st.button("🚀 Processar Análise de Riscos e Requisitos"):
    # 1. Aplicando o LLM Guard para evitar Prompt Injection na User Story
    is_valid = True
    if scanner_available:
        sanitized_prompt, is_valid, risk_score = scanner.scan(user_story)
        if not is_valid:
            st.error(f"⚠️ Alerta de Segurança: Risco de Prompt Injection Detectado! (Score: {risk_score:.2f}). A execução foi bloqueada.")

    if is_valid:
        if scanner_available:
            st.success("✅ Validação de segurança aprovada.")
            
        with st.spinner("Analisando grafos de decisão e derivando conhecimento..."):
            app_graph = build_us4ai_graph()
            
            initial_state = {
                "story_id": "US-001",
                "description": user_story,
                "acceptance_criteria": [c.strip() for c in criteria.split("\n") if c.strip()],
                "retrieved_docs": [],
                "identified_risks": [],
                "derived_requirements": []
            }
            
            # Invoca o LangGraph
            result = app_graph.invoke(initial_state)

            st.success("Análise RAG concluída com sucesso!")

            # Exibição de Resultados em Abas
            tab1, tab2, tab3 = st.tabs(["📜 Requisitos Gerados (AIR)", "⚠️ Matriz de Riscos", "🔍 Fontes Recuperadas (RAG)"])

            with tab1:
                if result.get("derived_requirements"):
                    df_reqs = pd.DataFrame(result["derived_requirements"])
                    st.dataframe(df_reqs, use_container_width=True)
                else:
                    st.info("Nenhum requisito específico derivado. O LLM pode ter falhado ou os riscos eram insuficientes.")

            with tab2:
                if result.get("identified_risks"):
                    st.json(result["identified_risks"])
                else:
                    st.info("Nenhum risco de IA associado a este contexto.")

            with tab3:
                if result.get("retrieved_docs"):
                    st.json(result["retrieved_docs"])
                else:
                    st.info("Nenhum documento de catálogo encontrado. Verifique se o ChromaDB está povoado.")

