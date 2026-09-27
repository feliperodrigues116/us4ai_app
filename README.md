# US4AI

**US4AI** é uma ferramenta de apoio à Engenharia de Requisitos para sistemas baseados em Inteligência Artificial.

O MVP utiliza uma abordagem baseada em **RAG (Retrieval-Augmented Generation)** para recuperar conhecimento de catálogos especializados e apoiar a identificação de riscos relacionados à IA e a derivação de requisitos.

## Arquitetura do MVP

Fluxo principal:

User Story  
→ Detecção de relação com IA  
→ Recuperação de conhecimento (RAG)  
→ Reranking  
→ Identificação de riscos  
→ Derivação de requisitos

Principais tecnologias:

- Python 3.12
- Streamlit
- OpenAI API
- LangGraph
- LangChain
- ChromaDB
- FastEmbed
- Instructor
- Pydantic

## Bases de conhecimento

O MVP utiliza bases locais contendo conhecimento proveniente de:

- NIST AI RMF Playbook
- OWASP Top 10 para LLM/IA
- Design Patterns for AI-Based Systems

Os catálogos estão armazenados em:

    catalogs/

A base vetorial persistida está em:

    data/chroma/

Para o MVP, essas bases são consideradas estáticas e fazem parte do baseline do projeto.

## Estrutura principal

    us4ai_app/
    ├── app.py
    ├── catalogs/
    ├── data/
    │   └── chroma/
    ├── src/
    │   ├── config.py
    │   ├── ingestion.py
    │   ├── retriever.py
    │   ├── schemas.py
    │   └── graph/
    │       ├── nodes.py
    │       ├── state.py
    │       └── workflow.py
    ├── requirements.txt
    ├── requirements-lock.txt
    ├── .python-version
    └── README.md

## Ambiente validado

O baseline atual foi validado com:

    Python 3.12.14

Algumas das principais versões utilizadas:

    streamlit==1.64.0
    langchain==1.4.2
    langchain-chroma==1.1.0
    langchain-community==0.4.2
    langchain-openai==1.6.6
    langgraph==1.2.12
    chromadb==1.5.9
    fastembed==0.4.0
    instructor==1.17.0
    openai==3.3.0
    pandas==3.0.6
    pydantic==2.13.5
    pysqlite3-binary==0.5.4.post2

As versões completas do ambiente funcional estão registradas em:

    requirements-lock.txt

## Configuração

Crie um arquivo `.env` na raiz do projeto.

Exemplo:

    OPENAI_API_KEY=sua_chave_aqui
    CHROMA_PATH=./data/chroma

O arquivo `.env` não deve ser versionado.

## Instalação

Crie e ative um ambiente virtual:

    python -m venv .venv
    source .venv/bin/activate

Instale as dependências:

    pip install -r requirements.txt

## FastEmbed e reranking

O MVP utiliza:

    fastembed==0.4.0

O modelo utilizado para reranking é:

    BAAI/bge-reranker-base

Esse identificador é utilizado pelo `TextCrossEncoder` do FastEmbed 0.4.0.

## SQLite e ChromaDB

Em ambientes com uma versão antiga do SQLite, como Ubuntu 20.04, o projeto utiliza:

    pysqlite3-binary

Isso permite atender ao requisito de versão do SQLite utilizado pelo ChromaDB sem modificar o Python do sistema operacional.

## LLM Guard

A dependência `llm-guard` está temporariamente desabilitada no baseline atual.

O código da aplicação permite sua ausência e mantém o restante do pipeline funcional.

A proteção contra Prompt Injection poderá ser reavaliada em uma etapa posterior de evolução do MVP.

## Execução

Com o ambiente virtual ativo:

    streamlit run app.py

A aplicação normalmente ficará disponível em:

    http://localhost:8501

## Pipeline

O fluxo implementado com LangGraph executa as principais etapas:

1. Detecção de relação da User Story com IA.
2. Recuperação de documentos relevantes no ChromaDB.
3. Reranking dos documentos recuperados.
4. Identificação de riscos relacionados à IA.
5. Derivação de requisitos.
6. Apresentação dos riscos, requisitos e fontes recuperadas na interface Streamlit.

## Baseline

Este README descreve o baseline funcional validado antes da evolução arquitetural do US4AI.

O objetivo é preservar uma versão reproduzível do MVP antes da implementação de novas funcionalidades, melhorias no RAG, alterações nos prompts ou mudanças arquiteturais.
