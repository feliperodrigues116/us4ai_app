import instructor
from openai import OpenAI
from pydantic import BaseModel
from src.schemas import OutputRisks, OutputRequirements
from src.retriever import HybridRetriever
from src.config import OPENAI_API_KEY

# Usamos o instructor para garantir as respostas estruturadas Pydantic nativamente
client = instructor.from_openai(OpenAI(api_key=OPENAI_API_KEY))

class AIDetectorOutput(BaseModel):
    is_ai_related: bool
    reason: str

def detect_ai_node(state: dict) -> dict:
    """Nó 1: Avalia rapidamente se a User Story envolve inteligência artificial."""
    story = state.get("description", "")
    
    prompt = f"Analyze if this user story requires Artificial Intelligence (LLMs, ML, Chatbots, Predictors, etc): {story}"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=AIDetectorOutput,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return {
        "is_ai_related": response.is_ai_related,
        "ai_reason": response.reason
    }

def risk_identification_node(state: dict) -> dict:
    """Nó 2: Recupera catálogos e identifica riscos ancorados no conhecimento."""
    story = state.get("description", "")
    criteria = "\n".join(state.get("acceptance_criteria", []))
    
    query = f"{story}\n{criteria}"
    retriever = HybridRetriever()
    retrieved_docs = retriever.get_relevant_knowledge(query, top_k=5)
    
    if not retrieved_docs:
        return {"retrieved_docs": [], "identified_risks": []}

    context = "\n".join([str(doc) for doc in retrieved_docs])
    
    prompt = f"""
    Based ONLY on the retrieved knowledge items below, identify AI risks for this User Story.
    You MUST NOT invent risks. You MUST map your risks explicitly to the knowledge_item_id provided.
    Story: {story}
    Criteria: {criteria}
    
    Knowledge Items: 
    {context}
    """
    
    response = client.chat.completions.create(
         model="gpt-4o-mini",
         response_model=OutputRisks,
         messages=[{"role": "user", "content": prompt}]
    )
    
    # Validação anti-alucinação rudimentar
    valid_ids = {doc.get("knowledge_item_id") for doc in retrieved_docs}
    risks_dict = []
    
    for risk in response.identified_risks:
        valid_sources = [s for s in risk.grounded_in if s.knowledge_item_id in valid_ids]
        if valid_sources:
            risk.grounded_in = valid_sources
            risks_dict.append(risk.model_dump())
            
    return {
        "retrieved_docs": retrieved_docs,
        "identified_risks": risks_dict
    }

def requirement_derivation_node(state: dict) -> dict:
    """Nó 3: Deriva requisitos formais a partir dos riscos mapeados."""
    story = state.get("description", "")
    risks = state.get("identified_risks", [])
    
    if not risks:
         return {"derived_requirements": []}
         
    prompt = f"""
    Derive specific AI requirements to mitigate the identified risks below.
    Each requirement MUST start with "The system shall...".
    Story: {story}
    Risks: {risks}
    """
    
    response = client.chat.completions.create(
         model="gpt-4o-mini",
         response_model=OutputRequirements,
         messages=[{"role": "user", "content": prompt}]
    )
    
    reqs_dict = [req.model_dump() for req in response.derived_ai_requirements]
    
    return {"derived_requirements": reqs_dict}
