"""Streamlit interface for the NIST-grounded US4AI research prototype."""

import streamlit as st
from pydantic import ValidationError

from src.app_support import (
    AnalysisExecutionError, STATUS_MESSAGES, analysis_export, initialize_analysis_graph,
    input_from_form, run_analysis, traceability_csv, traceability_rows,
)
from src.config import OPENAI_API_KEY


st.set_page_config(page_title="US4AI | NIST-grounded requirements", layout="wide")


@st.cache_resource(show_spinner=False)
def get_analysis_graph():
    """Reuse the validated index and local model resources across UI reruns."""
    return initialize_analysis_graph()


def show_results(result):
    status = result.get("status", "failed")
    message = STATUS_MESSAGES.get(status, STATUS_MESSAGES["failed"])
    if status == "complete":
        st.success(message)
    elif status == "failed":
        st.error(message)
    else:
        st.warning(message)
    with st.expander("Input used for this analysis"):
        st.json(result["analysis_input"].model_dump())
        st.dataframe([
            {"task_id": key, **task.model_dump()}
            for key, task in result["ai_task_references"].items()
        ], hide_index=True)

    evidence_tab, risks_tab, requirements_tab, trace_tab = st.tabs([
        "Retrieved NIST Evidence", "Contextual AI Risks", "AI-specific Requirements", "Traceability",
    ])
    with evidence_tab:
        st.caption("Source guidance from the NIST AI RMF Playbook. Scores describe retrieval ordering, not risk severity or requirement importance.")
        st.caption("Vector distance: lower is better. Reranking score: raw relative-ordering score, not a probability.")
        for item in result.get("retrieved_evidence", []):
            with st.expander(f"{item.evidence_id} | {item.type} | {item.category}"):
                st.write({"Vector distance": item.vector_distance, "Reranking score": item.reranking_score})
                st.text(item.retrieval_text)
        if not result.get("retrieved_evidence"):
            st.info("No NIST evidence is available for this analysis.")
    with risks_tab:
        st.caption("These are contextual inferences grounded in retrieved NIST guidance, not risks quoted from NIST.")
        for risk in result.get("contextual_risks", []):
            st.subheader(f"{risk.risk_id}: {risk.title}")
            st.write(risk.description)
            st.write("AI Tasks: " + ", ".join(risk.ai_task_ids))
            st.write("NIST evidence: " + ", ".join(risk.evidence_ids))
        if not result.get("contextual_risks"):
            st.info("No supported contextual risks are available. This is not a risk-free assessment.")
    with requirements_tab:
        st.caption("Risk-informed requirements may address the system, governance, human oversight, monitoring, or project processes.")
        for requirement in result.get("ai_requirements", []):
            st.subheader(requirement.requirement_id)
            st.write(requirement.statement)
            st.write("Contextual risks: " + ", ".join(requirement.risk_ids))
            st.write("AI Tasks: " + ", ".join(requirement.ai_task_ids))
            st.write("NIST evidence: " + ", ".join(requirement.evidence_ids))
            st.write("Rationale: " + requirement.rationale)
        if not result.get("ai_requirements"):
            st.info("No validated requirements are available; review any preserved risks and evidence.")
    with trace_tab:
        st.caption("Each row is an explicit validated reference. Direct AI Task–NIST Evidence links are not inferred from shared references.")
        rows = traceability_rows(result)
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        else:
            st.info("No generated traceability relationships are available.")
    st.download_button("Download analysis JSON", analysis_export(result), "us4ai_analysis.json", "application/json")
    st.download_button("Download traceability CSV", traceability_csv(result), "us4ai_traceability.csv", "text/csv")


st.title("US4AI")
st.write("NIST-grounded contextual AI risk inference and risk-informed requirements for research review.")
with st.sidebar:
    st.header("Knowledge base")
    st.write("NIST AI RMF Playbook · 72 subcategories")
    st.caption("The existing index is validated and reused. Incompatible indexes are never silently rebuilt.")

try:
    with st.spinner("Initializing and validating the NIST index and local models..."):
        graph = get_analysis_graph()
except Exception as error:
    st.error(f"NIST initialization failed ({type(error).__name__}). Verify the authoritative source and configured NIST_CHROMA_PATH. Run `python -m src.nist_index` for validation. Back up an incompatible index and choose a fresh path; existing data is not automatically deleted.")
    st.stop()

if not OPENAI_API_KEY:
    st.warning("Generation requires OPENAI_API_KEY in the environment or project .env file. Configure it and restart the application.")

with st.form("analysis_input"):
    purpose = st.text_area("System Purpose", help="Required: describe what the system is intended to do.")
    story = st.text_area("User Story", help="Required: describe the stakeholder need.")
    criteria = st.text_area("Acceptance Criteria (optional, one per line)")
    st.write("AI Tasks (at least one)")
    tasks = st.data_editor(
        [{"category": "", "task": ""}], num_rows="dynamic", hide_index=True,
        column_config={
            "category": st.column_config.TextColumn("Category", help="Use your own category; no fixed taxonomy."),
            "task": st.column_config.TextColumn("Task", help="Describe the AI task."),
        }, key="ai_tasks_editor", width="stretch",
    )
    submitted = st.form_submit_button("Run analysis", disabled=not bool(OPENAI_API_KEY))

if submitted:
    st.session_state.pop("analysis_result", None)
    try:
        scenario = input_from_form(purpose, story, criteria, tasks)
    except ValidationError as error:
        labels = {"system_purpose": "System Purpose", "user_story": "User Story", "ai_tasks": "AI Tasks"}
        for issue in error.errors(include_input=False):
            location = issue["loc"]
            label = labels.get(location[0], str(location[0]))
            if location[0] == "ai_tasks" and len(location) > 2:
                label += f" row {location[1] + 1}, {location[2]}"
            st.error(f"{label}: provide a non-empty value (at least one complete category/task row is required).")
    else:
        try:
            with st.spinner("Retrieving NIST evidence, inferring contextual risks, and deriving requirements..."):
                st.session_state["analysis_result"] = run_analysis(graph, scenario)
        except AnalysisExecutionError as error:
            st.session_state["analysis_result"] = error.partial_state
            st.error(f"Analysis failed ({error.error_type}). Check API access and index compatibility. Invalid generated references are rejected rather than repaired. Retry only when ready.")

if "analysis_result" in st.session_state:
    show_results(st.session_state["analysis_result"])
