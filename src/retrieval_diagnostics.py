"""Separate research diagnostics derived from captured rankings and output references."""

from copy import deepcopy


def _single_report(result: dict) -> dict | None:
    """Report structural usage of final evidence without retrieval or generation calls."""
    captured = result.get("retrieval_diagnostics")
    if captured is None:
        return None
    report = deepcopy(captured)
    final_ids = report["final_evidence_ids"]
    risk_used = {identity for risk in result.get("contextual_risks", [])
                 for identity in risk.evidence_ids}
    requirement_used = {identity for requirement in result.get("ai_requirements", [])
                        for identity in requirement.evidence_ids}
    report["analysis_status"] = result.get("status", "unknown")
    report["risk_used_evidence_ids"] = [identity for identity in final_ids if identity in risk_used]
    report["requirement_used_evidence_ids"] = [identity for identity in final_ids if identity in requirement_used]
    report["unused_final_evidence_ids"] = [
        identity for identity in final_ids if identity not in risk_used | requirement_used
    ]
    report["evidence_usage"] = [
        {"evidence_id": identity, "used_by_risk": identity in risk_used,
         "used_by_requirement": identity in requirement_used,
         "usage": ("used_by_both" if identity in requirement_used else "used_by_risk_only")
         if identity in risk_used else
         ("used_by_requirement_only" if identity in requirement_used else "unused")}
        for identity in final_ids
    ]
    return report


def diagnostic_report(result: dict) -> dict | None:
    """Keep legacy risk diagnostics and expose independent stage membership and usage."""
    risk = _single_report(result)
    failure = {key: result[key] for key in ("failed_stage", "error_type", "error_message") if key in result}
    if risk is None:
        if not failure:
            return None
        return {"analysis_status": result.get("status", "failed"), **failure,
                "risk_retrieval": None, "treatment_retrieval": None,
                "cross_stage_summary": {}}
    treatment = _single_report({**result, "retrieval_diagnostics": result.get("treatment_diagnostics")})
    report = deepcopy(risk)
    report.update(failure)
    risk["used_evidence_ids"] = risk["risk_used_evidence_ids"]
    risk["unused_evidence_ids"] = [i for i in risk["final_evidence_ids"] if i not in risk["used_evidence_ids"]]
    if treatment is not None:
        treatment["used_evidence_ids"] = treatment["requirement_used_evidence_ids"]
        treatment["unused_evidence_ids"] = [i for i in treatment["final_evidence_ids"] if i not in treatment["used_evidence_ids"]]
    risk_ids = risk["final_evidence_ids"]
    treatment_ids = treatment["final_evidence_ids"] if treatment else []
    report["risk_retrieval"] = risk
    report["treatment_retrieval"] = treatment
    report["cross_stage_summary"] = {
        "risk_top_k_only": [i for i in risk_ids if i not in treatment_ids],
        "treatment_top_k_only": [i for i in treatment_ids if i not in risk_ids],
        "both_top_k": [i for i in risk_ids if i in treatment_ids],
        "used_for_risk_inference": risk["risk_used_evidence_ids"],
        "used_for_requirement_derivation": treatment["requirement_used_evidence_ids"] if treatment else [],
    }
    return report
