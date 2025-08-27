# from app.rag_pipeline import rag_query, retrieve_candidates
# from app.cpt_lookup import search_by_cpt
# from app.utils import embed_text, normalize_embedding
# import numpy as np
#
#
# # -----------------------
# # Helper functions
# # -----------------------
#
# def cosine_similarity(vec1, vec2):
#     """Compute cosine similarity between two vectors"""
#     vec1 = normalize_embedding(vec1)
#     vec2 = normalize_embedding(vec2)
#     return float(np.dot(vec1, vec2.T))
#
#
# def calculate_confidence(note, suggestion, candidates):
#     """
#     Simple confidence aggregation based on:
#     - FAISS similarity of note to top candidates
#     - Optional LLM consensus (placeholder for now)
#     """
#     note_emb = embed_text(note)
#
#     # FAISS similarity scores
#     sim_scores = []
#     for c in candidates:
#         c_emb = embed_text(c["text"])
#         sim_scores.append(cosine_similarity(note_emb, c_emb))
#
#     if sim_scores:
#         retrieval_score = max(sim_scores)
#     else:
#         retrieval_score = 0.0
#
#     # Placeholder LLM consensus score (set to 1.0 if no extra LLM)
#     llm_score = 1.0
#
#     # Aggregate confidence (weighted)
#     confidence = 0.6 * retrieval_score + 0.4 * llm_score
#     return round(confidence, 2)
#
#
# # -----------------------
# # Main Agent Function
# # -----------------------
#
# def agentic_cpt_suggestion(note, top_k=5):
#     """
#     Full agentic flow: retrieves candidates, generates CPT suggestion,
#     computes confidence, and decides next action.
#     """
#     # Step 1: retrieve candidates
#     candidates = retrieve_candidates(note, top_k=top_k)
#
#     # Step 2: initial RAG suggestion
#     suggestion = rag_query(note, top_k=top_k)
#
#     # Step 3: compute confidence
#     confidence = calculate_confidence(note, suggestion, candidates)
#     suggestion["Confidence"] = confidence
#
#     # Step 4: decide next action
#     if confidence < 0.7:
#         suggestion["Next_Action"] = "Ask clarifying question or show multiple CPT candidates"
#     else:
#         suggestion["Next_Action"] = "Accept suggestion"
#
#     return suggestion
#
#
# # -----------------------
# # Optional: Agentic reverse lookup
# # -----------------------
#
# def agentic_cpt_reverse_lookup(cpt_code):
#     """
#     Return NL variants and confidence summary for a given CPT code
#     """
#     variants = search_by_cpt(cpt_code)
#     confidence = 1.0 if variants else 0.0
#
#     return {
#         "CPT_Code": cpt_code,
#         "NL_Variants": variants,
#         "Confidence": confidence,
#         "Next_Action": "Accept" if variants else "Review required"
#     }


import os
import json
import numpy as np
from typing import List, Dict, Any

from openai import OpenAI
from app.rag_pipeline import rag_query, retrieve_candidates
from app.utils import embed_text, normalize_embedding

# Use env var for auth
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------
# Similarity helpers
# -----------------------

def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    vec1 = normalize_embedding(vec1)
    vec2 = normalize_embedding(vec2)
    return float(np.dot(vec1, vec2.T))

def _retrieval_score(note: str, candidates: List[Dict[str, Any]]) -> float:
    """Max cosine similarity between note and retrieved candidate texts."""
    if not candidates:
        return 0.0
    note_emb = embed_text(note)
    sims = []
    for c in candidates:
        text = c.get("text", "")
        if not text:
            continue
        c_emb = embed_text(text)
        sims.append(_cosine_similarity(note_emb, c_emb))
    return max(sims) if sims else 0.0

# -----------------------
# Self-critique (verification) step
# -----------------------

def _verify_suggestion(note: str,
                       suggestion: Dict[str, Any],
                       candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ask the model to verify the suggested CPT against the note + retrieved snippets.
    Returns a compact, structured summary (NO chain-of-thought).
    """
    # Package minimal, privacy-safe context
    safe_candidates = [{"text": (c.get("text", "") or "")[:240]} for c in candidates[:5]]
    payload = {
        "note": note[:1200],  # cap size
        "suggested": {
            "CPT_Code": suggestion.get("CPT_Code"),
            "Description": suggestion.get("Description"),
        },
        "retrieved_snippets": safe_candidates
    }

    system_msg = (
        "You are a medical coding verifier. Assess whether the suggested CPT matches "
        "the doctor's note, using the retrieved snippets as evidence. "
        "Return ONLY JSON with keys: "
        "verdict ('pass'|'warn'|'fail'), "
        "short_rationale (<=2 sentences), "
        "missing_info (list of short strings), "
        "clarifying_questions (list of short questions), "
        "supporting_snippets (list of short quotes/paraphrases). "
        "Be concise. Do not include chain-of-thought."
    )
    user_msg = (
        "Data:\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n\nRespond with JSON only."
    )

    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = resp.output_text
        # Attempt to parse JSON (strip code fences if any)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            # possible leading 'json'
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].lstrip()
        verification = json.loads(cleaned)
        verification["raw_verification_output"] = raw
        return verification
    except Exception as e:
        # Safe fallback
        return {
            "verdict": "warn",
            "short_rationale": "Automatic verification unavailable; recommend human review if uncertain.",
            "missing_info": [],
            "clarifying_questions": [],
            "supporting_snippets": [],
            "raw_verification_output": f"parse_error: {str(e)}"
        }

# -----------------------
# Confidence aggregation
# -----------------------

def _aggregate_confidence(retrieval_score: float, verdict: str) -> float:
    """
    Combine retrieval score (0-1) with verification verdict into a final confidence.
    'pass' boosts, 'warn' neutral, 'fail' penalizes.
    """
    verdict_factor = {"pass": 0.95, "warn": 0.65, "fail": 0.35}.get(verdict, 0.6)
    # Weighted aggregate
    conf = 0.6 * retrieval_score + 0.4 * verdict_factor
    # Clamp and round
    return round(max(0.0, min(1.0, conf)), 2)

def _decide_next_action(confidence: float, verdict: str) -> str:
    if verdict == "fail" or confidence < 0.5:
        return "Low confidence — ask clarifying questions or escalate for manual review."
    if confidence < 0.7 or verdict == "warn":
        return "Medium confidence — show top candidates and consider clarifications."
    return "Accept suggestion"

# -----------------------
# Public API
# -----------------------

def agentic_cpt_suggestion(note: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Full agentic flow with a light self-critique loop.
    - Retrieve candidates (FAISS)
    - Generate initial suggestion (RAG)
    - Verify suggestion (self-critique)
    - Aggregate confidence & next action
    - Return structured result + concise verification summary (no chain-of-thought)
    """
    # Retrieve evidence
    candidates = retrieve_candidates(note, top_k=top_k) or []

    # Initial suggestion
    suggestion = rag_query(note, top_k=top_k) or {}
    # Ensure presence of keys expected downstream
    suggestion.setdefault("CPT_Code", suggestion.get("cpt_code"))
    suggestion.setdefault("Description", suggestion.get("description"))
    suggestion.setdefault("Reasoning", suggestion.get("reason"))

    # Self-critique
    verification = _verify_suggestion(note, suggestion, candidates)

    # Confidence
    r_score = _retrieval_score(note, candidates)
    confidence = _aggregate_confidence(r_score, verification.get("verdict", "warn"))

    # Next action
    next_action = _decide_next_action(confidence, verification.get("verdict", "warn"))

    # Compact evidence preview for UI
    evidence_preview = [{"text": (c.get("text", "") or "")[:160]} for c in candidates[:3]]

    result = {
        **suggestion,
        "Confidence": confidence,
        "Next_Action": next_action,
        "Verification": {
            "verdict": verification.get("verdict"),
            "short_rationale": verification.get("short_rationale"),
            "missing_info": verification.get("missing_info", []),
            "clarifying_questions": verification.get("clarifying_questions", []),
            "supporting_snippets": verification.get("supporting_snippets", []),
        },
        # Keep raw fields for debugging expanders in UI
        "raw_output": suggestion.get("raw_output"),
        "error": suggestion.get("error"),
        "raw_verification_output": verification.get("raw_verification_output"),
        "Evidence": evidence_preview
    }
    return result


def agentic_cpt_reverse_lookup(cpt_code: str) -> Dict[str, Any]:
    """
    (Unchanged) Simple wrapper for NL variants lookup + trivial confidence.
    """
    from .cpt_lookup import search_by_cpt
    variants = search_by_cpt(cpt_code) or []
    confidence = 1.0 if variants else 0.0
    return {
        "CPT_Code": cpt_code,
        "NL_Variants": variants,
        "Confidence": confidence,
        "Next_Action": "Accept" if variants else "Review required"
    }
