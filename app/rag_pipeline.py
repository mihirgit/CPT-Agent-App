from app.utils import load_faiss_index, load_metadata, embed_text, normalize_embedding
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import faiss

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# LLM model for RAG
RAG_MODEL = "gpt-4o-mini"  # can adjust to gpt-4 or gpt-4.1-mini
TOP_K = 5  # number of candidates to retrieve from FAISS

# Load FAISS index and metadata once
_index = load_faiss_index()
_metadata = load_metadata()


# -------------------
# RAG Functions
# -------------------

def retrieve_candidates(query: str, top_k: int = TOP_K):
    """
    Retrieve top-k CPT candidates from FAISS given a doctor's note.
    Args:
        query (str): natural language doctor's note
        top_k (int): number of nearest neighbors to retrieve
    Returns:
        list[dict]: retrieved candidates from metadata
    """
    query_emb = embed_text(query)
    query_emb = normalize_embedding(query_emb)

    # Normalize FAISS index vectors for cosine similarity
    faiss.normalize_L2(_index.reconstruct_n(0, _index.ntotal))

    distances, indices = _index.search(query_emb, top_k)

    candidates = [_metadata[idx] for idx in indices[0] if idx >= 0]
    return candidates


def generate_cpt_suggestion(query: str, candidates: list):
    """
    Given a doctor's note and retrieved candidates, generate structured CPT suggestion via LLM.
    Args:
        query (str): doctor's note
        candidates (list[dict]): retrieved FAISS candidates
    Returns:
        dict: structured JSON with CPT code, description, reasoning
    """
    if not candidates:
        return {"error": "No candidates retrieved from FAISS"}

    # Prepare context for LLM
    context_texts = "\n".join(
        [f"- CPT {c['CPT_Code']} ({c['source']}): {c['text']}" for c in candidates]
    )

    prompt = f"""
You are a medical coding assistant.
A doctor wrote the following note: "{query}"

From the retrieved CPT candidates below, select the most appropriate CPT code(s) for this note.
Provide a short reasoning why it matches the note.
Format the output as JSON like:

{{
  "CPT_Code": "<code>",
  "Description": "<formal description / explanation>",
  "Reasoning": "<why this CPT code fits the note>"
}}

Candidates:
{context_texts}
"""

    try:
        response = client.chat.completions.create(
            model=RAG_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        text = response.choices[0].message.content.strip()

        # Try to parse JSON
        return json.loads(text)
    except Exception as e:
        # fallback in case JSON parsing fails
        return {"raw_output": text, "error": str(e)}


# Optional: convenience function for full RAG flow
def rag_query(query: str, top_k: int = TOP_K):
    """
    Full RAG flow: retrieve + LLM generation.
    Args:
        query (str): doctor's note
        top_k (int): number of FAISS candidates
    Returns:
        dict: structured output
    """
    candidates = retrieve_candidates(query, top_k)
    return generate_cpt_suggestion(query, candidates)
