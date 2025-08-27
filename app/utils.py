import os
import json
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Paths to FAISS index and metadata
FAISS_INDEX_FILE = "data/cpt_faiss.index"
METADATA_FILE = "data/cpt_metadata.json"

# Embedding model to use
EMBED_MODEL = "text-embedding-3-small"

# -------------------
# Utility functions
# -------------------

def load_faiss_index():
    """
    Load the FAISS index from file.
    Returns:
        faiss.Index: loaded FAISS index
    """
    if not os.path.exists(FAISS_INDEX_FILE):
        raise FileNotFoundError(f"FAISS index file not found: {FAISS_INDEX_FILE}")
    index = faiss.read_index(FAISS_INDEX_FILE)
    return index

def save_faiss_index(index):
    """
    Save the FAISS index to file.
    Args:
        index (faiss.Index): FAISS index to save
    """
    if not os.path.exists(os.path.dirname(FAISS_INDEX_FILE)):
        os.makedirs(os.path.dirname(FAISS_INDEX_FILE))
    faiss.write_index(index, FAISS_INDEX_FILE)


def load_metadata():
    """
    Load CPT metadata JSON.
    Returns:
        list[dict]: metadata list containing CPT codes, descriptions, and variants
    """
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def embed_text(text: str):
    """
    Generate embedding for a single text string using OpenAI embeddings.
    Args:
        text (str): text to embed
    Returns:
        np.ndarray: 1 x embedding_dim float32 array
    """
    response = client.embeddings.create(model=EMBED_MODEL, input=[text])
    vector = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
    return vector

def normalize_embedding(vec: np.ndarray):
    """
    Normalize an embedding vector for cosine similarity.
    Args:
        vec (np.ndarray): embedding vector
    Returns:
        np.ndarray: normalized vector
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm
