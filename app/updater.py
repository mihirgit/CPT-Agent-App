import json
import numpy as np
from pathlib import Path
from .utils import embed_text, load_faiss_index, save_faiss_index, normalize_embedding

# -----------------------
# Paths
# -----------------------
METADATA_FILE = Path("data/cpt_metadata.json")
FAISS_INDEX_FILE = Path("data/cpt_faiss.index")

# -----------------------
# Load existing data
# -----------------------
def load_metadata():
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_metadata(metadata):
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

# -----------------------
# Main Updater Class
# -----------------------
class CPTUpdater:
    def __init__(self):
        self.metadata = load_metadata()
        self.faiss_index = load_faiss_index()  # Assumes utils.py has a load function

    # -----------------------
    # Add new CPT code with variants
    # -----------------------
    def add_new_cpt(self, cpt_code, formal_description, nl_variants):
        """
        Add a new CPT code and its NL variants
        """
        # Check if CPT already exists
        existing = [m for m in self.metadata if m["CPT_Code"] == cpt_code]
        if existing:
            raise ValueError(f"CPT {cpt_code} already exists. Use add_variants instead.")

        # Normalize and deduplicate NL variants
        nl_variants = list({v.strip() for v in nl_variants if v.strip()})

        # Append to metadata
        new_entry = {
            "CPT_Code": cpt_code,
            "formal_description": formal_description,
            "nl_variants": nl_variants
        }
        self.metadata.append(new_entry)
        save_metadata(self.metadata)

        # Add embeddings to FAISS
        self._add_to_faiss(nl_variants, cpt_code)

        return new_entry

    # -----------------------
    # Add NL variants to existing CPT code
    # -----------------------
    def add_variants(self, cpt_code, new_variants):
        """
        Add new NL variants for an existing CPT code
        """
        entry = next((m for m in self.metadata if m["CPT_Code"] == cpt_code), None)
        if not entry:
            raise ValueError(f"CPT {cpt_code} does not exist. Use add_new_cpt instead.")

        # Normalize and deduplicate
        new_variants = list({v.strip() for v in new_variants if v.strip()})
        existing_set = set(entry["nl_variants"])
        variants_to_add = [v for v in new_variants if v not in existing_set]
        if not variants_to_add:
            return entry  # Nothing to add

        # Update metadata
        entry["nl_variants"].extend(variants_to_add)
        save_metadata(self.metadata)

        # Add embeddings to FAISS
        self._add_to_faiss(variants_to_add, cpt_code)

        return entry

    # -----------------------
    # Internal: add embeddings to FAISS
    # -----------------------
    def _add_to_faiss(self, texts, cpt_code):
        vectors = [normalize_embedding(embed_text(t)) for t in texts]
        # FAISS index expects a 2D numpy array
        vectors_array = np.array(vectors).astype("float32")
        self.faiss_index.add(vectors_array)
        save_faiss_index(self.faiss_index)
