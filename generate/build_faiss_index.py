"""
1. Loads your JSON (CPT codes + variants).

2. Flattens the data into a list of texts to embed.

3. Each entry will keep metadata like CPT_code, source (description or variant), and the actual text.

    Uses text-embedding-3-large (or text-embedding-3-small) to create embeddings.

4. Stores everything in a FAISS index + a metadata store (JSON/CSV/SQLite).

5. Saves the index to disk so you can load it later in your web app.


"""

import os
import json
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# File paths
INPUT_JSON = "cpt_with_nl_variants.json"   # your enriched CPT+variants file
FAISS_INDEX_FILE = "../data/cpt_faiss.index"
METADATA_FILE = "../data/cpt_metadata.json"

# Embedding model
EMBED_MODEL = "text-embedding-3-small"  # cheaper, or "text-embedding-3-large"

def get_embedding(text: str):
    """Generate embedding for a given text using OpenAI embeddings API."""
    response = client.embeddings.create(
        input=text,
        model=EMBED_MODEL
    )
    return response.data[0].embedding

def build_index():
    # Load CPT data
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        cpt_data = json.load(f)

    texts = []
    metadata = []

    # Flatten CPT data: description + variants
    for entry in cpt_data:
        code = entry["CPT_Code"]
        description = entry["formal_description"]
        variants = entry.get("nl_variants", [])

        # Add description
        texts.append(description)
        metadata.append({
            "CPT_Code": code,
            "source": "description",
            "text": description
        })

        # Add each variant
        for v in variants:
            texts.append(v)
            metadata.append({
                "CPT_Code": code,
                "source": "variant",
                "text": v
            })

    print(f"Total texts to embed: {len(texts)}")

    # Generate embeddings in batches (to avoid rate limits)
    batch_size = 50
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model=EMBED_MODEL
        )
        batch_embeddings = [d.embedding for d in response.data]
        all_embeddings.extend(batch_embeddings)
        print(f"Embedded {i+len(batch)}/{len(texts)}")

    # Convert to numpy array
    embeddings_np = np.array(all_embeddings, dtype="float32")

    # Build FAISS index
    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dim)  # L2 distance index
    index.add(embeddings_np)

    # Save FAISS index
    faiss.write_index(index, FAISS_INDEX_FILE)
    print(f"FAISS index saved to {FAISS_INDEX_FILE}")

    # Save metadata
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {METADATA_FILE}")

if __name__ == "__main__":
    build_index()
