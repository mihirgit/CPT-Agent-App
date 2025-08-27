"""
Code to build synthetic data: Natural language variations for descriptions of CPT codes stored in csv file

"""


import pandas as pd
from openai import OpenAI
import json
import time
import os
from typing import List
from dotenv import load_dotenv
import re

# ------------------------------
# LOAD ENV VARIABLES
# ------------------------------
load_dotenv()  # loads variables from .env

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INPUT_CSV = "E:/CPT-agent-prototype/data-cleaning/clean_cpt_codes.csv"
OUTPUT_JSON = os.getenv("OUTPUT_JSON", "../data/cpt_with_nl_variants.json")
NUM_VARIANTS = int(os.getenv("NUM_VARIANTS", 10))
MODEL = os.getenv("MODEL", "gpt-4o-mini")
RETRY_LIMIT = 3
SLEEP_BETWEEN_CALLS = 0.5  # seconds

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------
# FUNCTION: Generate NL variants
# ------------------------------
def generate_nl_variants(description: str, num_variants: int = NUM_VARIANTS, model: str = MODEL) -> List[str]:
    """
    Generate natural language variants for a CPT description using OpenAI API.
    Returns a list of strings.
    """
    prompt = f"""
You are a medical assistant. A CPT code corresponds to a clinical procedure. 
Your task is to generate {num_variants} distinct ways a doctor might describe this procedure in natural language in patient notes or medical documentation.
Each variant should be concise, professional, and medically accurate. 
Do not include the CPT code. 
Provide your answer strictly as a JSON list of strings.

Procedure: "{description}"
"""
    for attempt in range(RETRY_LIMIT):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            content = response.choices[0].message.content.strip()
            variants = extract_json_list(content)
            if isinstance(variants, list) and variants:
                return variants
            else:
                raise ValueError("No valid JSON list found.")
        except Exception as e:
            print(f"[Attempt {attempt + 1}] Error generating variants for '{description}': {e}")
            time.sleep(2)
    return []


def extract_json_list(text: str):
    # Match first JSON list in the output
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        import json
        return json.loads(match.group())
    else:
        return []

# ------------------------------
# FUNCTION: Load CPT CSV
# ------------------------------
def load_cpt_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {'CPT_Code', 'Description'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    return df

# ------------------------------
# FUNCTION: Save results
# ------------------------------
def save_json(data: List[dict], path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# ------------------------------
# MAIN PIPELINE
# ------------------------------
def main():
    df = load_cpt_data(INPUT_CSV)
    enriched_data = []

    # Load existing progress if exists
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r") as f:
            enriched_data = json.load(f)
        completed_codes = {entry["CPT_Code"] for entry in enriched_data}
    else:
        completed_codes = set()

    for _, row in df.iterrows():
        cpt_code = row['CPT_Code']
        description = row['Description']

        if cpt_code in completed_codes:
            print(f"Skipping {cpt_code} (already processed)")
            continue

        print(f"Generating NL variants for CPT {cpt_code}: {description}")
        variants = generate_nl_variants(description, NUM_VARIANTS, MODEL)

        enriched_data.append({
            "CPT_Code": cpt_code,
            "formal_description": description,
            "nl_variants": variants
        })

        # Save intermediate results
        save_json(enriched_data, OUTPUT_JSON)
        time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"All CPT codes processed. Results saved to {OUTPUT_JSON}")

# ------------------------------
# ENTRY POINT
# ------------------------------
if __name__ == "__main__":
    main()
