from app.utils import load_metadata

# Load metadata once at import
_metadata = load_metadata()

def reload_metadata():
    """
    Reload metadata from JSON file.
    Use this if the metadata has been updated dynamically.
    """
    global _metadata
    _metadata = load_metadata()

def search_by_cpt(cpt_code: str):
    """
    Lookup natural language variants for a given CPT code.
    Returns the latest variants from metadata, including any dynamically added entries.

    Args:
        cpt_code (str): CPT code to search

    Returns:
        list[str]: list of NL variants, empty if none found
    """
    # Ensure we have the latest metadata
    reload_metadata()

    variants = [
        entry.get("text", entry.get("nl_variants", []))  # handle old vs new keys
        for entry in _metadata
        if entry.get("CPT_Code") == cpt_code
    ]

    # Flatten in case of lists inside lists
    flattened = []
    for v in variants:
        if isinstance(v, list):
            flattened.extend(v)
        else:
            flattened.append(v)

    return flattened
