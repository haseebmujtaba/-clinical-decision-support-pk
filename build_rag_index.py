"""
build_rag_index.py
===================
Phase 2 - Step 3: Build a local ChromaDB vector index over
knowledge_base.json so the LLM layer can retrieve grounded,
citation-backed chunks instead of relying on its own (potentially
hallucinated) medical knowledge.

Embedding model: sentence-transformers "all-MiniLM-L6-v2" (local, free,
CPU-friendly, ~80MB).

Indexing strategy
------------------
For each condition in knowledge_base.json, we create ONE chunk PER
MEDICINE plus ONE "condition overview" chunk. This keeps each chunk
focused enough for good retrieval while preserving the citation that
must travel with it.

Each chunk's metadata carries everything citation_validator.py and
llm_pipeline.py need downstream:
    - condition_id, condition_name, icd10
    - chunk_type: "overview" | "medicine"
    - medicine_name (for medicine chunks)
    - tier (1-4)
    - citation (the exact source string)

Run:
    python build_rag_index.py

Produces a persistent ChromaDB store at ./chroma_db/
"""
import json
import os
import sys

import chromadb
from chromadb.utils import embedding_functions

KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "cdss_kb_phase2"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def load_kb(path: str = KB_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_chunks(kb: dict):
    """
    Yield (chunk_id, text, metadata) tuples for every condition
    overview and every medicine entry in the knowledge base.
    """
    for cond in kb["conditions"]:
        cond_id = cond["condition_id"]
        cond_name = cond["condition_name"]
        icd10 = cond.get("icd10", "")

        # --- Condition overview chunk ---
        differentials = ", ".join(d["name"] for d in cond.get("differentials", []))
        overview_text = (
            f"Condition: {cond_name} (ICD-10: {icd10}). "
            f"Typical vitals/presentation: {cond.get('key_vitals_pattern', '')} "
            f"Differential diagnoses to consider: {differentials}. "
            f"Non-pharmacological management: "
            f"{cond.get('treatment_plan', {}).get('non_pharmacological', '')} "
            f"Confidence notes: {cond.get('confidence_notes', '')}"
        )
        yield (
            f"{cond_id}__overview",
            overview_text,
            {
                "condition_id": cond_id,
                "condition_name": cond_name,
                "icd10": icd10,
                "chunk_type": "overview",
                "medicine_name": "",
                "tier": 0,
                "citation": "",
            },
        )

        # --- One chunk per medicine ---
        medicines = cond.get("treatment_plan", {}).get("medicines", [])
        for i, med in enumerate(medicines):
            med_text = (
                f"Condition: {cond_name} (ICD-10: {icd10}). "
                f"Medicine: {med['name']}. "
                f"Dosage form: {med.get('dosage_form', '')}. "
                f"Dose instructions: {med.get('dose_instruction', '')} "
                f"Source citation (Tier {med.get('tier', '?')}): {med.get('citation', '')}"
            )
            yield (
                f"{cond_id}__med_{i}",
                med_text,
                {
                    "condition_id": cond_id,
                    "condition_name": cond_name,
                    "icd10": icd10,
                    "chunk_type": "medicine",
                    "medicine_name": med["name"],
                    "tier": med.get("tier", 4),
                    "citation": med.get("citation", ""),
                },
            )


def main():
    kb = load_kb()
    print(f"Loaded knowledge base: {len(kb['conditions'])} conditions "
          f"(version: {kb.get('_meta', {}).get('version', 'unknown')})")

    print(f"Loading embedding model '{EMBED_MODEL_NAME}' "
          f"(first run downloads ~80MB, then cached locally)...")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL_NAME
    )

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Fresh build each run: drop any existing collection with this name.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"description": "CDSS Phase 2 starter KB (WHO EML 24th List 2025)"},
    )

    ids, texts, metadatas = [], [], []
    for chunk_id, text, meta in build_chunks(kb):
        ids.append(chunk_id)
        texts.append(text)
        metadatas.append(meta)

    print(f"Indexing {len(ids)} chunks "
          f"({sum(1 for m in metadatas if m['chunk_type'] == 'overview')} overviews, "
          f"{sum(1 for m in metadatas if m['chunk_type'] == 'medicine')} medicines)...")

    collection.add(ids=ids, documents=texts, metadatas=metadatas)

    print(f"Done. Persisted ChromaDB collection '{COLLECTION_NAME}' at: {CHROMA_DIR}")

    # Quick sanity-check query
    if "--test" in sys.argv or True:
        print("\n--- Sanity check query: 'fever and chills, possible malaria' ---")
        results = collection.query(
            query_texts=["fever and chills, possible malaria"],
            n_results=3,
        )
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            print(f"  [{meta['condition_id']} / {meta['chunk_type']} / "
                  f"{meta.get('medicine_name') or '-'}] dist={dist:.4f}")
            print(f"    {doc[:120]}...")


if __name__ == "__main__":
    main()
