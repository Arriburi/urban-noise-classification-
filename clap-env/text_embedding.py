import json
from pathlib import Path

import numpy as np
import laion_clap

ONTOLOGY_PATH = Path("ontology.json")
EMBEDDINGS_PATH = Path("clap_text_embeddings.npy")


def build_prompts() -> list[str]:
    with ONTOLOGY_PATH.open("r", encoding="utf-8") as f:
        ontology = json.load(f)

    prompts: list[str] = []
    for entry in ontology:
        name = entry.get("name", "").strip()
        description = entry.get("description", "").strip()
        if description:
            prompt = f"{name}: {description}"
        else:
            prompt = name
        prompts.append(prompt)
    return prompts


def main() -> None:
    prompts = build_prompts()

    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()

    text_embeddings = model.get_text_embedding(prompts, use_tensor=False)

    np.save(EMBEDDINGS_PATH, text_embeddings)
    print(f"Saved text embeddings to: {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()