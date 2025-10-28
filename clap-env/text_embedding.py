import numpy as np
import json
import laion_clap


model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt() 

classes_file = "classes.json"
with open(classes_file, 'r') as f:
    classes = json.load(f)

text_prompts = ["This is a sound of " + class_name for class_name in classes]

try:
    text_embeddings = model.get_text_embedding(text_prompts, use_tensor=False)
except Exception as e:
    exit(1)

embeddings_path = "clap_text_embeddings.npy"
np.save(embeddings_path, text_embeddings)
print(f"Saved text embeddings to: {embeddings_path}")

