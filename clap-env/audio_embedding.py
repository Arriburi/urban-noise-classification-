import glob
import os

import laion_clap
import numpy as np

INPUT_DIR = "/home/lucaa/audio_data/unc/audioset/eval_set_flac"
OUTPUT_NPZ = "clap_audio_embeddings.npz"


def find_audio_files(root: str) -> list[str]:
    patterns = [
        os.path.join(root, "**", "*.wav"),
        os.path.join(root, "**", "*.flac"),
    ]
    files: set[str] = set()
    for pattern in patterns:
        files.update(glob.glob(pattern, recursive=True))
    return sorted(files)


def extract_video_id(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("_", 1)
    return parts[1] if len(parts) == 2 else stem ## just return after _


model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()

audio_files = find_audio_files(INPUT_DIR)

print(f"Found {len(audio_files)} audio files to process")

batch_size = 8
all_embeddings: list[np.ndarray] = []

for i in range(0, len(audio_files), batch_size):
    batch_files = audio_files[i : i + batch_size]
    print(f"Processing batch {i//batch_size + 1}/{(len(audio_files)-1)//batch_size + 1}")

    try:
        batch_embeddings = model.get_audio_embedding_from_filelist(
            x=batch_files, use_tensor=False
        )
        all_embeddings.append(batch_embeddings)
    except Exception as e:
        print(f"Error processing batch: {e}")
        exit(1)

audio_embed = np.vstack(all_embeddings)
paths_array = np.asarray(audio_files)
video_ids = np.asarray([extract_video_id(path) for path in audio_files])
print(f"Final embeddings shape: {audio_embed.shape}")

np.savez(
    OUTPUT_NPZ,
    embeddings=audio_embed,
    paths=paths_array,
    video_ids=video_ids,
)
print(f"Saved embeddings + paths + video_ids to: {OUTPUT_NPZ}")

