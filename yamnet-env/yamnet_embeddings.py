import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import tensorflow as tf
import tensorflow_hub as hub


# =============================
# Configuration
# =============================
INPUT_DIR = Path("/home/lucaa/audio_data/resampled16")
TARGET_SAMPLE_RATE = 16000
EMBEDDING_DIMENSION = 1024

# Output files (memmap and paths list)
OUTPUT_DIR = Path("/home/lucaa/audio_data/unc/yamnet-env")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_PATH = OUTPUT_DIR / "yamnet_embeddings_memmap.dat"
PATHS_TXT_PATH = OUTPUT_DIR / "yamnet_paths.txt"


def list_audio_files(input_dir: Path) -> List[str]:
    files = sorted(str(p) for p in input_dir.rglob("*.wav"))
    return files


def read_wav_as_float_mono_16k(path: str) -> np.ndarray:
    sample_rate, wav = wavfile.read(path)

    # Convert to float32 [-1, 1]
    if np.issubdtype(wav.dtype, np.integer):
        max_val = np.iinfo(wav.dtype).max
        wav = wav.astype(np.float32) / float(max_val)
    else:
        wav = wav.astype(np.float32)

    # Convert to mono
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    return wav


def main():
    yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
    model = hub.load(yamnet_model_handle)
    file_paths = list_audio_files(INPUT_DIR)
    num_files = len(file_paths)
    print(f"[YAMNet] Found {num_files} files")

    # Ensure existing outputs are overwritten on rerun
    if EMBEDDINGS_PATH.exists():
        EMBEDDINGS_PATH.unlink()
    if PATHS_TXT_PATH.exists():
        PATHS_TXT_PATH.unlink()

    # Pre-allocate memmap directly at final path (allow partial results if interrupted)
    print(f"[YAMNet] Allocating memmap: {EMBEDDINGS_PATH} with shape ({num_files}, {EMBEDDING_DIMENSION})")
    embeddings_memmap = np.memmap(
        EMBEDDINGS_PATH,
        dtype='float32',
        mode='w+',
        shape=(num_files, EMBEDDING_DIMENSION),
    )

    # Open paths file directly at final path
    paths_fh = open(PATHS_TXT_PATH, 'w', encoding='utf-8')

    for idx, path in enumerate(file_paths):
        if (idx % 200) == 0:
            print(f"[YAMNet] Processing {idx}/{num_files} ...")

        # Always write the path first to keep alignment consistent
        paths_fh.write(path + "\n")

        wav = read_wav_as_float_mono_16k(path)
        wav_tensor = tf.convert_to_tensor(wav, dtype=tf.float32)
        _, embeddings, _ = model(wav_tensor)
        # embeddings: (num_patches, 1024)
        emb_np = embeddings.numpy()
        file_vec = emb_np.mean(axis=0).astype(np.float32)
        # L2-normalize the 1024-D vector for cosine similarity via dot product
        norm = np.linalg.norm(file_vec) + 1e-12
        file_vec = file_vec / norm

        embeddings_memmap[idx] = file_vec

        # Periodically flush to keep progress on disk
        if (idx % 1000) == 0:
            embeddings_memmap.flush()
            paths_fh.flush()

    # Final flushes
    embeddings_memmap.flush()
    paths_fh.flush()
    paths_fh.close()

    print("[YAMNet] Done.")
    print(f"Embeddings memmap: {EMBEDDINGS_PATH}")
    print(f"Paths list:        {PATHS_TXT_PATH}")


if __name__ == "__main__":
    main()


