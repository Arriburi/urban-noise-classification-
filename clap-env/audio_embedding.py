import numpy as np
import laion_clap
import os
import glob

model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt() 

# Find all audio files in resampled48 directory
input_dir = "/home/lucaa/audio_data/resampled48"
audio_pattern = os.path.join(input_dir, "**", "*.wav")
audio_files = glob.glob(audio_pattern, recursive=True)
audio_files = sorted(audio_files)  # Ensure consistent ordering

print(f"Found {len(audio_files)} audio files to process")

batch_size = 8  
all_embeddings = []

for i in range(0, len(audio_files), batch_size):
    batch_files = audio_files[i:i+batch_size]
    print(f"Processing batch {i//batch_size + 1}/{(len(audio_files)-1)//batch_size + 1}")
    
    try:
        batch_embeddings = model.get_audio_embedding_from_filelist(x=batch_files, use_tensor=False)
        all_embeddings.append(batch_embeddings)
    except Exception as e:
        print(f"Stopping program due to error: Error processing batch: {e}")
        exit(1)

# Concatenate all embeddings
if all_embeddings:
    audio_embed = np.vstack(all_embeddings)
    print(f"Final embeddings shape: {audio_embed.shape}")
else:
    print("No embeddings were generated!")
    exit(1)

# Save embeddings
embeddings_path = "clap_embeddings.npy"
np.save(embeddings_path, audio_embed)
print(f"Saved embeddings to: {embeddings_path}")

# Save paths
paths_file = "clap_paths.txt"
with open(paths_file, 'w') as f:
    for path in audio_files:
        f.write(f"{path}\n")
print(f"Saved paths to: {paths_file}")

print(f"Successfully processed {len(audio_files)} audio files")
