import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import csv
import pandas as pd
import os

###FIX PADNAS
# Load the YAMNet model to get class names
model = hub.load('https://tfhub.dev/google/yamnet/1')

def class_names_from_csv(class_map_csv_text):
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names

class_names = class_names_from_csv(model.class_map_path().numpy())

# Load the saved predictions
OUTPUT_FILE = 'audio_data/unc/yamnet-env/yamnet_predictions.npz'
data = np.load(OUTPUT_FILE, allow_pickle=True)
predictions = data['predictions'].item()

# --- Pandas DataFrame Visualization ---

N = 5  
results = []

for filename, prediction in predictions.items():
    # prediction shape: (num_patches, num_classes)
    # Average over time (patches) to get a single score per class
    mean_scores = prediction.mean(axis=0)
    top_indices = mean_scores.argsort()[-N:][::-1]
    top_scores = mean_scores[top_indices]
    top_classes = [class_names[i] for i in top_indices]
    
    # Build a row: filename, then interleaved class and score
    row = [filename]
    for cls, score in zip(top_classes, top_scores):
        row.extend([cls, score])
    results.append(row)

# Build column names: filename, class_1, score_1, class_2, score_2, ...
columns = ['filename']
for i in range(1, N+1):
    columns.extend([f'class_{i}', f'score_{i}'])

# df = pd.DataFrame(results, columns=columns)
# print(f"We have {len(df)} files predicted in the predictions table.")
# df.to_csv('audio_data/unc/yamnet-env/yamnet_predictions_table.csv', index=False)

if os.path.exists('audio_data/unc/yamnet-env/yamnet_predictions_table.csv'):
    df = pd.read_csv('audio_data/unc/yamnet-env/yamnet_predictions_table.csv')
else:
    df = pd.DataFrame(results, columns=columns)

print(f"We have {len(df)} files")
df.to_csv('audio_data/unc/yamnet-env/yamnet_predictions_table.csv', index=False)


THRESHOLD = 0.96
def filter_high_score_class1(df, threshold=THRESHOLD): #threshold 
    filtered_df = df.loc[df['score_1'] > threshold, ['filename', 'class_1', 'score_1']].copy()
    return filtered_df

# Save filtered high confidence class_1 scores to CSV
filtered_df = filter_high_score_class1(df, threshold=THRESHOLD)
print(f"We have {len(filtered_df)} files with high score class_1 (>{THRESHOLD}).")
filtered_df.to_csv('audio_data/unc/yamnet-env/high_confidence_frames.csv', index=False)

# Find the highest scoring recording for each class directly from the predictions dict
highest_per_class = []
best_scores = [-1] * len(class_names)
best_files = [None] * len(class_names)

for filename, prediction in predictions.items():
    # prediction shape: (num_patches, num_classes)
    mean_scores = prediction.mean(axis=0)
    for class_idx, score in enumerate(mean_scores):
        if score > best_scores[class_idx]:
            best_scores[class_idx] = score
            best_files[class_idx] = filename

for class_idx, class_name in enumerate(class_names):
    if best_files[class_idx] is not None:
        highest_per_class.append({
            'filename': best_files[class_idx],
            'class_name': class_name,
            'score': best_scores[class_idx]
        })

highest_df = pd.DataFrame(highest_per_class)
highest_df = highest_df.sort_values(by='score', ascending=False)
highest_df.to_csv('audio_data/unc/yamnet-env/yamnet_highest_per_class.csv', index=False)
print(f"should be 521 == ({len(highest_df)} classes found)")



