import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from scipy.io import wavfile
import csv
import scipy.signal


INPUT_FOLDER = '/home/lucaa/audio_data/resampled16' 
OUTPUT_FILE = '/home/lucaa/audio_data/unc/yamnet-env/yamnet_predictions.npz'


model = hub.load('https://tfhub.dev/google/yamnet/1')

# find all .wav files
def find_wav_files(folder):
    wav_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            full_path = os.path.join(root, file)
            wav_files.append(full_path)
    return wav_files

def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) /
                                   original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform

# "https://www.tensorflow.org/hub/tutorials/yamnet"
if __name__ == '__main__':
    wav_files = find_wav_files(INPUT_FOLDER)
    print(f'Found {len(wav_files)} .wav files.')
    predictions = {}
    # embeddings_dict = {}
    # spectrograms_dict = {}

    for full_path in wav_files:
        sr, wav_data = wavfile.read(full_path)
        # Convert to mono if needed
        #if wav_data.ndim > 1:
        #   wav_data = np.mean(wav_data, axis=1)
        # sr, wav_data = ensure_sample_rate(sr, wav_data)
        # Convert to float32 and normalize to [-1, 1]
        wav_data = wav_data.astype(np.float32)
        wav_data = wav_data / np.iinfo(np.int16).max
        # Run model
        scores, embeddings, spectrogram = model(wav_data)
        predictions[full_path] = scores.numpy()
        # embeddings_dict[full_path] = embeddings.numpy()
        # spectrograms_dict[full_path] = spectrogram.numpy()

    np.savez_compressed(
        OUTPUT_FILE,
        predictions=predictions
        # embeddings=embeddings_dict,
        # spectrograms=spectrograms_dict
    )
    print(f'Saved predictions to {OUTPUT_FILE}')
