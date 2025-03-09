import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Parameters
input_dir_clean = "sample_data/test-clean"  # Path to clean audio directory
input_dir_noisy = "sample_data/test-noisy" # Path to noisy audio directory
epochs = 50
batch_size = 32
sampling_rate = 16000
audio_duration = 1  # Seconds
audio_length = sampling_rate * audio_duration

# Function to load and preprocess audio data
def load_and_preprocess_audio(file_path):
    audio, _ = librosa.load(file_path, sr=sampling_rate, duration=audio_duration)
    if len(audio) < audio_length:
        audio = np.pad(audio, (0, audio_length - len(audio)))
    elif len(audio) > audio_length:
        audio = audio[:audio_length]

    return audio.astype(np.float32)

# Function to load data from directories
def load_data(clean_dir, noisy_dir):
    clean_audio_files = []
    noisy_audio_files = []

    for root, _, files in os.walk(clean_dir):
        for file in files:
            if file.endswith(".flac"):
                clean_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(clean_file_path, clean_dir)
                noisy_file_path = os.path.join(noisy_dir, relative_path)

                if os.path.exists(noisy_file_path):
                    clean_audio_files.append(clean_file_path)
                    noisy_audio_files.append(noisy_file_path)

    clean_data = np.array([load_and_preprocess_audio(f) for f in clean_audio_files])
    noisy_data = np.array([load_and_preprocess_audio(f) for f in noisy_audio_files])

    return clean_data, noisy_data

# Load data
clean_data, noisy_data = load_data(input_dir_clean, input_dir_noisy)

# Normalize data
clean_data = (clean_data - np.mean(clean_data)) / np.std(clean_data)
noisy_data = (noisy_data - np.mean(noisy_data)) / np.std(noisy_data)

# Split data into training and validation sets
noisy_train, noisy_val, clean_train, clean_val = train_test_split(noisy_data, clean_data, test_size=0.2, random_state=42)

# Autoencoder Model
def create_autoencoder():
    input_audio = layers.Input(shape=(audio_length,))
    # Encoder
    encoded = layers.Dense(1024, activation='relu')(input_audio)
    encoded = layers.Dense(512, activation='relu')(encoded)
    encoded = layers.Dense(256, activation='relu')(encoded)

    # Decoder
    decoded = layers.Dense(512, activation='relu')(encoded)
    decoded = layers.Dense(1024, activation='relu')(decoded)
    decoded = layers.Dense(audio_length, activation='linear')(decoded)

    autoencoder = models.Model(input_audio, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Create and train the autoencoder
autoencoder = create_autoencoder()
autoencoder.summary()

history = autoencoder.fit(noisy_train, clean_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(noisy_val, clean_val))

# Save the model
autoencoder.save('audio_denoiser.h5')