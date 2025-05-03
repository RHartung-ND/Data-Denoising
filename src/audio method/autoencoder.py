import os
import librosa
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split

# Parameters
input_dir_clean = "sample_data/test-clean"  # Path to clean audio directory
input_dir_noisy = "sample_data/train-noisy"  # Path to noisy audio directory
epochs = 50
batch_size = 32
sampling_rate = 16000
audio_length = sampling_rate * 1


try:
    with open("src/config.txt", "r") as config:
        print("----------------------------------------------------------------")
        line = config.readline()
        args = line.strip().split("=")
        if len(args) > 1:
            input_dir_clean = str(args[1].strip())
            print(f"Using the following testing directory: {input_dir_clean}")
        
        line = config.readline()
        args = line.strip().split("=")
        if len(args) > 1:
            input_dir_noisy = str(args[1].strip())
            print(f"Using the following training directory: {input_dir_noisy}")

        line = config.readline()

        line = config.readline()
        args = line.strip().split("=")
        if len(args) > 1:
            epochs = int(args[1].strip())
            print(f"Running for {epochs} epochs")
        print("----------------------------------------------------------------")
except TypeError:
    print("Please use correct data paths or epoch numbers")

@tf.keras.utils.register_keras_serializable()
class WeightedMSELossLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(WeightedMSELossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        y_true, y_pred, input_audio = inputs
        mse = tf.keras.losses.MSE(y_true, y_pred)
        weights = 1.0 / (tf.reduce_mean(tf.abs(input_audio), axis=-1) + 1e-6)
        weighted_mse = mse * weights
        self.add_loss(tf.reduce_mean(weighted_mse))
        return y_pred
    
    def get_config(self):
        config = super(WeightedMSELossLayer, self).get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def load_and_window_audio(file_path, augment=False):
    audio, _ = librosa.load(file_path, sr=sampling_rate)
    audio = audio.astype(np.float32)

    windows = []
    for i in range(0, len(audio), audio_length):
        window = audio[i:i + audio_length]
        if len(window) < audio_length:
            window = np.pad(window, (0, audio_length - len(window)))
        if augment:
            scale_factor = random.uniform(0.1, 1.0) #Create quiet audio.
            window = window * scale_factor

        windows.append(window)

    return np.array(windows)

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

    clean_windows = []
    noisy_windows = []

    for clean_file, noisy_file in zip(clean_audio_files, noisy_audio_files):
        clean_windows.extend(load_and_window_audio(clean_file, augment=True)) #Add augmentation to clean.
        noisy_windows.extend(load_and_window_audio(noisy_file, augment=True)) #Add augmentation to noisy.

    return np.array(clean_windows), np.array(noisy_windows)

# Load data
clean_data, noisy_data = load_data(input_dir_clean, input_dir_noisy)

# Normalize data
clean_data = (clean_data - np.mean(clean_data)) / np.std(clean_data)
noisy_data = (noisy_data - np.mean(noisy_data)) / np.std(noisy_data)

# Split data into training and validation sets
noisy_train, noisy_val, clean_train, clean_val = train_test_split(noisy_data, clean_data, test_size=0.2, random_state=42)

# Autoencoder Model
def create_autoencoder():
    input_audio = tf.keras.layers.Input(shape=(audio_length,))
    # Encoder
    encoded = tf.keras.layers.Dense(1024, activation='relu')(input_audio)
    encoded = tf.keras.layers.Dense(512, activation='relu')(encoded)
    encoded = tf.keras.layers.Dense(256, activation='relu')(encoded)

    # Decoder
    decoded_layer = tf.keras.layers.Dense(512, activation='relu')(encoded)
    decoded_layer = tf.keras.layers.Dense(1024, activation='relu')(decoded_layer)
    decoded = tf.keras.layers.Dense(audio_length, activation='linear')(decoded_layer)

    # Custom loss layer
    loss_layer = WeightedMSELossLayer()
    decoded_with_loss = loss_layer([input_audio, decoded, input_audio])

    autoencoder = tf.keras.models.Model(input_audio, decoded_with_loss)
    autoencoder.compile(optimizer='adam')
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
autoencoder.save('src/audio method/audio_denoiser.keras')