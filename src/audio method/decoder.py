import os
import librosa
import numpy as np
import scipy.signal
import tensorflow as tf
import soundfile as sf
import scipy
from autoencoder import WeightedMSELossLayer

# Parameters
input_dir_noisy = None  # Path to noisy audio directory
output_dir_denoised = None  # Path to saved denoised audio
model_path = "src/audio method/audio_denoiser.keras"  # Path to trained autoencoder model
sampling_rate = 16000
audio_duration = 1  # Seconds
audio_length = sampling_rate * audio_duration
overlap = 0.5  # 50% overlap

try:
    with open("src/config.txt", "r") as config:
        print("----------------------------------------------------------------")
        line = config.readline() # input_dir_clean

        line = config.readline() # noisy_dir
        args = line.strip().split("=")
        if len(args) > 1:
            input_dir_noisy = str(args[1].strip())
            print(f"Using the following noisy directory: {input_dir_noisy}")
        
        line = config.readline() # testing_dir

        line = config.readline() # output_dir_denoised
        args = line.strip().split("=")
        if len(args) > 1:
            output_dir_denoised = str(args[1].strip())
            print(f"Output to the denoised directory: {output_dir_denoised}")
        print("----------------------------------------------------------------")
except TypeError:
    print("Please use correct data paths or epoch numbers")


def denoise_and_save_audio(model, input_file, output_file):
    try:
        noisy_audio, sr = librosa.load(input_file, sr=sampling_rate)
        noisy_audio = noisy_audio.astype(np.float32)

        # Normalize the entire audio
        noisy_audio = (noisy_audio - np.mean(noisy_audio)) / np.std(noisy_audio)

        hop_length = int(audio_length * (1 - overlap))
        window = scipy.signal.windows.hann(audio_length)
        windowed_audio = []
        for i in range(0, len(noisy_audio) - audio_length, hop_length):
            segment = noisy_audio[i:i + audio_length] * window
            #Local Normalization
            segment = (segment - np.mean(segment))/ np.std(segment)
            windowed_audio.append(segment)
        windowed_audio = np.array(windowed_audio)

        # Denoise
        denoised_windowed_audio = model.predict(windowed_audio)

        # Overlap-Add Recombination
        denoised_audio = np.zeros(len(noisy_audio))
        for i, segment in enumerate(denoised_windowed_audio):
            start = i * hop_length
            denoised_audio[start:start + audio_length] += segment * window

        # Reverse normalization (global)
        denoised_audio = (denoised_audio * np.std(denoised_audio)) + np.mean(denoised_audio)

        # Clip the audio to avoid distortion
        denoised_audio = np.clip(denoised_audio, -1.0, 1.0)

        sf.write(output_file, denoised_audio, sampling_rate)
        print(f"Denoised {input_file} and saved to {output_file}")

    except Exception as e:
        print(f"Error processing {input_file}: {e}")

# Load the trained autoencoder model
with tf.keras.utils.custom_object_scope({'WeightedMSELossLayer': WeightedMSELossLayer}):
    autoencoder = tf.keras.models.load_model(model_path)

# Load the trained autoencoder model
autoencoder = tf.keras.models.load_model(model_path)

# Process all noisy files in the input directory
if not os.path.exists(output_dir_denoised):
    os.makedirs(output_dir_denoised)

for root, _, files in os.walk(input_dir_noisy):
    for file in files:
        if file.endswith(".flac"):
            input_file = os.path.join(root, file)
            relative_path = os.path.relpath(input_file, input_dir_noisy)
            output_file = os.path.join(output_dir_denoised, relative_path)
            output_dir_for_file = os.path.dirname(output_file)

            if not os.path.exists(output_dir_for_file):
                os.makedirs(output_dir_for_file)

            denoise_and_save_audio(autoencoder, input_file, output_file)