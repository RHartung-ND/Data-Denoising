import os
import librosa
import numpy as np
import tensorflow as tf
import soundfile as sf

# Parameters
input_dir_noisy = "sample_data/test-noisy"  # Path to noisy audio directory
output_dir_denoised = "sample_data/test-cleaned"  # Path to save denoised audio
model_path = "audio_denoiser.keras"  # Path to your trained autoencoder model
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

# Function to denoise and save audio
def denoise_and_save_audio(model, input_file, output_file):
    try:
        noisy_audio = load_and_preprocess_audio(input_file)
        noisy_audio = (noisy_audio - np.mean(noisy_audio)) / np.std(noisy_audio) #Normalize the input in the same way the model was trained.
        noisy_audio = np.expand_dims(noisy_audio, axis=0)  # Add batch dimension

        denoised_audio = model.predict(noisy_audio)
        denoised_audio = denoised_audio.squeeze() #Remove batch dimension.

        #Reverse normalization
        denoised_audio = (denoised_audio * np.std(denoised_audio)) + np.mean(denoised_audio)

        # Clip the audio to avoid distortion
        denoised_audio = np.clip(denoised_audio, -1.0, 1.0)

        sf.write(output_file, denoised_audio, sampling_rate)
        print(f"Denoised {input_file} and saved to {output_file}")

    except Exception as e:
        print(f"Error processing {input_file}: {e}")

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