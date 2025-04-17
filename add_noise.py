import os
import librosa
import soundfile as sf
import numpy as np
import random
from scipy.signal import butter, lfilter

def add_high_frequency_noise(input_file, output_file, noise_level=0.1, frequency_range=(4000, 8000)):
    """
    Adds random high-frequency noise to a FLAC audio file.

    Args:
        input_file (str): Path to the input FLAC file.
        output_file (str): Path to save the noisy FLAC file.
        noise_level (float): Level of noise to add (0.0 to 1.0).
        frequency_range (tuple): Frequency range (low, high) for the noise.
    """
    try:
        audio, sr = librosa.load(input_file, sr=16000)  # Ensure 16 kHz sampling rate

        # Generate high-frequency noise
        noise = np.random.normal(0, 1, len(audio))

        # Filter the noise to the specified frequency range
        nyquist = sr / 2
        low_cut = frequency_range[0] / nyquist
        high_cut = frequency_range[1] / nyquist

        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return b, a

        def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
            b, a = butter_bandpass(lowcut, highcut, fs, order=order)
            y = lfilter(b, a, data)
            return y

        filtered_noise = butter_bandpass_filter(noise, frequency_range[0], frequency_range[1], sr, order=6)

        # Scale the noise
        scaled_noise = filtered_noise * noise_level

        # Add the noise to the audio
        noisy_audio = audio + scaled_noise

        # Clip the audio to avoid distortion
        noisy_audio = np.clip(noisy_audio, -1.0, 1.0)

        # Save the noisy audio
        sf.write(output_file, noisy_audio, sr)

        print(f"Added noise to {input_file} and saved to {output_file}")

    except Exception as e:
        print(f"Error processing {input_file}: {e}")

def process_directory(input_dir, output_dir, noise_level=0.1, frequency_range=(4000, 8000)):
    """
    Processes all FLAC files in a directory, adding noise and saving to a new directory.

    Args:
        input_dir (str): Path to the input directory.
        output_dir (str): Path to the output directory.
        noise_level (float): Level of noise to add.
        frequency_range (tuple): Frequency range for the noise.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".flac"):
                input_file = os.path.join(root, file)
                relative_path = os.path.relpath(input_file, input_dir)
                output_file = os.path.join(output_dir, relative_path)
                output_dir_for_file = os.path.dirname(output_file)

                if not os.path.exists(output_dir_for_file):
                    os.makedirs(output_dir_for_file)

                add_high_frequency_noise(input_file, output_file, noise_level, frequency_range)

if __name__ == "__main__":
    input_directory = "sample_data/train-clean"  # Replace with your input directory
    output_directory = "sample_data/train-noisy"  # Replace with your output directory
    noise_strength = 0.05 # Adjust noise level
    frequency_band = (5000, 7500) # Adjust frequency band

    process_directory(input_directory, output_directory, noise_strength, frequency_band)