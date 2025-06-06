import librosa
import os
import numpy as np
from PIL import Image
import soundfile as sf

def spectrogram_image_to_audio(png_path, output_wav_path, sr=22050):
    # Load image and convert to grayscale values (0–255)
    img = Image.open(png_path).convert('L')
    img_data = np.array(img).astype(np.float32) / 255.0  # normalize to [0,1]

    # Convert image back to log scale (assuming that's how you made it)
    log_spectrogram = img_data * 80.0 - 80.0  # Rescale assuming range [-80, 0] dB

    # Convert log scale to linear magnitude
    spectrogram = librosa.db_to_amplitude(log_spectrogram)

    # Use Griffin-Lim algorithm to reconstruct the waveform
    audio = librosa.griffinlim(spectrogram, n_iter=60)

    # Save as WAV
    sf.write(output_wav_path, audio, sr)
    print(f"Saved reconstructed audio to {output_wav_path}")

if __name__ == "__main__":
    input_dir = None
    output_dir = None


    try:
        with open("src/config.txt", "r") as config:
            print("----------------------------------------------------------------")
            line = config.readline() # input_dir_clean
            line = config.readline() # input_dir_clean
            line = config.readline() # input_dir_clean
            line = config.readline() # input_dir_clean
            line = config.readline() # input_dir_clean
            # -------------------------------------------------
            line = config.readline() # input_dir_clean
            line = config.readline() # noisy_dir
            args = line.strip().split("=")
            if len(args) > 1:
                input_dir = str(args[1].strip())
                print(f"Using the following training directory: {input_dir}")
            line = config.readline() # input_dir_clean
            line = config.readline() # input_dir_clean
            args = line.strip().split("=")
            if len(args) > 1:
                output_dir = str(args[1].strip())
                print(f"Using the following training directory: {output_dir}")
            print("----------------------------------------------------------------")
    except TypeError:
        print("Please use correct data paths or epoch numbers")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".png"):
                input_file = os.path.join(root, file)
                spectrogram_image_to_audio(input_file, f"{output_dir}/{file[:-4]}.wav")