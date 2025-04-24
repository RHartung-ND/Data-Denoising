import librosa, os
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

def generate_spectrogram(audio_path, output_path):
    try:
        y, sr = sf.read(audio_path)
        stft_result = librosa.stft(y)
        spectrogram_db = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrogram_db, sr=sr, y_axis='log')
        plt.axis('off')
        # plt.colorbar(format='%+2.0f dB')
        # plt.title('Spectrogram')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches = 0, dpi = 200)
        plt.close()
    except FileNotFoundError:
        print(f"Error: Audio file not found at {audio_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_dir = "sample_data/train-noisy"
    output_dir = "output/spectrographs/train-noisy"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".flac"):
                input_file = os.path.join(root, file)
                generate_spectrogram(input_file, f"{output_dir}/{file[:-5]}.png")