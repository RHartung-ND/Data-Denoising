import numpy as np
import librosa, os


def calculate_snr(clean_audio, denoised_audio):
    """
    Calculates the Signal-to-Noise Ratio (SNR) in dB.

    Args:
        clean_audio (numpy.ndarray): The original clean audio signal.
        denoised_audio (numpy.ndarray): The denoised audio signal.

    Returns:
        float: The SNR in dB.
    """

    # Ensure both signals have the same length
    min_len = min(len(clean_audio), len(denoised_audio))
    clean_audio = clean_audio[:min_len]
    denoised_audio = denoised_audio[:min_len]

    # Calculate signal power
    signal_power = np.mean(clean_audio**2)

    # Calculate noise power
    noise = clean_audio - denoised_audio
    noise_power = np.mean(noise**2)

    # Calculate SNR
    if noise_power == 0:  # To avoid division by zero
        return float('inf')  # Or a very large number

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def calculate_mse(clean_audio, denoised_audio):
    """
    Calculates the Mean Squared Error (MSE).

    Args:
        clean_audio (numpy.ndarray): The original clean audio signal.
        denoised_audio (numpy.ndarray): The denoised audio signal.

    Returns:
        float: The MSE.
    """

    # Ensure both signals have the same length
    min_len = min(len(clean_audio), len(denoised_audio))
    clean_audio = clean_audio[:min_len]
    denoised_audio = denoised_audio[:min_len]

    # Calculate MSE
    mse = np.mean((clean_audio - denoised_audio)**2)
    return mse


# Function to load data from directories
def load_data(original_files, denoised_files):
    original_audio_files = []
    denoised_audio_files = []

    for root, _, files in os.walk(original_files):
        for file in files:
            if file.endswith(".flac"):
                input_file = os.path.join(root, file)
                relative_path = os.path.relpath(input_file, original_files)
                output_file = os.path.join(denoised_files, relative_path)

                if os.path.exists(output_file):
                    original_audio_files.append(input_file)
                    denoised_audio_files.append(output_file)

    values = []
    for i in range(len(original_audio_files)):
        try:
            clean_audio, _ = librosa.load(original_audio_files[i], sr=None)
            denoised_audio, _ = librosa.load(denoised_audio_files[i], sr=None)
            snr_value = calculate_snr(clean_audio, denoised_audio)
            mse_value = calculate_mse(clean_audio, denoised_audio)

            values.append((original_audio_files[i], snr_value, mse_value))
        except FileNotFoundError:
            print("Error: Audio files not found. Please provide valid file paths.")
            exit()

    return values

if __name__ == '__main__':
    original_files = None
    denoised_files = None
    try:
        with open("src/config.txt", "r") as config:
            print("----------------------------------------------------------------")
            line = config.readline() # input_dir_clean
            args = line.strip().split("=")
            if len(args) > 1:
                original_files = str(args[1].strip())
                print(f"Using the following testing directory: {original_files}")

            line = config.readline() # noisy_dir

            line = config.readline() # output_dir_denoised
            args = line.strip().split("=")
            if len(args) > 1:
                denoised_files = str(args[1].strip())
                print(f"Using the following training directory: {denoised_files}")
            print("----------------------------------------------------------------")
    except TypeError:
        print("Please use correct data paths or epoch numbers")

    output = load_data(original_files, denoised_files)

    total_snr = 0
    total_mse = 0
    idx = 0
    for file, snr, mse in output:
        print(f"Audio file: {file} had an snr of {snr:.2f} and an mse of {mse:.4f}")
        total_snr += snr
        total_mse += mse
        idx += 1
    
    print(f'\n\nThe {idx} audio files had an average snr of {total_snr/idx:.2f} and an average mse of {total_mse/idx:.4f}')