#!/bin/bash
echo "Do you want to run the 'audio only' method or the 'spectrogram' method?"
echo "Enter 'a' for audio, 's' for spectrogram, or 'e' to exit:"
read -r choice

case "$choice" in
    a|A)
        echo "You chose audio."
        if conda env list | grep "denoiser"
        then
            echo "already created denoiser environment"
        else
            echo "creating denoiser environment"
            conda env create --file=environment.yml
        fi
        ;;
    s|S)
        echo "You chose spectrogram."
        if conda env list | grep "denoiser"
        then
            echo "already created denoiser environment"
        else
            echo "creating denoiser environment"
            conda env create --file=environment.yml
        fi

        conda activate denoiser

        python "src/add_noise.py"

        python "src/spectrogram method/create_spectrogram.py"
        python "src/spectrogram method/picture_encoder.py"
        python "src/spectrogram method/picture_decoder.py"
        python "src/spectrogram method/reverse_spectrogram.py"
        ;;
    e|E)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Please enter 'a' or 's'."
        read -r choice
        ;;
esac