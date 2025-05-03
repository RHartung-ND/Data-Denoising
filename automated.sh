#!/bin/bash
echo "Do you want to run the 'audio only' method or the 'spectrogram' method?"
echo "Enter 'a' for audio, 's' for spectrogram, or 'e' to exit:"
read -r choice

__conda_setup="$("$(which conda)" 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"

case "$choice" in
    a|A)
        echo "You chose audio."
        if conda env list | grep "audio_denoiser"
        then
            echo "already created audio_denoiser environment"
        else
            echo "creating audio_denoiser environment"
            conda env create --file=audio_environment.yml
        fi

        conda activate audio_denoiser

        python "src/add_noise.py"

        python "src/audio method/autoencoder.py"
        python "src/audio method/decoder.py"

        python "src/calculate_metrics.py"
        ;;
    s|S)
        echo "You chose spectrogram."
        if conda env list | grep "name: sepctrogram_denoiser"
        then
            echo "already created name: sepctrogram_denoiser environment"
        else
            echo "creating name: sepctrogram_denoiser environment"
            conda env create --file=spec_environment.yml
        fi

        conda activate sepctrogram_denoiser

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
        ;;
esac