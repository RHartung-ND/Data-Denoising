#!/bin/bash
if [ conda list --name denoiser ]
then
    echo "already created denoiser environment"
else
    echo "creating denoiser environment"
    conda env create --file=environment.yml
fi

conda activate denoiser

python src/add_noise.py


python "src/spectrogram method/create_spectrogram.py"
python "src/spectrogram method/picture_encoder.py"
python "src/spectrogram method/picture_decoder.py"
python "src/spectrogram method/reverse_spectrogram.py"