# Data-Denoising

## Stage 1: Conceptual Design

### Part 1: Project Overview

This project aims to build a neural network for removing high-frequency noise from audio files for the purpose of Automatic Speech Recognition (ASR). The core concept involves training a self-supervised autoencoder. The autoencoder will learn to encode and then decode the input audio signal.

The project will follow a standard machine learning process, including data acquisition, preprocessing, model development, training, validation, and testing. A crucial aspect will be creating a robust and versatile model.

### Part 2: Proposed Solution

The proposed technique is training a neural network to learn a mapping from noisy to clean audio. An autoencoder design is ideal for this task. The noisy audio will be supplied into the encoder, which will learn to compress the data into a lower-dimensional latent representation. The decoder will then use this latent representation to attempt to recover the original, clean audio.

The key idea is that the network will prioritize learning the audio signal's dominant, important elements when attempting to recreate it. High-frequency noise, which is less organized and connected with the underlying signal, will be more difficult for the network to precisely recreate. As a result, the decoder's output will be closer to the clean signal, thus reducing the input noise.

For the encoder and decoder, a variety of neural network architectures can be considered, including Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and a hybrid approach. CNNs are good at detecting local patterns in audio signals, whereas RNNs excel at modeling temporal dependencies. Experimentation and analysis of auditory data will determine the architecture.

### Part 3: Dataset Acquisition and Preparation

Finding an appropriate dataset is an essential component of this endeavor. The dataset should ideally include pairs of noisy and clean audio. A feasible method is to begin with a dataset of clean audio recordings and then artificially introduce high-frequency noise to create the noisy equivalents, although acquiring ideally paired data can be difficult.

#### Data Sources:

1. **Clean Audio:**  Publicly available audio datasets will be explored. Potential sources include:
    * **[LibriSpeech](https://www.openslr.org/12/):** A corpus of English speech read aloud.
    * **[UrbanSound8K](https://www.kaggle.com/datasets/chrisfilo/urbansound8k):** A dataset of urban sounds.
    * **[VoxPopuli](https://github.com/facebookresearch/voxpopulic)**: Public GitHub repository containing a large-scale multilingual speech corpus for representation learning, semi-supervised learning, and interpretation.
    * **[AudioSet](https://research.google.com/audioset/)**: Publicly available ontology of 632 audio event classes and a collection of 2,084,320 human-labeled 10-second sound clips drawn from YouTube videos.

2. **Noise:** High-frequency noise can be generated synthetically. Different types of noise can be considered, such as white noise, pink noise, or recordings of real-world high-frequency interference. The characteristics of the added noise (frequency range, amplitude) will be varied to create a more robust model.

3. If more data is needed, there is a great resource for finding more audio [here](https://github.com/jim-schwoebel/voice_datasets).
#### Dataset Split:

The dataset will be divided into three subsets:

1. **Training Set (60%):**  Used to train the neural network's parameters.
2. **Validation Set (20%):**  Used to monitor the model's performance during training and tune hyperparameters. This helps prevent overfitting to the training data.
3. **Test Set (20%):**  A held-out set used for final evaluation of the trained model's performance on unseen data. This set will only be used after the model is fully trained and validated.

#### Data Preprocessing:

1. **Audio Representation:** The audio signals will be converted into a suitable representation for neural networks. Common techniques include:
    * **Waveform:** Directly using the raw audio samples.
    * **Spectrogram:** Representing the audio as a time-frequency image. This can be generated using Short-Time Fourier Transform (STFT). Spectrograms are often more suitable for CNNs.
    * **Mel-frequency cepstral coefficients (MFCCs):**  Features that are perceptually relevant to human hearing.

2. **Normalization:** The audio data will be normalized to a standard range (e.g., between 0 and 1) to improve the training process.

3. **Noise Addition:** High-frequency noise will be added to the clean audio signals in the training and validation sets. The noise level will be varied to create a more robust model.

### Part 4: Evaluation Metrics

The performance of the denoising model will be evaluated using several metrics:

1. **Signal-to-Noise Ratio (SNR):**  Measures the ratio of the desired signal power to the noise power. A higher SNR indicates better denoising performance.
2. **Mean Squared Error (MSE):**  Measures the average squared difference between the clean audio and the denoised audio. A lower MSE indicates better performance.
3. **Perceptual Evaluation of Audio Quality (PEAQ):**  A more sophisticated metric that takes into account human perception of audio quality.

### Part 5: Tools and Technologies

* **Programming Language:** Python
* **Deep Learning Framework:** TensorFlow or PyTorch
* **Audio Processing Libraries:** Librosa, PyDub
* **Data Visualization:** Matplotlib, Seaborn


### Part 6: Generative AI Disclosure

I did use Google Gemini to explore different high-level solutions, suggesting potential datasets, and outlining the project timeline. Specifically, Gemini contributed to the initial brainstorming of the autoencoder approach, the suggestion of using spectrograms and MFCCs for audio representation, and the outline of the data preprocessing steps. The generated text was used as inspiration and then re-written and expanded upon to create this document. The core ideas and the final structure of the project are my own.



## Stage 2: Datasets:

**Source and Relevance:**

The [LibriSpeech dataset](https://www.openslr.org/12/) originates from audiobooks read from the LibriVox project, a collection of public domain audiobooks read by volunteers. It was prepared and released by Vassil Panayotov, Georgi Tusev, Daniel Povey, and Sanjeev Khudanpur. The dataset is publicly hosted and distributed through the OpenSLR website. For this project, the clean samples will be used as the target of the autoencoder, and the other samples will be used to test the models generalizability.

**Subsets and Important Differences (Focus on Noise):**

LibriSpeech is divided into several subsets, primarily for training, validation, and testing purposes.

* **Train:**
    * `train-clean-100`: 100 hours of "clean" speech, considered relatively noise-free. This subset will primarily provide the clean target data for the autoencoder. Although labeled clean, it still contains minor imperfections, that will improve the robustness of the model.
    * `train-clean-360`: 360 hours of clean speech. This subset will also provide clean target data, and increase the diversity of the training data.
    * `train-other-500`: 500 hours of speech from other LibriVox readers, which may include more challenging acoustic conditions, and therefore more high frequency noise. This subset will be used for testing the generalizability of the model.
* **Validation:**
    * `dev-clean`: Validation set with clean speech. This set will be used to validate the performance of the model during training.
    * `dev-other`: Validation set with speech from other readers. This set will be used to validate the generalizability of the model during training.
* **Test:**
    * `test-clean`: Test set with clean speech. This set will be used to test the final performance of the model.
    * `test-other`: Test set with speech from other readers. This set will be used to test the final generalizability of the model.

The key difference between the "clean" and "other" subsets lies in the acoustic quality and variability, especially concerning high-frequency noise. "Clean" subsets are curated to minimize noise and reverberation, providing a relatively consistent training or evaluation environment. "Other" subsets, in contrast, contain a wider range of acoustic conditions, including varying levels of background noise, reverberation, and speaker accents, and therefore, more high frequency noise.

The LibriSpeech dataset contains audio from over 2,400 speakers. This allows for an increase in the variability of the training data, which helps the autoencoder learn a more general representation of clean speech.

**Sample Characteristics (Focus on High-Frequency Content):**

LibriSpeech audio samples are characterized by the following, with emphasis on aspects relevant to high-frequency noise:

* **Resolution:** 16-bit PCM (Pulse Code Modulation).
* **Sensors:** The audio was recorded using various microphones, depending on the LibriVox volunteers. This variability introduces variations in the high-frequency content, which can be seen as a form of natural noise.
* **Ambient Conditions:** The "clean" subsets were intended to be recorded in quiet environments, but the "other" subsets contain a wider range of ambient conditions, including background noise and reverberation, and therefore more high frequency noise.
* **Sampling Frequency:** 16 kHz. This is critical as it limits the highest representable frequency to 8 kHz (Nyquist frequency), defining the range of high-frequency noise.
* **Audio Format:** FLAC (Free Lossless Audio Codec).
* **Transcription:** Each audio file is accompanied by a corresponding text transcription.
* **Artificial Noise Addition:** For training, the "clean" data will have high-frequency noises artificial added by:
    * Generating white noise and filtering it to the desired high-frequency range.
    * Simulating specific types of high-frequency noise (e.g., electrical interference).
    * Adding noise from other audio sources that contain high frequency noise.
* **Preprocessing:** Before feeding the audio into the autoencoder, some preprocessing steps will need to be performed. Such as:
    * Normalization.
    * Potentially, short-time Fourier transform (STFT) to analyze the frequency content.
    * Windowing the audio.

**Data Sample Illustration:**

It might be beneficial to visualize the frequency content of the audio. This could be done by generating spectrograms of the audio samples.

**Example Spectrogram Visualization:**

1.  **Select a clean audio sample:** e.g., `229-130880-0018.flac` from `train-clean-100`.
2.  **Compute the STFT:** Use scipy.signal to compute the STFT of the audio signal.
3.  **Generate a spectrogram:** Plot the magnitude of the STFT as a function of time and frequency.
4.  **Repeat with a sample from "train-other-500":** Observe the differences in the spectrogram, particularly in the high-frequency regions.
5.  **Add artificial noise:** Add chosen type of high-frequency noise to the clean sample, and generate a new spectrogram. This will be an example of the input to the autoencoder.
6.  Display all three spectrograms side by side. This will give a good visual representation of the data that the autoencoder will be processing.


## Stage 3: Creating and running the autoencoder:

Creating the conda environment:

`conda env create --file=environment.yml`

`conda activate denoiser`