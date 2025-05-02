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


## Stage 3: First Update:


To simulate real-world scenarios, a Python script (`add_noise.py`) was developed to add artificial high-frequency noise to the clean audio files. This script generated random noise, filtered it to a specified frequency range (e.g., 4000-8000 Hz), and added it to the clean audio samples. This resulted in a parallel dataset of noisy audio files, which served as the input for the autoencoder.

A crucial aspect of preprocessing involved windowing. Given that the autoencoder was trained on fixed-length audio segments (1 second), a sliding window approach was implemented. This allowed the processing of audio files of arbitrary length. Overlapping windows with a Hann window function were used to minimize phase discontinuities during recombination. Additionally, normalization was performed on the entire audio file before windowing, ensuring consistency across segments.

**Autoencoder Architecture and Training:**

The autoencoder was implemented using TensorFlow and Keras. The model consisted of a dense encoder with three layers (1024, 512, 256 neurons) and a corresponding dense decoder. The model was trained using a custom weighted Mean Squared Error (MSE) loss function, which prioritized errors in quieter sections of the audio. This was achieved by creating a custom Keras Layer that calculated the weighted MSE. Data augmentation was also implemented, scaling down audio samples to create quiet samples, thus helping the model to better handle quiet speech.

The training process involved splitting the data into training and validation sets. The autoencoder was trained for a specified number of epochs, with the validation set used to monitor performance and prevent overfitting.

**Denoising (Decoding):**

A separate Python script (`decoder.py`) was developed to denoise noisy audio files using the trained autoencoder. This script loaded the trained model, preprocessed the noisy audio using the same windowing and normalization techniques as during training, and passed the processed audio through the autoencoder. The denoised audio segments were then recombined using overlap-add, and saved as FLAC files.

**Results and Observations:**

The results obtained so far are promising, demonstrating the feasibility of using a self-supervised autoencoder for high-frequency noise removal. The denoised audio exhibits a noticeable reduction in high-frequency noise, leading to improved clarity. However, further tuning is required to enhance the quality of the denoised audio, especially for quieter speakers.

Specifically, the autoencoder appears to struggle with very quiet speech segments, sometimes cutting them out or introducing artifacts. This is likely due to the global normalization and the model's sensitivity to amplitude differences. The use of a weighted MSE loss and data augmentation has helped to mitigate this issue, but further improvements are needed.

The overlapping window technique with the Hann window has significantly reduced the artifacts associated with segment recombination, minimizing phase discontinuities. However, some residual artifacts are still present, indicating the need for further refinements.

**Technical Challenges and Refinements:**

Throughout the project, several technical challenges were encountered and addressed. One significant hurdle was the handling of variable-length audio files. The initial approach of truncating or padding audio to a fixed length proved inadequate, leading to information loss and poor denoising quality. The implementation of a sliding window approach, with overlapping windows and Hann windowing, effectively resolved this issue. However, the choice of window size and overlap percentage required careful experimentation to balance denoising performance and computational efficiency.

The selection and implementation of the loss function also played a crucial role. The initial MSE loss function, while effective for general reconstruction, did not adequately address the issue of quiet speech segments. The development of a custom weighted MSE loss function, which prioritized errors in quieter sections, significantly improved the model's ability to handle low-amplitude signals. However, further exploration of alternative loss functions, such as log-magnitude loss or perceptual loss functions, could potentially yield even better results.

The model architecture itself presented another area for refinement. The current dense autoencoder, while functional, might not be optimal for capturing the temporal dependencies and spectral characteristics of audio signals. Exploring convolutional neural networks (CNNs) or recurrent neural networks (RNNs) could lead to improved denoising performance. CNNs, with their ability to learn local features and spatial hierarchies, are particularly well-suited for processing spectrogram representations of audio. RNNs, with their memory capabilities, are effective at capturing temporal dependencies in sequential data. Attention mechanisms could also be incorporated to allow the model to focus on specific regions of the audio signal, particularly those containing noise or quiet speech.


**Next Steps:**

The next crucial step is to train the autoencoder on the full LibriSpeech dataset, rather than the significantly reduced subset of data in `test-clean`. This will provide the model with a more diverse training set, leading to improved generalization and robustness.

Further tuning of the model architecture, loss function, and training parameters is also necessary. This includes experimenting with different network architectures (e.g., convolutional or recurrent layers), exploring alternative loss functions (e.g., log-magnitude loss), and adjusting the learning rate and other hyperparameters.

Additionally, a more comprehensive evaluation of the denoised audio is needed, including objective metrics (e.g., Signal-to-Noise Ratio (SNR), Perceptual Evaluation of Speech Quality (PESQ)) and subjective listening tests. This will provide a more accurate assessment of the model's performance and guide further improvements.

**Context and Broader Implications:**

Audio denoising is a critical preprocessing step for ASR systems, especially in noisy environments. By improving the signal-to-noise ratio (SNR) of audio recordings, denoising can enhance the accuracy and robustness of ASR, leading to better speech recognition performance. This has significant implications for various applications, including voice assistants, transcription services, and telecommunications.

The development of a self-supervised autoencoder for audio denoising offers several advantages. Self-supervised learning eliminates the need for paired clean and noisy audio data, which can be difficult and expensive to obtain. Instead, the autoencoder learns to denoise by reconstructing clean audio from noisy inputs. This makes it a highly adaptable and scalable approach.

Furthermore, the use of deep learning techniques, such as autoencoders, allows for the development of highly effective denoising models that can learn complex patterns and adapt to diverse noise conditions. This contrasts with traditional signal processing techniques, which often rely on handcrafted features and assumptions about the noise characteristics.

The LibriSpeech dataset, used in this project, is a widely recognized and valuable resource for ASR research. By training and evaluating the autoencoder on this dataset, the project contributes to the broader research community and provides a benchmark for future work in audio denoising.

**Potential Future Directions:**

Beyond training on the full LibriSpeech dataset and refining the model architecture, several other avenues for future research exist. One potential direction is to explore the use of generative adversarial networks (GANs) for audio denoising. GANs have demonstrated impressive results in image denoising and could potentially be adapted to audio denoising. Another direction is to investigate the use of unsupervised domain adaptation techniques to improve the model's performance on unseen noise conditions.

Additionally, exploring the integration of the denoising autoencoder with an ASR system could provide valuable insights into the impact of denoising on ASR performance. This would involve training an ASR model on the denoised audio and comparing its performance to an ASR model trained on the original noisy audio.

Finally, exploring real-time denoising applications could expand the project's practical impact. This would involve optimizing the autoencoder for low-latency processing and deploying it on embedded devices or cloud platforms.

**Code Organization and Environment:**

The project code is organized into three main Python files:

1.  **`add_noise.py`:** Generates artificial high-frequency noise and adds it to clean audio files.
2.  **`autoencoder.py`:** Implements the autoencoder architecture, training process, and custom loss function.
3.  **`decoder.py`:** Denoises noisy audio files using the trained autoencoder.

To ensure reproducibility, a Conda environment file (`environment.yml`) has been created. This file specifies the required dependencies and their versions, allowing others to easily set up the project environment.

### Instructions on how to setup conda environment:

`conda env create --file=environment.yml`

`conda activate denoiser`



## Stage 4: Second Update:

There are two different methods to test and run the program. The first method is to traing an autoencoder on just the audio. The second method is to first convert the audio to a spectrogram and then train the model off of that. Here is how to run both versions:


### Getting packages ready

1. Make sure that you have Miniconda installed on you computer.

2. Run the following command to generate the conda environment `conda env create --file environment.yml`

3. Run `conda activate denoiser`

### Running the Program.

Run the `spectrogram_method.sh` file. This should output the audio files to `output/spectrographs/temp`. Unfortunately, they aren't great right now. It needs some more fine-tuning and tweaking.

The audio method somehow got broken in between this update and the previous one, so I need to figure out why it's broken and fix it.

### Update on accuracy and training

**Audio Waveform Training:**

The autoencoder, trained directly on the audio waveform data (both clean and artificially noisy), can paritally reduce the random high-frequency noise. The overlapping windowing and Hann window application have effectively mitigated the artifacts associated with segment recombination, leading to a smoother and more coherent output. Furthermore, the implementation of a weighted Mean Squared Error (MSE) loss, prioritizing quieter sections, has shown some success in preserving the integrity of low-amplitude speech.

However, the model still struggles with the complete removal of all instances of random noise. Sporadic bursts or persistent low-level high-frequency noise can still be present in the denoised audio. This suggests that the model, in its current architecture and training configuration, might not be fully capturing the subtle characteristics of the noise or effectively disentangling it from the underlying speech signal in all scenarios. The noise and its interaction with the speech waveform might require a more sophisticated model or a different representation of the audio data.

**Spectrogram-Based Approach:**

To address the limitations of direct waveform training, an alternative method involving the conversion of audio files into spectrograms was explored. Spectrograms provide a visual representation of the frequency content of audio over time, potentially offering a more informative input for the denoising task, particularly for frequency-specific noise like the high-frequency noise targeted in this project.

The initial implementation involved converting the audio files into spectrogram images and then training the autoencoder on these images. The autoencoder architecture was adapted to process image data, typically involving convolutional layers for feature extraction and upsampling layers for reconstruction.

**Challenges with Spectrogram Compression:**

The spectrogram-based approach, unfortunately, did not yield satisfactory results in its initial implementation. A significant challenge encountered was the heavy compression of the spectrogram data during the training process. This compression likely occurred due to the inherent dimensionality reduction within the autoencoder's latent space and the lossy nature of the image representation and processing.

During the encoding phase, the spectrogram images were compressed into a lower-dimensional representation. While this is a standard mechanism in autoencoders for learning essential features, in this context, it appeared to discard crucial details necessary for accurate audio reconstruction. Consequently, when the compressed spectrogram was decoded back into an audio waveform, a significant loss of audio quality and fidelity was observed. The reconstructed audio often sounded muffled, distorted, or contained significant artifacts, failing to preserve the nuances of the original speech.

This heavy compression suggests that the information required to accurately reconstruct the audio waveform from the compressed spectrogram representation was not being adequately retained during the encoding process. The image-based representation and the convolutional layers might be inadvertently discarding information vital for preserving the temporal and fine-grained spectral details necessary for high-quality audio synthesis.

**Next Steps: Preserving Spectrogram Quality and Size:**

Before the final update, the immediate next step is to investigate methods for preserving the quality and size of the spectrograms throughout the autoencoder training process. The goal is to retain sufficient information in the latent space and during the decoding phase to enable high-fidelity audio reconstruction from the denoised spectrogram.

Several potential strategies will be explored:

* **Higher Dimensional Latent Space:** Increasing the dimensionality of the autoencoder's latent space could allow for the retention of more information from the input spectrograms, potentially preserving crucial details for audio reconstruction.
* **Loss Functions Optimized for Reconstruction:** Experimenting with loss functions specifically designed for image reconstruction or those that consider perceptual audio quality metrics might encourage the autoencoder to preserve more relevant information.
* **Less Aggressive Downsampling:** If the autoencoder architecture involves downsampling layers that contribute to information loss, exploring architectures with less aggressive downsampling or alternative methods for dimensionality reduction might be beneficial.
* **Preserving Phase Information:** Standard spectrograms typically represent only the magnitude of the audio signal, discarding phase information which is crucial for accurate audio reconstruction. Investigating methods to incorporate or predict phase information alongside the magnitude spectrogram could significantly improve audio quality. This might involve techniques like Phase Vocoder or specialized neural network architectures for phase estimation.
* **Larger Spectrogram Resolution:** Training on spectrograms with higher temporal and frequency resolution could provide the model with more detailed information to learn from and reconstruct. However, this would also increase the computational cost.

The success of the spectrogram-based approach hinges on the ability to overcome the information loss during the encoding and decoding stages. By focusing on preserving the quality and size of the spectrogram representation, the aim is to create a denoising pipeline that leverages the spectral information effectively while maintaining high audio fidelity upon reconstruction. The results of these investigations will inform the final update on the project's outcomes and the effectiveness of the chosen denoising methods.


### Another AI Disclosure

The initial groundwork for exploring the spectrogram-based audio denoising approach was aided by the utilization of Google Gemini and several online resources. Google Gemini was a useful tool for quickly creating the basic code for audio-to-spectrogram conversion. However, I did use several online tutorials and documentation to give me fundamental knowledge of spectrogram production parameters and best practices in image-based autoencoder design.

I also used Google Gemini for tips and suggestions on improving the efficiency and effectiveness of the code, as well as for brainstorming potential strategies to enhance the training of the dataset.

The core ideas, specific implementations, and analysis of the results are entirely the product of my own thought and experimentation. The AI served as a catalyst for initial exploration and a source of technical suggestions, but the intellectual ownership and the critical evaluation of the project are my own.


## Stage 5: Final Update:

### Getting packages ready

1. Make sure that you have Miniconda installed on you computer.

2. Run the following command to generate the conda environment `conda env create --file environment.yml`

conda env update --file environment.yml --prune

3. Run `conda activate denoiser`

### Running the Program.

To get started, run `python src/add_noise.py`. This python script will go through the training data and randomly add white noise. 

#### Spectrogram Method

1. run `python "src/spectrogram method/create_spectrogram.py"` to generate spectrograms of the training data

2. run `python "src/spectrogram method/picture_encoder.py"` to run the encoder on all of the training data

3. run `python "src/spectrogram method/picture_decoder.py"` to run the decoder on all of the training data

4. run `python "src/spectrogram method/reverse_spectrogram.py"` to convert the spectrogram files back into audio files

This should output the audio files to `output/spectrographs/temp`. Unfortunately, they aren't great right now. It needs some more fine-tuning and tweaking.

#### Audio Only Method


#### Quick Method (only works/tested on Linux)

If you want to run the various Python scripts quickly and are using Linux, then you can run the `src/scripts/automated.sh` file and choose either the spectrogram method or the audio only method by entering either "s" or "a". This script assumes that your Linux shell is using bash.