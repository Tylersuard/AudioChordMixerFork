# AudioChordMixer


This is an adaptation of ChordMixer by Ruslan Khalitov, et. al. See ChordMixer: A Scalable Neural Attention Model For Sequences With Different Lengths [Accepted to ICLR'23]

[OpenReview](https://openreview.net/forum?id=E8mzu3JbdR)


This model is designed to take in audio files as a sequence of audio samples, rather than a spectrogram image as is customary with audio processing.

The dataset comes from HuggingFace's speech_commands v0.01 dataset (which is thousands of one-second audio recordings of 30 different speech commands), and has been pared down to only 3 different classes.

The examples vary in length, but all are around 16,000 samples long.  

I tokenized each example and then trained the ChordMixer network on them, and it is now able to classify those examples with a 77% accuracy on the validation set.  That is not an especially high accuracy, but it proves that audio files can be understood as a sequence by a neural network rather than as a spectrogram, which is the prevailing method at the time of this repository's release.

To run in Google Colab, just open Google Colab, go to file --> open, then open from github, and paste in the URL of this repo.
