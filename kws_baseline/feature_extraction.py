import numpy as np
import librosa
from typing import List
from config import AudioConfig


def extract_mfcc(audio: np.ndarray, config: AudioConfig) -> np.ndarray:
    """
    Extract MFCC features from audio waveform.
    Returns: (num_frames, n_features)
    """
    mfcc = librosa.feature.mfcc(
        y=audio, sr=config.sample_rate,
        n_mfcc=config.n_mfcc, n_fft=config.n_fft,
        hop_length=config.hop_length, win_length=config.win_length,
        n_mels=config.n_mels
    )

    if config.include_delta:
        delta = librosa.feature.delta(mfcc, order=1)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.vstack([mfcc, delta, delta2])
    else:
        features = mfcc

    features = features.T  # (num_frames, n_features)

    # Cepstral Mean and Variance Normalization (CMVN)
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True) + 1e-8
    features = (features - mean) / std

    return features


def extract_features_batch(
    file_paths: List[str],
    audio_config: AudioConfig
) -> List[np.ndarray]:
    """Extract features from a list of audio files."""
    from data_loader import load_audio
    features_list = []
    for fp in file_paths:
        audio = load_audio(fp, target_sr=audio_config.sample_rate,
                           target_length=audio_config.duration)
        feat = extract_mfcc(audio, audio_config)
        features_list.append(feat)
    return features_list