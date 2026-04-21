import os
import random
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from config import Config
import torchaudio


@dataclass
class FewShotSplit:
    """Container for a few-shot experiment split."""
    # Enrollment: keyword -> list of audio file paths
    enrollment: Dict[str, List[str]]
    # Test known: keyword -> list of audio file paths
    test_known: Dict[str, List[str]]
    # Test unknown: list of audio file paths (from unknown keyword classes)
    test_unknown: List[str]

    # ---- Validation split (SEPARATE keywords, for threshold tuning) ----
    # Val enrollment: val_keyword -> list of file paths (enrolled separately)
    val_enrollment: Dict[str, List[str]]
    # Val test known: val_keyword -> list of file paths
    val_test_known: Dict[str, List[str]]
    # Val test unknown: list of file paths
    val_test_unknown: List[str]

    # Metadata
    k_shot: int
    known_keywords: List[str]
    val_known_keywords: List[str]


def download_gsc(root: str = "./data"):
    """Download Google Speech Commands v2 using torchaudio."""
    os.makedirs(root, exist_ok=True)
    dataset = torchaudio.datasets.SPEECHCOMMANDS(
        root=root, download=True, subset=None
    )
    print(f"GSC dataset downloaded to {root}")
    return dataset


def get_keyword_files(gsc_root: str, keyword: str) -> List[str]:
    """Get all .wav file paths for a given keyword."""
    keyword_dir = os.path.join(gsc_root, keyword)
    if not os.path.isdir(keyword_dir):
        raise FileNotFoundError(f"Keyword directory not found: {keyword_dir}")
    files = [
        os.path.join(keyword_dir, f)
        for f in sorted(os.listdir(keyword_dir))
        if f.endswith(".wav")
    ]
    return files


def create_few_shot_split(
    config: Config,
    k_shot: int,
    trial_seed: int
) -> FewShotSplit:
    """
    Create one few-shot split for evaluation.
    
    IMPORTANT: Validation uses SEPARATE keywords with their own enrollment,
    so threshold tuning sees real known-vs-unknown separation.
    """
    rng = random.Random(trial_seed)
    data_cfg = config.data

    # ============================================================
    # MAIN EVALUATION SPLIT (test keywords)
    # ============================================================
    enrollment = {}
    test_known = {}

    for kw in data_cfg.known_keywords:
        all_files = get_keyword_files(data_cfg.gsc_root, kw)
        rng.shuffle(all_files)
        needed = k_shot + data_cfg.n_test_per_keyword
        if len(all_files) < needed:
            print(f"Warning: {kw} has {len(all_files)} files, need {needed}")
            enrollment[kw] = all_files[:k_shot]
            test_known[kw] = all_files[k_shot:]
        else:
            enrollment[kw] = all_files[:k_shot]
            test_known[kw] = all_files[k_shot:k_shot + data_cfg.n_test_per_keyword]

    # Unknown test utterances
    test_unknown = []
    per_class = data_cfg.n_test_unknown // len(data_cfg.unknown_keywords)
    for kw in data_cfg.unknown_keywords:
        all_files = get_keyword_files(data_cfg.gsc_root, kw)
        rng.shuffle(all_files)
        test_unknown.extend(all_files[:per_class])
    rng.shuffle(test_unknown)

    # ============================================================
    # VALIDATION SPLIT (separate keywords for threshold tuning)
    # ============================================================
    val_enrollment = {}
    val_test_known = {}
    val_k = data_cfg.val_k_shot

    for kw in data_cfg.val_known_keywords:
        all_files = get_keyword_files(data_cfg.gsc_root, kw)
        rng.shuffle(all_files)
        # Enroll val keywords just like test keywords
        val_enrollment[kw] = all_files[:val_k]
        val_test_known[kw] = all_files[val_k:val_k + data_cfg.n_val_per_keyword]

    val_test_unknown = []
    val_per_class = data_cfg.n_val_unknown // len(data_cfg.val_unknown_keywords)
    for kw in data_cfg.val_unknown_keywords:
        all_files = get_keyword_files(data_cfg.gsc_root, kw)
        rng.shuffle(all_files)
        val_test_unknown.extend(all_files[:val_per_class])
    rng.shuffle(val_test_unknown)

    return FewShotSplit(
        enrollment=enrollment,
        test_known=test_known,
        test_unknown=test_unknown,
        val_enrollment=val_enrollment,
        val_test_known=val_test_known,
        val_test_unknown=val_test_unknown,
        k_shot=k_shot,
        known_keywords=list(data_cfg.known_keywords),
        val_known_keywords=list(data_cfg.val_known_keywords),
    )


def load_audio(file_path: str, target_sr: int = 16000,
               target_length: float = 1.0) -> np.ndarray:
    """Load audio, resample, pad/trim to target length."""
    waveform, sr = librosa.load(file_path, sr=target_sr, mono=True)
    target_samples = int(target_sr * target_length)
    if len(waveform) < target_samples:
        waveform = np.pad(waveform, (0, target_samples - len(waveform)),
                          mode='constant')
    elif len(waveform) > target_samples:
        waveform = waveform[:target_samples]
    return waveform