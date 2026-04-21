from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class AudioConfig:
    """Audio and feature extraction settings."""
    sample_rate: int = 16000
    duration: float = 1.0
    n_mfcc: int = 13
    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 40
    include_delta: bool = True  # 13*3 = 39 features


@dataclass
class DataConfig:
    """Dataset and split settings."""
    gsc_root: str = "./data/SpeechCommands/speech_commands_v0.02"

    # ---- Evaluation keywords (8 known + 8 unknown) ----
    known_keywords: List[str] = field(default_factory=lambda: [
        "yes", "no", "up", "down", "go",
        "stop", "left", "right"
    ])

    unknown_keywords: List[str] = field(default_factory=lambda: [
        "bed", "bird", "cat", "dog",
        "happy", "house", "marvin", "sheila"
    ])

    # ---- Validation keywords (SEPARATE from eval, for threshold tuning) ----
    # These keywords appear NOWHERE in the test evaluation
    val_known_keywords: List[str] = field(default_factory=lambda: [
        "on", "off"
    ])
    val_unknown_keywords: List[str] = field(default_factory=lambda: [
        "tree", "wow"
    ])

    k_shots: List[int] = field(default_factory=lambda: [5, 10])
    n_test_per_keyword: int = 50
    n_test_unknown: int = 200

    # For validation threshold tuning
    val_k_shot: int = 5  # enroll val keywords with this many shots
    n_val_per_keyword: int = 40  # test utterances per val keyword
    n_val_unknown: int = 80

    seed: int = 42


@dataclass
class DTWConfig:
    """DTW baseline settings."""
    distance_metric: str = "cosine"     # Changed: cosine works better in high dims
    aggregation: str = "min"            # Changed: min is more robust than mean
    use_sakoe_chiba_band: bool = True   # Constraint to speed up + regularize
    sakoe_chiba_radius: int = 10        # Band width in frames


@dataclass
class HMMConfig:
    """HMM/GMM baseline settings."""
    n_states: int = 5
    n_mix: int = 1
    covariance_type: str = "diag"
    n_iter: int = 50
    topology: str = "left-right"
    cov_regularization: float = 1e-2
    use_ubm: bool = True
    ubm_n_states: int = 8


@dataclass
class EvalConfig:
    """Evaluation settings."""
    n_trials: int = 5
    output_dir: str = "./results"


@dataclass
class Config:
    """Master config."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    data: DataConfig = field(default_factory=DataConfig)
    dtw: DTWConfig = field(default_factory=DTWConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self):
        os.makedirs(self.eval.output_dir, exist_ok=True)