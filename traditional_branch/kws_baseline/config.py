from dataclasses import dataclass, field
from typing import List, Optional
import os

KNOWN_GSC = [
    "yes", "no", "up", "down", "left",
    "right", "on", "off", "stop", "go",
]

UNKNOWN_GSC = [
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine",
    "bed", "bird", "cat", "dog", "happy",
    "house", "marvin", "sheila", "tree", "wow",
]

# Held-out GSC words for threshold tuning only
VAL_KNOWN_GSC = ["backward", "forward"]
VAL_UNKNOWN_GSC = ["follow", "learn", "visual"]


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
    include_delta: bool = True  # 13 * 3 = 39 dims


@dataclass
class DataConfig:
    """Dataset and split settings."""
    gsc_root: str = "./data/SpeechCommands/speech_commands_v0.02"

    split_manifest_template: Optional[str] = None

    known_keywords: List[str] = field(default_factory=lambda: KNOWN_GSC.copy())
    unknown_keywords: List[str] = field(default_factory=lambda: UNKNOWN_GSC.copy())

    # Validation classes are disjoint from final eval classes
    val_known_keywords: List[str] = field(default_factory=lambda: VAL_KNOWN_GSC.copy())
    val_unknown_keywords: List[str] = field(default_factory=lambda: VAL_UNKNOWN_GSC.copy())

    k_shots: List[int] = field(default_factory=lambda: [5, 10])

    # Per-class counts
    n_test_per_keyword: int = 50
    n_test_unknown_per_keyword: int = 50

    # Validation for threshold tuning
    val_k_shot: int = 5
    n_val_per_keyword: int = 40
    n_val_unknown_per_keyword: int = 40

    seed: int = 42


@dataclass
class DTWConfig:
    """DTW baseline settings."""
    distance_metric: str = "cosine"
    aggregation: str = "min"
    use_sakoe_chiba_band: bool = True
    sakoe_chiba_radius: int = 10


@dataclass
class HMMConfig:
    """HMM/GMM baseline settings."""
    n_states: int = 5
    n_mix: int = 2             
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
    # tune threshold to 5% FAR on validation
    target_far: float = 0.05    
    output_dir: str = "./results"
    save_per_trial: bool = True
    official_metrics_mode: bool = True


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

        eval_classes = set(self.data.known_keywords) | set(self.data.unknown_keywords)
        val_classes = set(self.data.val_known_keywords) | set(self.data.val_unknown_keywords)
        overlap = eval_classes & val_classes
        if overlap:
            raise ValueError(
                f"Validation keywords must be disjoint from evaluation keywords: {sorted(overlap)}"
            )