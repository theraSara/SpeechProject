import json
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

import librosa
import numpy as np
import torchaudio

from config import Config


@dataclass
class FewShotSplit:
    """Container for a few-shot experiment split."""
    enrollment: Dict[str, List[str]]
    test_known: Dict[str, List[str]]
    test_unknown: List[str]

    val_enrollment: Dict[str, List[str]]
    val_test_known: Dict[str, List[str]]
    val_test_unknown: List[str]

    k_shot: int
    known_keywords: List[str]
    val_known_keywords: List[str]

    # official episodic JSON splits
    official_mode: bool = False
    official_episode_idx: Optional[int] = None
    official_n_episodes: Optional[int] = None

    # support['_unknown_'] in official episodes
    background_support: List[str] = None


def download_gsc(root: str = "./data"):
    """Download Google Speech Commands v2 using torchaudio."""
    os.makedirs(root, exist_ok=True)
    dataset = torchaudio.datasets.SPEECHCOMMANDS(
        root=root, download=True, subset=None
    )
    print(f"GSC dataset downloaded to {root}")
    return dataset


@lru_cache(maxsize=None)
def _official_split_sets(gsc_root: str) -> Tuple[set, set]:
    """Read official validation/testing lists from GSC."""
    val_path = os.path.join(gsc_root, "validation_list.txt")
    test_path = os.path.join(gsc_root, "testing_list.txt")

    if not os.path.exists(val_path) or not os.path.exists(test_path):
        return set(), set()

    def _read_list(path: str) -> set:
        with open(path, "r", encoding="utf-8") as f:
            return {line.strip().replace("\\", "/") for line in f if line.strip()}

    return _read_list(val_path), _read_list(test_path)


def _keyword_relpaths(gsc_root: str, keyword: str) -> List[str]:
    keyword_dir = os.path.join(gsc_root, keyword)
    if not os.path.isdir(keyword_dir):
        raise FileNotFoundError(f"Keyword directory not found: {keyword_dir}")

    relpaths = [
        f"{keyword}/{fname}"
        for fname in sorted(os.listdir(keyword_dir))
        if fname.endswith(".wav")
    ]
    if not relpaths:
        raise ValueError(f"No .wav files found for keyword '{keyword}'")
    return relpaths


def get_keyword_files(gsc_root: str, keyword: str, subset: str = "all") -> List[str]:
    """Get keyword file paths from official GSC subsets."""
    relpaths = _keyword_relpaths(gsc_root, keyword)
    val_set, test_set = _official_split_sets(gsc_root)

    if subset == "all":
        selected = relpaths
    elif subset == "training":
        selected = [rp for rp in relpaths if rp not in val_set and rp not in test_set]
    elif subset == "validation":
        selected = [rp for rp in relpaths if rp in val_set]
    elif subset == "testing":
        selected = [rp for rp in relpaths if rp in test_set]
    else:
        raise ValueError(f"Unknown subset: {subset}")

    return [os.path.join(gsc_root, rp) for rp in selected]


def _sample_files(rng: random.Random, files: List[str], n: int, desc: str) -> List[str]:
    files = files.copy()
    rng.shuffle(files)
    if len(files) < n:
        raise ValueError(f"Not enough files for {desc}: need {n}, found {len(files)}")
    return files[:n]


def _resolve_audio_path(path: str, gsc_root: str) -> str:
    """Resolve relative paths from shared JSON manifests."""
    if os.path.isabs(path):
        return path

    candidate = os.path.join(gsc_root, path)
    if os.path.exists(candidate):
        return candidate

    if os.path.exists(path):
        return path

    raise FileNotFoundError(f"Could not resolve audio path: {path}")

def _resolve_audio_path_list(paths: List[str], gsc_root: str) -> List[str]:
    return [_resolve_audio_path(p, gsc_root) for p in paths]

def load_few_shot_split_from_manifest(
    config: Config,
    manifest_path: str,
    trial_idx: int = 0
) -> FewShotSplit:
    """
    Load either:
      1. legacy manifest format:
         enrollment, test_known, test_unknown,
         val_enrollment, val_test_known, val_test_unknown,
         known_keywords, val_known_keywords, k_shot
      2. official episodic format:
         config, negative_query, episodes
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # ------------------------------------------------------------
    # Official episodic format
    # ------------------------------------------------------------
    if {"config", "negative_query", "episodes"}.issubset(raw.keys()):
        cfg = raw["config"]
        n_episodes = int(cfg["n_episodes"])
        k_shot = int(cfg["n_support"])

        if trial_idx >= n_episodes:
            raise IndexError(
                f"Requested episode index {trial_idx}, but manifest only has {n_episodes} episodes."
            )

        episode = raw["episodes"][trial_idx]

        # Query keys are the 10 positive classes for this episode
        known_keywords = list(episode["query"].keys())

        enrollment = {
            kw: _resolve_audio_path_list(episode["support"][kw], config.data.gsc_root)
            for kw in known_keywords
        }
        test_known = {
            kw: _resolve_audio_path_list(episode["query"][kw], config.data.gsc_root)
            for kw in known_keywords
        }

        # Global negative pool used both for evaluation and for thresholding at fixed FAR
        negative_query = raw["negative_query"]
        test_unknown = [
            _resolve_audio_path(item["file"], config.data.gsc_root)
            for item in negative_query
        ]

        background_support = []
        if "_unknown_" in episode["support"]:
            background_support = _resolve_audio_path_list(
                episode["support"]["_unknown_"], config.data.gsc_root
            )

        split = FewShotSplit(
            enrollment=enrollment,
            test_known=test_known,
            test_unknown=test_unknown,
            # For the official metric protocol, threshold is selected from the same episode
            # against the shared negative pool to produce ACC+_5% / FRR+_5%.
            val_enrollment=enrollment,
            val_test_known=test_known,
            val_test_unknown=test_unknown,
            k_shot=k_shot,
            known_keywords=known_keywords,
            val_known_keywords=known_keywords,
            official_mode=True,
            official_episode_idx=trial_idx,
            official_n_episodes=n_episodes,
            background_support=background_support,
        )
        return split


    def _resolve_map(items: Dict[str, List[str]]) -> Dict[str, List[str]]:
        return {
            key: [_resolve_audio_path(p, config.data.gsc_root) for p in paths]
            for key, paths in items.items()
        }

    split = FewShotSplit(
        enrollment=_resolve_map(raw["enrollment"]),
        test_known=_resolve_map(raw["test_known"]),
        test_unknown=[_resolve_audio_path(p, config.data.gsc_root) for p in raw["test_unknown"]],
        val_enrollment=_resolve_map(raw["val_enrollment"]),
        val_test_known=_resolve_map(raw["val_test_known"]),
        val_test_unknown=[_resolve_audio_path(p, config.data.gsc_root) for p in raw["val_test_unknown"]],
        k_shot=int(raw.get("k_shot", raw.get("shot", 0))),
        known_keywords=list(raw.get("known_keywords", config.data.known_keywords)),
        val_known_keywords=list(raw.get("val_known_keywords", config.data.val_known_keywords)),
        official_mode=False,
        official_episode_idx=None,
        official_n_episodes=None,
        background_support=[],
    )
    return split

def create_few_shot_split(config: Config, k_shot: int, trial_seed: int) -> FewShotSplit:
    """
    Fallback split builder using official GSC train/val/test lists:
      - enrollment from TRAINING
      - final evaluation from TESTING
      - threshold tuning from VALIDATION
    """
    rng = random.Random(trial_seed)
    data_cfg = config.data

    enrollment: Dict[str, List[str]] = {}
    test_known: Dict[str, List[str]] = {}

    for kw in data_cfg.known_keywords:
        enrollment[kw] = _sample_files(
            rng,
            get_keyword_files(data_cfg.gsc_root, kw, subset="training"),
            k_shot,
            f"enrollment/{kw}",
        )
        test_known[kw] = _sample_files(
            rng,
            get_keyword_files(data_cfg.gsc_root, kw, subset="testing"),
            data_cfg.n_test_per_keyword,
            f"test_known/{kw}",
        )

    test_unknown: List[str] = []
    for kw in data_cfg.unknown_keywords:
        test_unknown.extend(
            _sample_files(
                rng,
                get_keyword_files(data_cfg.gsc_root, kw, subset="testing"),
                data_cfg.n_test_unknown_per_keyword,
                f"test_unknown/{kw}",
            )
        )
    rng.shuffle(test_unknown)

    val_enrollment: Dict[str, List[str]] = {}
    val_test_known: Dict[str, List[str]] = {}
    for kw in data_cfg.val_known_keywords:
        val_enrollment[kw] = _sample_files(
            rng,
            get_keyword_files(data_cfg.gsc_root, kw, subset="training"),
            data_cfg.val_k_shot,
            f"val_enrollment/{kw}",
        )
        val_test_known[kw] = _sample_files(
            rng,
            get_keyword_files(data_cfg.gsc_root, kw, subset="validation"),
            data_cfg.n_val_per_keyword,
            f"val_test_known/{kw}",
        )

    val_test_unknown: List[str] = []
    for kw in data_cfg.val_unknown_keywords:
        val_test_unknown.extend(
            _sample_files(
                rng,
                get_keyword_files(data_cfg.gsc_root, kw, subset="validation"),
                data_cfg.n_val_unknown_per_keyword,
                f"val_test_unknown/{kw}",
            )
        )
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


def get_few_shot_split(config: Config, k_shot: int, trial_idx: int, trial_seed: int) -> FewShotSplit:
    template = config.data.split_manifest_template
    if template:
        manifest_path = template.format(
            trial=trial_idx + 1,
            k_shot=k_shot,
            seed=trial_seed,
        )
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Shared split manifest not found: {manifest_path}")
        return load_few_shot_split_from_manifest(config, manifest_path, trial_idx=trial_idx)

    return create_few_shot_split(config, k_shot, trial_seed)

def load_audio(
    file_path: str,
    target_sr: int = 16000,
    target_length: float = 1.0
) -> np.ndarray:
    """Load audio, resample, pad/trim to target length."""
    waveform, _ = librosa.load(file_path, sr=target_sr, mono=True)
    target_samples = int(target_sr * target_length)

    if len(waveform) < target_samples:
        waveform = np.pad(waveform, (0, target_samples - len(waveform)), mode="constant")
    elif len(waveform) > target_samples:
        waveform = waveform[:target_samples]

    return waveform

def validate_split(split: FewShotSplit) -> None:
    """Sanity-check split integrity."""
    if getattr(split, "official_mode", False):
        if split.official_n_episodes is None:
            raise AssertionError("Official split missing official_n_episodes.")

        if split.official_episode_idx is None:
            raise AssertionError("Official split missing official_episode_idx.")

        for kw in split.known_keywords:
            enroll = set(split.enrollment[kw])
            query = set(split.test_known[kw])

            if len(split.enrollment[kw]) != split.k_shot:
                raise AssertionError(
                    f"Official episode keyword '{kw}' has {len(split.enrollment[kw])} support files, "
                    f"expected {split.k_shot}"
                )

            overlap = enroll & query
            if overlap:
                raise AssertionError(
                    f"Official episode overlap in keyword '{kw}': {len(overlap)} files"
                )

        print(
            f"[Official split] Validation passed for episode "
            f"{split.official_episode_idx + 1}/{split.official_n_episodes}."
        )
        return


    eval_known_classes = set(split.known_keywords)
    val_known_classes = set(split.val_known_keywords)

    overlap_classes = eval_known_classes & val_known_classes
    if overlap_classes:
        raise AssertionError(
            f"Validation known classes overlap with eval known classes: {sorted(overlap_classes)}"
        )

    eval_files = set()
    val_files = set()

    for kw in split.known_keywords:
        enroll = set(split.enrollment[kw])
        test = set(split.test_known[kw])

        overlap = enroll & test
        if overlap:
            raise AssertionError(f"Eval overlap in keyword '{kw}': {len(overlap)} files")

        if split.k_shot != len(split.enrollment[kw]):
            raise AssertionError(
                f"Keyword '{kw}' has {len(split.enrollment[kw])} enrollment files, expected {split.k_shot}"
            )

        eval_files |= enroll
        eval_files |= test

    test_unknown = set(split.test_unknown)
    if eval_files & test_unknown:
        raise AssertionError("Eval unknown overlaps with eval known enrollment/test files")
    eval_files |= test_unknown

    for kw in split.val_known_keywords:
        enroll = set(split.val_enrollment[kw])
        test = set(split.val_test_known[kw])

        overlap = enroll & test
        if overlap:
            raise AssertionError(f"Validation overlap in keyword '{kw}': {len(overlap)} files")

        val_files |= enroll
        val_files |= test

    val_unknown = set(split.val_test_unknown)
    if val_files & val_unknown:
        raise AssertionError("Validation unknown overlaps with validation known files")
    val_files |= val_unknown

    if eval_files & val_files:
        raise AssertionError("Validation files overlap with evaluation files")

    print("[Split] Validation passed.")