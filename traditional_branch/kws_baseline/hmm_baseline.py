import copy
import time
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from hmmlearn import hmm
from tqdm import tqdm

from config import Config, HMMConfig
from data_loader import FewShotSplit
from evaluation import find_threshold_for_target_far
from feature_extraction import extract_features_batch

HMMModel = Union[hmm.GaussianHMM, hmm.GMMHMM]


def build_left_right_hmm(
    n_states: int,
    n_features: int,
    hmm_config: HMMConfig
) -> HMMModel:
    """Build a left-to-right HMM or GMM-HMM."""
    use_gmm = hmm_config.n_mix > 1

    if use_gmm:
        model = hmm.GMMHMM(
            n_components=n_states,
            n_mix=hmm_config.n_mix,
            covariance_type=hmm_config.covariance_type,
            n_iter=hmm_config.n_iter,
            verbose=False,
            params="mcw",
            init_params="",
        )
    else:
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=hmm_config.covariance_type,
            n_iter=hmm_config.n_iter,
            verbose=False,
            params="mc",
            init_params="",
        )

    startprob = np.zeros(n_states)
    startprob[0] = 1.0
    model.startprob_ = startprob

    transmat = np.zeros((n_states, n_states))
    for i in range(n_states - 1):
        transmat[i, i] = 0.7
        transmat[i, i + 1] = 0.3
    transmat[-1, -1] = 1.0
    model.transmat_ = transmat

    if use_gmm:
        model.weights_ = np.full((n_states, hmm_config.n_mix), 1.0 / hmm_config.n_mix)
        model.means_ = np.zeros((n_states, hmm_config.n_mix, n_features))
        if hmm_config.covariance_type == "diag":
            model.covars_ = np.ones((n_states, hmm_config.n_mix, n_features))
        elif hmm_config.covariance_type == "full":
            model.covars_ = np.tile(
                np.eye(n_features)[None, None, :, :],
                (n_states, hmm_config.n_mix, 1, 1),
            )
        else:
            raise ValueError(f"Unsupported covariance type: {hmm_config.covariance_type}")
    else:
        model.means_ = np.zeros((n_states, n_features))
        if hmm_config.covariance_type == "diag":
            model.covars_ = np.ones((n_states, n_features))
        elif hmm_config.covariance_type == "full":
            model.covars_ = np.tile(np.eye(n_features)[None, :, :], (n_states, 1, 1))
        else:
            raise ValueError(f"Unsupported covariance type: {hmm_config.covariance_type}")

    return model


def initialize_hmm_with_data(
    model: HMMModel,
    features_list: List[np.ndarray],
    hmm_config: HMMConfig
) -> HMMModel:
    """Initialize emissions from uniform segmentation."""
    n_states = model.n_components
    n_features = features_list[0].shape[1]

    state_frames = [[] for _ in range(n_states)]
    for feat in features_list:
        n_frames = len(feat)
        frames_per_state = max(1, n_frames // n_states)
        for s in range(n_states):
            start = s * frames_per_state
            end = (s + 1) * frames_per_state if s < n_states - 1 else n_frames
            state_frames[s].append(feat[start:end])

    state_means = np.zeros((n_states, n_features))
    state_covars = np.zeros((n_states, n_features))

    for s in range(n_states):
        all_frames = np.vstack(state_frames[s])
        state_means[s] = np.mean(all_frames, axis=0)
        state_covars[s] = np.var(all_frames, axis=0) + hmm_config.cov_regularization

    if isinstance(model, hmm.GMMHMM):
        rng = np.random.default_rng(0)
        means = np.repeat(state_means[:, None, :], hmm_config.n_mix, axis=1)

        for s in range(n_states):
            all_frames = np.vstack(state_frames[s])
            if len(all_frames) >= hmm_config.n_mix:
                idx = np.linspace(0, len(all_frames) - 1, hmm_config.n_mix, dtype=int)
                means[s] = all_frames[idx]
            else:
                means[s] += 0.01 * rng.standard_normal((hmm_config.n_mix, n_features))

        model.weights_ = np.full((n_states, hmm_config.n_mix), 1.0 / hmm_config.n_mix)
        model.means_ = means

        if hmm_config.covariance_type == "diag":
            model.covars_ = np.repeat(state_covars[:, None, :], hmm_config.n_mix, axis=1)
        else:
            model.covars_ = np.array(
                [[np.diag(state_covars[s]) for _ in range(hmm_config.n_mix)]
                 for s in range(n_states)]
            )
    else:
        model.means_ = state_means
        if hmm_config.covariance_type == "diag":
            model.covars_ = state_covars
        else:
            model.covars_ = np.array([np.diag(state_covars[s]) for s in range(n_states)])

    return model


def _estimate_model_storage_bytes(model: HMMModel) -> int:
    total = 0
    for attr in ["startprob_", "transmat_", "weights_", "means_", "covars_"]:
        if hasattr(model, attr):
            value = getattr(model, attr)
            if isinstance(value, np.ndarray):
                total += value.nbytes
    return int(total)

class HMMKeywordSpotter:
    """HMM/GMM-based few-shot keyword spotter with open-set rejection."""

    def __init__(self, config: Config):
        self.config = config
        self.hmm_cfg = config.hmm
        self.audio_cfg = config.audio

        self.keyword_models: Dict[str, HMMModel] = {}
        self.keywords: List[str] = []
        self.ubm: Optional[HMMModel] = None
        self.rejection_threshold: float = float("-inf")
        self.last_threshold_stats: Dict[str, float] = {}

    def _train_single_hmm(
        self, keyword: str, features_list: List[np.ndarray]
    ) -> HMMModel:
        """Train one keyword model; fall back to Gaussian if GMM-HMM is unstable."""
        n_features = features_list[0].shape[1]
        avg_frames = np.mean([len(f) for f in features_list])
        effective_states = min(self.hmm_cfg.n_states, max(3, int(avg_frames / 10)))

        train_cfg = copy.deepcopy(self.hmm_cfg)

        model = build_left_right_hmm(effective_states, n_features, train_cfg)
        model = initialize_hmm_with_data(model, features_list, train_cfg)

        X = np.vstack(features_list)
        lengths = [len(f) for f in features_list]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(X, lengths)
            except Exception as exc:
                if train_cfg.n_mix > 1:
                    print(
                        f"  Warning: GMM-HMM training failed for '{keyword}', "
                        f"falling back to GaussianHMM: {exc}"
                    )
                    train_cfg.n_mix = 1
                    model = build_left_right_hmm(effective_states, n_features, train_cfg)
                    model = initialize_hmm_with_data(model, features_list, train_cfg)
                    model.fit(X, lengths)
                else:
                    raise

        return model

    def _train_ubm(self, split: FewShotSplit):
        """
        Train UBM using background/validation material.
        """
        print("[HMM] Training Universal Background Model (UBM)...")

        ubm_files = []

        if getattr(split, "background_support", None):
            ubm_files.extend(split.background_support)

        ubm_files.extend(split.val_test_unknown)

        for _, files in split.val_test_known.items():
            ubm_files.extend(files[:20])

        ubm_files.extend(split.test_unknown[:50])

        seen = set()
        deduped = []
        for fp in ubm_files:
            if fp not in seen:
                deduped.append(fp)
                seen.add(fp)
        ubm_files = deduped

        if not ubm_files:
            print("  No data for UBM, skipping.")
            self.ubm = None
            return

        ubm_features = extract_features_batch(ubm_files, self.audio_cfg)
        self.ubm = self._train_single_hmm("ubm", ubm_features)
        print(f"  UBM trained on {len(ubm_files)} utterances")

    def enroll(
        self, keywords: List[str], enrollment: Dict[str, List[str]], split: FewShotSplit
    ):
        """Train per-keyword HMM/GMM models."""
        self.keywords = keywords
        self.keyword_models = {}

        print(f"[HMM] Enrolling {len(next(iter(enrollment.values())))}-shot keyword models...")

        for kw in self.keywords:
            features = extract_features_batch(enrollment[kw], self.audio_cfg)
            model = self._train_single_hmm(kw, features)
            self.keyword_models[kw] = model
            print(f"  Keyword '{kw}': HMM trained ({model.n_components} states)")

        if self.hmm_cfg.use_ubm:
            self._train_ubm(split)

    def _score_utterance(self, features: np.ndarray) -> Dict[str, float]:
        """Score utterance against all keyword models."""
        scores: Dict[str, float] = {}
        n_frames = max(1, len(features))

        ubm_score = None
        if self.ubm is not None:
            try:
                ubm_score = float(self.ubm.score(features) / n_frames)
            except Exception:
                ubm_score = -100.0

        for kw in self.keywords:
            try:
                ll = float(self.keyword_models[kw].score(features) / n_frames)
                scores[kw] = ll - ubm_score if ubm_score is not None else ll
            except Exception:
                scores[kw] = -1e10

        return scores

    def predict_single(
        self, query_features: np.ndarray
    ) -> Tuple[str, float, Dict[str, float]]:
        scores = self._score_utterance(query_features)
        best_kw = max(scores, key=scores.get)
        best_score = scores[best_kw]

        if best_score < self.rejection_threshold:
            return "unknown", best_score, scores
        return best_kw, best_score, scores

    def predict_batch(
        self,
        known_test: Dict[str, List[str]],
        unknown_test: List[str]
    ) -> Dict:
        results = {
            "predictions": [], "true_labels": [],
            "scores": [], "all_scores": [], "is_known": []
        }

        print("[HMM] Testing known keywords...")
        for kw in tqdm(known_test.keys()):
            features_list = extract_features_batch(known_test[kw], self.audio_cfg)
            for feat in features_list:
                pred, score, all_scores = self.predict_single(feat)
                results["predictions"].append(pred)
                results["true_labels"].append(kw)
                results["scores"].append(score)
                results["all_scores"].append(all_scores)
                results["is_known"].append(True)

        print("[HMM] Testing unknown utterances...")
        unknown_features = extract_features_batch(unknown_test, self.audio_cfg)
        for feat in tqdm(unknown_features):
            pred, score, all_scores = self.predict_single(feat)
            results["predictions"].append(pred)
            results["true_labels"].append("unknown")
            results["scores"].append(score)
            results["all_scores"].append(all_scores)
            results["is_known"].append(False)

        return results

    def tune_threshold(self, split: FewShotSplit) -> float:
        print(
            f"[HMM] Tuning threshold on validation keywords "
            f"for target FAR={self.config.eval.target_far:.2%}..."
        )

        original_models = self.keyword_models.copy()
        original_keywords = self.keywords.copy()

        self.keyword_models = {}
        self.keywords = split.val_known_keywords
        for kw in split.val_known_keywords:
            features = extract_features_batch(split.val_enrollment[kw], self.audio_cfg)
            self.keyword_models[kw] = self._train_single_hmm(kw, features)

        self.rejection_threshold = float("-inf")

        val_scores_known = []
        for _, files in split.val_test_known.items():
            features = extract_features_batch(files, self.audio_cfg)
            for feat in features:
                _, score, _ = self.predict_single(feat)
                val_scores_known.append(score)

        val_scores_unknown = []
        unknown_features = extract_features_batch(split.val_test_unknown, self.audio_cfg)
        for feat in unknown_features:
            _, score, _ = self.predict_single(feat)
            val_scores_unknown.append(score)

        self.keyword_models = original_models
        self.keywords = original_keywords

        stats = find_threshold_for_target_far(
            known_values=np.array(val_scores_known),
            unknown_values=np.array(val_scores_unknown),
            target_far=self.config.eval.target_far,
            score_type="score",
        )

        self.rejection_threshold = stats["threshold"]
        self.last_threshold_stats = stats

        print(
            f"  Threshold={stats['threshold']:.4f} | "
            f"val FAR={stats['far'] * 100:.2f}% | "
            f"val FRR={stats['frr'] * 100:.2f}% | "
            f"known accept={stats['known_accept_rate'] * 100:.2f}%"
        )

        return self.rejection_threshold

    def get_timing_info(self, n_samples: int = 10) -> Dict:
        """Measure inference time and storage footprint."""
        n_features = self.audio_cfg.n_mfcc * (
            3 if self.audio_cfg.include_delta else 1
        )
        dummy = np.random.randn(100, n_features)

        times = []
        for _ in range(n_samples):
            start = time.time()
            self.predict_single(dummy)
            times.append(time.time() - start)

        total_params = 0
        storage_bytes = 0

        for model in self.keyword_models.values():
            for attr in ["startprob_", "transmat_", "weights_", "means_", "covars_"]:
                if hasattr(model, attr):
                    value = getattr(model, attr)
                    if isinstance(value, np.ndarray):
                        total_params += value.size
            storage_bytes += _estimate_model_storage_bytes(model)

        if self.ubm is not None:
            storage_bytes += _estimate_model_storage_bytes(self.ubm)

        return {
            "mean_inference_time_ms": float(np.mean(times) * 1000),
            "std_inference_time_ms": float(np.std(times) * 1000),
            "model_params": int(total_params),
            "storage_bytes": int(storage_bytes),
            "n_keyword_models": len(self.keyword_models),
        }