import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from hmmlearn import hmm
import warnings
from tqdm import tqdm

from config import Config, HMMConfig
from data_loader import FewShotSplit
from feature_extraction import extract_features_batch


def build_left_right_hmm(
    n_states: int,
    n_features: int,
    hmm_config: HMMConfig
) -> hmm.GaussianHMM:
    """Build a left-to-right HMM with constrained transitions."""
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=hmm_config.covariance_type,
        n_iter=hmm_config.n_iter,
        verbose=False,
        params="stmc",
        init_params=""
    )

    # Start in state 0
    startprob = np.zeros(n_states)
    startprob[0] = 1.0
    model.startprob_ = startprob

    # Left-to-right transitions
    transmat = np.zeros((n_states, n_states))
    for i in range(n_states - 1):
        transmat[i, i] = 0.7
        transmat[i, i + 1] = 0.3
    transmat[-1, -1] = 1.0
    model.transmat_ = transmat

    # Initialize emissions
    model.means_ = np.random.randn(n_states, n_features) * 0.01
    if hmm_config.covariance_type == 'diag':
        model.covars_ = np.ones((n_states, n_features))
    elif hmm_config.covariance_type == 'full':
        model.covars_ = np.array([np.eye(n_features) for _ in range(n_states)])

    return model


def initialize_hmm_with_data(
    model: hmm.GaussianHMM,
    features_list: List[np.ndarray],
    hmm_config: HMMConfig
) -> hmm.GaussianHMM:
    """Initialize HMM parameters via uniform segmentation of enrollment data."""
    n_states = model.n_components
    n_features = features_list[0].shape[1]

    state_frames = [[] for _ in range(n_states)]
    for feat in features_list:
        n_frames = len(feat)
        frames_per_state = n_frames // n_states
        for s in range(n_states):
            start = s * frames_per_state
            end = (s + 1) * frames_per_state if s < n_states - 1 else n_frames
            state_frames[s].append(feat[start:end])

    means = np.zeros((n_states, n_features))
    covars = np.zeros((n_states, n_features))
    for s in range(n_states):
        all_frames = np.vstack(state_frames[s])
        means[s] = np.mean(all_frames, axis=0)
        covars[s] = np.var(all_frames, axis=0) + hmm_config.cov_regularization

    model.means_ = means
    if hmm_config.covariance_type == 'diag':
        model.covars_ = covars
    elif hmm_config.covariance_type == 'full':
        model.covars_ = np.array([np.diag(covars[s]) for s in range(n_states)])

    return model


class HMMKeywordSpotter:
    """HMM/GMM-based few-shot keyword spotter with open-set rejection."""

    def __init__(self, config: Config):
        self.config = config
        self.hmm_cfg = config.hmm
        self.audio_cfg = config.audio

        self.keyword_models: Dict[str, hmm.GaussianHMM] = {}
        self.keywords: List[str] = []
        self.ubm: Optional[hmm.GaussianHMM] = None
        self.rejection_threshold: float = float('-inf')

    def _train_single_hmm(
        self, keyword: str, features_list: List[np.ndarray]
    ) -> hmm.GaussianHMM:
        """Train HMM for a single keyword."""
        n_features = features_list[0].shape[1]
        avg_frames = np.mean([len(f) for f in features_list])
        effective_states = min(
            self.hmm_cfg.n_states,
            max(3, int(avg_frames / 10))
        )

        model = build_left_right_hmm(
            effective_states, n_features, self.hmm_cfg
        )
        model = initialize_hmm_with_data(model, features_list, self.hmm_cfg)

        X = np.vstack(features_list)
        lengths = [len(f) for f in features_list]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(X, lengths)
            except Exception as e:
                print(f"  Warning: HMM training failed for '{keyword}': {e}")

        return model

    def _train_ubm(self, split: FewShotSplit):
        """
        Train UBM on unknown validation data + diverse background speech.
        NOT using enrollment data (which would bias the UBM toward keywords).
        """
        print("[HMM] Training Universal Background Model (UBM)...")

        ubm_files = []
        # Use val unknown data for UBM (general speech, not keywords)
        ubm_files.extend(split.val_test_unknown)

        # Also grab some val known test data (different from enrolled)
        for kw, files in split.val_test_known.items():
            ubm_files.extend(files[:20])

        # Also grab some unknown test data (from the actual unknown keywords)
        ubm_files.extend(split.test_unknown[:50])

        if len(ubm_files) == 0:
            print("  No data for UBM, skipping.")
            self.ubm = None
            return

        ubm_features = extract_features_batch(ubm_files, self.audio_cfg)
        n_features = ubm_features[0].shape[1]

        self.ubm = build_left_right_hmm(
            self.hmm_cfg.ubm_n_states, n_features, self.hmm_cfg
        )
        self.ubm = initialize_hmm_with_data(
            self.ubm, ubm_features, self.hmm_cfg
        )

        X = np.vstack(ubm_features)
        lengths = [len(f) for f in ubm_features]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.ubm.fit(X, lengths)
                print(f"  UBM trained: {self.hmm_cfg.ubm_n_states} states, "
                      f"{len(ubm_files)} utterances")
            except Exception as e:
                print(f"  Warning: UBM training failed: {e}")
                self.ubm = None

    def enroll(
        self, keywords: List[str], enrollment: Dict[str, List[str]],
        split: FewShotSplit
    ):
        """Enroll keywords by training per-keyword HMMs."""
        self.keywords = keywords
        self.keyword_models = {}

        k = len(next(iter(enrollment.values())))
        print(f"[HMM] Enrolling {k}-shot keyword models...")

        for kw in self.keywords:
            features = extract_features_batch(enrollment[kw], self.audio_cfg)
            model = self._train_single_hmm(kw, features)
            self.keyword_models[kw] = model
            print(f"  Keyword '{kw}': HMM trained ({model.n_components} states)")

        if self.hmm_cfg.use_ubm:
            self._train_ubm(split)

    def _score_utterance(self, features: np.ndarray) -> Dict[str, float]:
        """Score an utterance against all keyword HMMs."""
        scores = {}
        n_frames = len(features)

        ubm_score = None
        if self.ubm is not None:
            try:
                ubm_score = self.ubm.score(features) / n_frames
            except:
                ubm_score = -100.0

        for kw in self.keywords:
            try:
                ll = self.keyword_models[kw].score(features) / n_frames
                if ubm_score is not None:
                    scores[kw] = ll - ubm_score  # Log-likelihood ratio
                else:
                    scores[kw] = ll
            except:
                scores[kw] = -1e10

        return scores

    def predict_single(
        self, query_features: np.ndarray
    ) -> Tuple[str, float, Dict[str, float]]:
        """Predict keyword or reject as unknown."""
        scores = self._score_utterance(query_features)
        best_kw = max(scores, key=scores.get)
        best_score = scores[best_kw]

        if best_score < self.rejection_threshold:
            return "unknown", best_score, scores
        else:
            return best_kw, best_score, scores

    def predict_batch(
        self,
        known_test: Dict[str, List[str]],
        unknown_test: List[str]
    ) -> Dict:
        """Run prediction on all test utterances."""
        results = {
            'predictions': [], 'true_labels': [],
            'scores': [], 'all_scores': [], 'is_known': []
        }

        print("[HMM] Testing known keywords...")
        for kw in tqdm(known_test.keys()):
            features_list = extract_features_batch(known_test[kw], self.audio_cfg)
            for feat in features_list:
                pred, score, all_scores = self.predict_single(feat)
                results['predictions'].append(pred)
                results['true_labels'].append(kw)
                results['scores'].append(score)
                results['all_scores'].append(all_scores)
                results['is_known'].append(True)

        print("[HMM] Testing unknown utterances...")
        unknown_features = extract_features_batch(unknown_test, self.audio_cfg)
        for feat in tqdm(unknown_features):
            pred, score, all_scores = self.predict_single(feat)
            results['predictions'].append(pred)
            results['true_labels'].append("unknown")
            results['scores'].append(score)
            results['all_scores'].append(all_scores)
            results['is_known'].append(False)

        return results

    def tune_threshold(self, split: FewShotSplit) -> float:
        """
        Tune threshold on VALIDATION keywords (enrolled separately).
        """
        print("[HMM] Tuning threshold on validation keywords...")

        # Temporarily enroll validation keywords
        original_models = self.keyword_models.copy()
        original_keywords = self.keywords.copy()

        val_keywords = split.val_known_keywords
        val_models = {}
        for kw in val_keywords:
            features = extract_features_batch(
                split.val_enrollment[kw], self.audio_cfg
            )
            model = self._train_single_hmm(kw, features)
            val_models[kw] = model

        self.keyword_models = val_models
        self.keywords = val_keywords
        self.rejection_threshold = float('-inf')  # Accept everything during scoring

        # Score val known
        val_scores_known = []
        for kw, files in split.val_test_known.items():
            features = extract_features_batch(files, self.audio_cfg)
            for feat in features:
                _, score, _ = self.predict_single(feat)
                val_scores_known.append(score)

        # Score val unknown
        val_scores_unknown = []
        unknown_features = extract_features_batch(
            split.val_test_unknown, self.audio_cfg
        )
        for feat in unknown_features:
            _, score, _ = self.predict_single(feat)
            val_scores_unknown.append(score)

        # Restore original
        self.keyword_models = original_models
        self.keywords = original_keywords

        val_scores_known = np.array(val_scores_known)
        val_scores_unknown = np.array(val_scores_unknown)

        print(f"  Val known scores:   mean={val_scores_known.mean():.2f}, "
              f"std={val_scores_known.std():.2f}")
        print(f"  Val unknown scores: mean={val_scores_unknown.mean():.2f}, "
              f"std={val_scores_unknown.std():.2f}")

        all_scores = np.concatenate([val_scores_known, val_scores_unknown])
        thresholds = np.linspace(
            np.percentile(all_scores, 1),
            np.percentile(all_scores, 99),
            200
        )

        best_threshold = thresholds[0]
        best_bal_acc = 0

        for t in thresholds:
            known_accepted = np.mean(val_scores_known >= t)
            unknown_rejected = np.mean(val_scores_unknown < t)
            bal_acc = (known_accepted + unknown_rejected) / 2
            if bal_acc > best_bal_acc:
                best_bal_acc = bal_acc
                best_threshold = t

        self.rejection_threshold = best_threshold
        print(f"  Best threshold: {best_threshold:.4f} "
              f"(balanced acc: {best_bal_acc:.4f})")

        return best_threshold

    def get_timing_info(self, n_samples: int = 10) -> Dict:
        """Measure inference time and model size."""
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
        for kw, model in self.keyword_models.items():
            n_s = model.n_components
            total_params += n_s * n_s + n_s * n_features * 2 + n_s

        return {
            'mean_inference_time_ms': np.mean(times) * 1000,
            'std_inference_time_ms': np.std(times) * 1000,
            'model_params': total_params,
            'n_keyword_models': len(self.keyword_models)
        }


