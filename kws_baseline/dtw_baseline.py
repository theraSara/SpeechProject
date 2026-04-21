import numpy as np
from typing import Dict, List, Tuple, Optional
from dtw import dtw as dtw_compute
import time
from tqdm import tqdm

from config import Config, DTWConfig
from data_loader import FewShotSplit
from feature_extraction import extract_features_batch


class DTWKeywordSpotter:
    """DTW-based few-shot keyword spotter with open-set rejection."""

    def __init__(self, config: Config):
        self.config = config
        self.dtw_cfg = config.dtw
        self.audio_cfg = config.audio

        self.templates: Dict[str, List[np.ndarray]] = {}
        self.keywords: List[str] = []
        self.rejection_threshold: float = float('inf')

    def enroll(self, keywords: List[str], enrollment: Dict[str, List[str]]):
        """Enroll keywords by extracting and storing MFCC templates."""
        self.templates = {}
        self.keywords = keywords

        for kw in self.keywords:
            file_paths = enrollment[kw]
            features = extract_features_batch(file_paths, self.audio_cfg)
            self.templates[kw] = features
            print(f"  Keyword '{kw}': {len(features)} templates, "
                  f"shape {features[0].shape}")

    def _dtw_distance(self, query: np.ndarray, template: np.ndarray) -> float:
        """Compute DTW distance between query and template."""
        kwargs = dict(
            dist_method=self.dtw_cfg.distance_metric,
            keep_internals=False
        )

        # Add Sakoe-Chiba band constraint
        if self.dtw_cfg.use_sakoe_chiba_band:
            kwargs['window_type'] = 'sakoechiba'
            kwargs['window_args'] = {
                'window_size': self.dtw_cfg.sakoe_chiba_radius
            }

        try:
            alignment = dtw_compute(query, template, **kwargs)
            return alignment.normalizedDistance
        except Exception:
            # Fallback without window constraint
            alignment = dtw_compute(
                query, template,
                dist_method=self.dtw_cfg.distance_metric,
                keep_internals=False
            )
            return alignment.normalizedDistance

    def _compute_keyword_distance(
        self, query: np.ndarray, keyword: str
    ) -> float:
        """Compute distance from query to a keyword's template set."""
        distances = [
            self._dtw_distance(query, template)
            for template in self.templates[keyword]
        ]

        if self.dtw_cfg.aggregation == "mean":
            return np.mean(distances)
        elif self.dtw_cfg.aggregation == "min":
            return np.min(distances)
        elif self.dtw_cfg.aggregation == "median":
            return np.median(distances)
        else:
            raise ValueError(f"Unknown aggregation: {self.dtw_cfg.aggregation}")

    def predict_single(
        self, query_features: np.ndarray
    ) -> Tuple[str, float, Dict[str, float]]:
        """Predict keyword or reject as unknown."""
        distances = {}
        for kw in self.keywords:
            distances[kw] = self._compute_keyword_distance(query_features, kw)

        best_kw = min(distances, key=distances.get)
        best_dist = distances[best_kw]

        if best_dist > self.rejection_threshold:
            return "unknown", best_dist, distances
        else:
            return best_kw, best_dist, distances

    def predict_batch(
        self,
        known_test: Dict[str, List[str]],
        unknown_test: List[str]
    ) -> Dict:
        """Run prediction on all test utterances."""
        results = {
            'predictions': [], 'true_labels': [],
            'distances': [], 'all_distances': [], 'is_known': []
        }

        print("[DTW] Testing known keywords...")
        for kw in tqdm(known_test.keys()):
            features_list = extract_features_batch(known_test[kw], self.audio_cfg)
            for feat in features_list:
                pred, dist, all_dist = self.predict_single(feat)
                results['predictions'].append(pred)
                results['true_labels'].append(kw)
                results['distances'].append(dist)
                results['all_distances'].append(all_dist)
                results['is_known'].append(True)

        print("[DTW] Testing unknown utterances...")
        unknown_features = extract_features_batch(unknown_test, self.audio_cfg)
        for feat in tqdm(unknown_features):
            pred, dist, all_dist = self.predict_single(feat)
            results['predictions'].append(pred)
            results['true_labels'].append("unknown")
            results['distances'].append(dist)
            results['all_distances'].append(all_dist)
            results['is_known'].append(False)

        return results

    def tune_threshold(self, split: FewShotSplit) -> float:
        """
        Tune rejection threshold on VALIDATION keywords.
        
        KEY FIX: We enroll the validation keywords separately, then
        test val_known (should be accepted) vs val_unknown (should be rejected).
        This gives the threshold tuner real signal.
        """
        print("[DTW] Tuning threshold on validation keywords...")

        # Temporarily enroll validation keywords
        original_templates = self.templates.copy()
        original_keywords = self.keywords.copy()

        # Enroll val keywords
        val_keywords = split.val_known_keywords
        val_templates = {}
        for kw in val_keywords:
            features = extract_features_batch(
                split.val_enrollment[kw], self.audio_cfg
            )
            val_templates[kw] = features

        self.templates = val_templates
        self.keywords = val_keywords
        self.rejection_threshold = float('inf')  # No rejection during scoring

        # Score val known (should be close to their enrolled keyword)
        val_distances_known = []
        for kw, files in split.val_test_known.items():
            features = extract_features_batch(files, self.audio_cfg)
            for feat in features:
                _, dist, _ = self.predict_single(feat)
                val_distances_known.append(dist)

        # Score val unknown (should be far from all enrolled keywords)
        val_distances_unknown = []
        unknown_features = extract_features_batch(
            split.val_test_unknown, self.audio_cfg
        )
        for feat in unknown_features:
            _, dist, _ = self.predict_single(feat)
            val_distances_unknown.append(dist)

        # Restore original enrollment
        self.templates = original_templates
        self.keywords = original_keywords

        # Find best threshold
        val_distances_known = np.array(val_distances_known)
        val_distances_unknown = np.array(val_distances_unknown)

        print(f"  Val known distances:   mean={val_distances_known.mean():.2f}, "
              f"std={val_distances_known.std():.2f}")
        print(f"  Val unknown distances: mean={val_distances_unknown.mean():.2f}, "
              f"std={val_distances_unknown.std():.2f}")

        all_dists = np.concatenate([val_distances_known, val_distances_unknown])
        thresholds = np.linspace(
            np.percentile(all_dists, 1),
            np.percentile(all_dists, 99),
            200
        )

        best_threshold = thresholds[0]
        best_score = 0

        for t in thresholds:
            known_accepted = np.mean(val_distances_known <= t)
            unknown_rejected = np.mean(val_distances_unknown > t)
            score = (known_accepted + unknown_rejected) / 2
            if score > best_score:
                best_score = score
                best_threshold = t

        self.rejection_threshold = best_threshold
        print(f"  Best threshold: {best_threshold:.4f} "
              f"(balanced acc: {best_score:.4f})")

        return best_threshold

    def get_timing_info(self, n_samples: int = 10) -> Dict:
        """Measure inference time per utterance."""
        n_features = self.audio_cfg.n_mfcc * (
            3 if self.audio_cfg.include_delta else 1
        )
        dummy = np.random.randn(100, n_features)

        times = []
        for _ in range(n_samples):
            start = time.time()
            self.predict_single(dummy)
            times.append(time.time() - start)

        return {
            'mean_inference_time_ms': np.mean(times) * 1000,
            'std_inference_time_ms': np.std(times) * 1000,
            'model_params': 0,
            'storage_bytes': sum(
                sum(t.nbytes for t in templates)
                for templates in self.templates.values()
            )
        }