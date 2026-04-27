import numpy as np
import time
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from dtw import dtw as dtw_compute

from config import Config, DTWConfig
from data_loader import FewShotSplit
from evaluation import find_threshold_for_target_far
from feature_extraction import extract_features_batch


class DTWKeywordSpotter:
    """DTW-based few-shot keyword spotter with open-set rejection."""

    def __init__(self, config: Config):
        self.config = config
        self.dtw_cfg = config.dtw
        self.audio_cfg = config.audio

        self.templates: Dict[str, List[np.ndarray]] = {}
        self.keywords: List[str] = []
        self.last_threshold_stats: Dict[str, float] = {}
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
        Tune rejection threshold on validation keywords to hit target FAR.
        """
        print(
            f"[DTW] Tuning threshold on validation keywords "
            f"for target FAR={self.config.eval.target_far:.2%}..."
        )

        original_templates = self.templates.copy()
        original_keywords = self.keywords.copy()

        self.templates = {
            kw: extract_features_batch(split.val_enrollment[kw], self.audio_cfg)
            for kw in split.val_known_keywords
        }
        self.keywords = split.val_known_keywords
        self.rejection_threshold = float("inf")

        val_distances_known = []
        for _, files in split.val_test_known.items():
            features = extract_features_batch(files, self.audio_cfg)
            for feat in features:
                _, dist, _ = self.predict_single(feat)
                val_distances_known.append(dist)

        val_distances_unknown = []
        unknown_features = extract_features_batch(split.val_test_unknown, self.audio_cfg)
        for feat in unknown_features:
            _, dist, _ = self.predict_single(feat)
            val_distances_unknown.append(dist)

        self.templates = original_templates
        self.keywords = original_keywords

        stats = find_threshold_for_target_far(
            known_values=np.array(val_distances_known),
            unknown_values=np.array(val_distances_unknown),
            target_far=self.config.eval.target_far,
            score_type="distance",
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