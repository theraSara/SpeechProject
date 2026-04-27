import argparse
import json
import os
from typing import Dict

import numpy as np

from config import Config
from data_loader import get_few_shot_split, download_gsc, validate_split
from dtw_baseline import DTWKeywordSpotter
from hmm_baseline import HMMKeywordSpotter
from evaluation import (
    compute_metrics,
    compute_operating_points,
    compute_roc,
    compute_fixed_far_metrics,
    apply_threshold_to_results,
    plot_confusion_matrix,
    plot_distance_distributions,
    plot_roc,
    print_operating_points_table,
    print_results_table,
    save_metrics_csv
)


def _to_builtin(value):
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value

def run_dtw_experiment(config: Config, split, k_shot: int) -> Dict:
    print(f"\n{'=' * 60}")
    print(f" DTW Baseline - {k_shot}-shot")
    print(f"{'=' * 60}")

    spotter = DTWKeywordSpotter(config)
    spotter.enroll(split.known_keywords, split.enrollment)

    if getattr(split, "official_mode", False):
        # For official episodic metrics, use the same episode + negative pool
        # to compute ACC+_5%, AUROC, FRR+_5%.
        spotter.rejection_threshold = float("inf")
    else:
        spotter.tune_threshold(split)

    results = spotter.predict_batch(split.test_known, split.test_unknown)

    if getattr(split, "official_mode", False):
        fixed_far = compute_fixed_far_metrics(
            results,
            target_far=config.eval.target_far,
            score_key="distances",
            score_type="distance",
        )
        thresholded_results = fixed_far["thresholded_results"]
        metrics = compute_metrics(thresholded_results)

        metrics.update({
            "auc": float(fixed_far["auroc"]),
            "auroc": float(fixed_far["auroc"]),
            "threshold": float(fixed_far["threshold_5"]),
            "threshold_5": float(fixed_far["threshold_5"]),
            "acc_plus_5": float(fixed_far["acc_plus_5"]),
            "frr_plus_5": float(fixed_far["frr_plus_5"]),
            "far_at_5": float(fixed_far["far_at_5"]),
            "val_far": float(fixed_far["far_at_5"]),
            "val_frr": float(fixed_far["frr_plus_5"]),
        })

        plot_results = thresholded_results
    else:
        metrics = compute_metrics(results)
        fpr, tpr, _, auc_score = compute_roc(
            results, score_key="distances", score_type="distance"
        )
        metrics.update({
            "auc": float(auc_score),
            "auroc": float(auc_score),
            "threshold": float(spotter.rejection_threshold),
            "threshold_5": float(spotter.rejection_threshold),
            "acc_plus_5": float(metrics.get("known_accuracy", 0.0)),
            "frr_plus_5": float(metrics.get("false_rejection_rate", 0.0)),
            "far_at_5": float(metrics.get("false_acceptance_rate", 0.0)),
            "val_far": float(spotter.last_threshold_stats.get("far", np.nan)),
            "val_frr": float(spotter.last_threshold_stats.get("frr", np.nan)),
        })
        plot_results = results

    fpr, tpr, _, auc_score = compute_roc(
        results, score_key="distances", score_type="distance"
    )

    return {
        "metrics": metrics,
        "results": plot_results,
        "raw_results": results,
        "roc": (fpr, tpr, auc_score),
        "timing": spotter.get_timing_info(),
        "keywords": split.known_keywords,
    }
def run_hmm_experiment(config: Config, split, k_shot: int) -> Dict:
    print(f"\n{'=' * 60}")
    print(f" HMM/GMM Baseline - {k_shot}-shot")
    print(f"{'=' * 60}")

    spotter = HMMKeywordSpotter(config)
    spotter.enroll(split.known_keywords, split.enrollment, split)

    if getattr(split, "official_mode", False):
        spotter.rejection_threshold = float("-inf")
    else:
        spotter.tune_threshold(split)

    results = spotter.predict_batch(split.test_known, split.test_unknown)

    if getattr(split, "official_mode", False):
        fixed_far = compute_fixed_far_metrics(
            results,
            target_far=config.eval.target_far,
            score_key="scores",
            score_type="score",
        )
        thresholded_results = fixed_far["thresholded_results"]
        metrics = compute_metrics(thresholded_results)

        metrics.update({
            "auc": float(fixed_far["auroc"]),
            "auroc": float(fixed_far["auroc"]),
            "threshold": float(fixed_far["threshold_5"]),
            "threshold_5": float(fixed_far["threshold_5"]),
            "acc_plus_5": float(fixed_far["acc_plus_5"]),
            "frr_plus_5": float(fixed_far["frr_plus_5"]),
            "far_at_5": float(fixed_far["far_at_5"]),
            "val_far": float(fixed_far["far_at_5"]),
            "val_frr": float(fixed_far["frr_plus_5"]),
        })

        plot_results = thresholded_results
    else:
        metrics = compute_metrics(results)
        fpr, tpr, _, auc_score = compute_roc(
            results, score_key="scores", score_type="score"
        )
        metrics.update({
            "auc": float(auc_score),
            "auroc": float(auc_score),
            "threshold": float(spotter.rejection_threshold),
            "threshold_5": float(spotter.rejection_threshold),
            "acc_plus_5": float(metrics.get("known_accuracy", 0.0)),
            "frr_plus_5": float(metrics.get("false_rejection_rate", 0.0)),
            "far_at_5": float(metrics.get("false_acceptance_rate", 0.0)),
            "val_far": float(spotter.last_threshold_stats.get("far", np.nan)),
            "val_frr": float(spotter.last_threshold_stats.get("frr", np.nan)),
        })
        plot_results = results

    fpr, tpr, _, auc_score = compute_roc(
        results, score_key="scores", score_type="score"
    )

    return {
        "metrics": metrics,
        "results": plot_results,
        "raw_results": results,
        "roc": (fpr, tpr, auc_score),
        "timing": spotter.get_timing_info(),
        "keywords": split.known_keywords,
    }

def _average_metrics(trial_list):
    avg = {}
    for key in trial_list[0]:
        values = [t[key] for t in trial_list]
        avg[f"{key}_mean"] = float(np.mean(values))
        avg[f"{key}_std"] = float(np.std(values))
        avg[key] = float(np.mean(values))
    return avg

def run_all_experiments(config: Config):
    all_metrics = {}
    all_rocs = {}
    all_timing = {}
    all_op_points = {}
    per_trial_dump = {}

    for k_shot in config.data.k_shots:
        probe_split = get_few_shot_split(config, k_shot, 0, config.data.seed)
        validate_split(probe_split)

        if getattr(probe_split, "official_mode", False):
            n_trials_for_k = int(probe_split.official_n_episodes)
            print(f"\n[Info] Official episodic split detected for {k_shot}-shot.")
            print(f"[Info] Overriding n_trials to {n_trials_for_k} official episodes.")
        else:
            n_trials_for_k = config.eval.n_trials

        print(f"\n{'#' * 60}")
        print(f"  Running {k_shot}-shot experiments")
        print(f"  ({n_trials_for_k} trials)")
        print(f"{'#' * 60}")

        dtw_trial_metrics = []
        hmm_trial_metrics = []

        last_split = None
        last_dtw_result = None
        last_hmm_result = None

        for trial_idx in range(n_trials_for_k):
            seed = config.data.seed + trial_idx
            print(f"\n--- Trial {trial_idx + 1}/{n_trials_for_k} (seed={seed}) ---")

            split = get_few_shot_split(config, k_shot, trial_idx, seed)
            validate_split(split)

            dtw_result = run_dtw_experiment(config, split, k_shot)
            hmm_result = run_hmm_experiment(config, split, k_shot)

            dtw_trial_metrics.append(dtw_result["metrics"])
            hmm_trial_metrics.append(hmm_result["metrics"])

            per_trial_dump[f"k{k_shot}_trial{trial_idx + 1}"] = {
                "dtw": _to_builtin(dtw_result["metrics"]),
                "hmm_gmm": _to_builtin(hmm_result["metrics"]),
            }

            last_split = split
            last_dtw_result = dtw_result
            last_hmm_result = hmm_result

        dtw_key = f"MFCC+DTW ({k_shot}-shot)"
        hmm_key = f"MFCC+HMM-GMM ({k_shot}-shot)"

        all_metrics[dtw_key] = _average_metrics(dtw_trial_metrics)
        all_metrics[hmm_key] = _average_metrics(hmm_trial_metrics)

        all_rocs[dtw_key] = last_dtw_result["roc"]
        all_rocs[hmm_key] = last_hmm_result["roc"]

        all_timing[dtw_key] = last_dtw_result["timing"]
        all_timing[hmm_key] = last_hmm_result["timing"]

        output_dir = config.eval.output_dir
        for name, result, score_key in [
            (dtw_key, last_dtw_result, "distances"),
            (hmm_key, last_hmm_result, "scores"),
        ]:
            safe_name = name.replace(" ", "_").replace("/", "_")
            plot_confusion_matrix(
                result["results"],
                result["keywords"],
                method_name=name,
                save_path=os.path.join(output_dir, f"cm_{safe_name}.png")
            )
            plot_distance_distributions(
                result["raw_results"],
                method_name=name,
                score_key=score_key,
                save_path=os.path.join(output_dir, f"dist_{safe_name}.png")
            )

        # Operating-point analysis only for legacy mode
        if not getattr(last_split, "official_mode", False):
            dtw_spotter = DTWKeywordSpotter(config)
            dtw_spotter.enroll(last_split.known_keywords, last_split.enrollment)
            dtw_spotter.rejection_threshold = float("inf")
            dtw_results = dtw_spotter.predict_batch(last_split.test_known, last_split.test_unknown)
            all_op_points[dtw_key] = compute_operating_points(dtw_results, "distances", "distance")

            hmm_spotter = HMMKeywordSpotter(config)
            hmm_spotter.enroll(last_split.known_keywords, last_split.enrollment, last_split)
            hmm_spotter.rejection_threshold = float("-inf")
            hmm_results = hmm_spotter.predict_batch(last_split.test_known, last_split.test_unknown)
            all_op_points[hmm_key] = compute_operating_points(hmm_results, "scores", "score")

    if all_op_points:
        print_operating_points_table(all_op_points)
        with open(os.path.join(config.eval.output_dir, "operating_points.json"), "w", encoding="utf-8") as f:
            json.dump(_to_builtin(all_op_points), f, indent=2)

    print_results_table(all_metrics, all_timing)
    save_metrics_csv(all_metrics, all_timing, save_path=os.path.join(config.eval.output_dir, "baseline_results_table.csv"))

    plot_roc(
        all_rocs,
        save_path=os.path.join(config.eval.output_dir, "roc_comparison.png")
    )

    with open(os.path.join(config.eval.output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(_to_builtin(all_metrics), f, indent=2)

    if config.eval.save_per_trial:
        with open(os.path.join(config.eval.output_dir, "per_trial_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(_to_builtin(per_trial_dump), f, indent=2)

    print(f"\nAll results saved to {config.eval.output_dir}/")
    return all_metrics, all_rocs, all_timing


def main():
    parser = argparse.ArgumentParser(description="Run classical KWS baselines")
    parser.add_argument(
        "--gsc_root",
        type=str,
        default="./data/SpeechCommands/speech_commands_v0.02"
    )
    parser.add_argument("--k_shots", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument(
        "--split_manifest_template",
        type=str,
        default="./splits/splits_{k_shot}shot.json",
        help="Optional shared split manifest template, e.g. ./splits/trial_{trial}_k{k_shot}.json"
    )
    parser.add_argument("--download", action="store_true")

    args = parser.parse_args()

    config = Config()
    config.data.gsc_root = args.gsc_root
    config.data.k_shots = args.k_shots
    config.data.split_manifest_template = args.split_manifest_template
    config.eval.n_trials = args.n_trials
    config.eval.output_dir = args.output_dir

    os.makedirs(config.eval.output_dir, exist_ok=True)

    if args.download:
        download_gsc("./data")

    run_all_experiments(config)


if __name__ == "__main__":
    main()