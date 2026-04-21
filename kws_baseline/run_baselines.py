import numpy as np
import os
import json
import argparse

from config import Config
from data_loader import create_few_shot_split, download_gsc
from dtw_baseline import DTWKeywordSpotter
from hmm_baseline import HMMKeywordSpotter
from evaluation import (
    compute_metrics, compute_roc, plot_roc,
    plot_confusion_matrix, plot_distance_distributions,
    print_results_table
)


def run_dtw_experiment(config: Config, split, k_shot: int) -> dict:
    """Run DTW baseline on one split."""
    print(f"\n{'='*60}")
    print(f"  DTW Baseline - {k_shot}-shot")
    print(f"{'='*60}")

    spotter = DTWKeywordSpotter(config)
    spotter.enroll(split.known_keywords, split.enrollment)
    spotter.tune_threshold(split)

    results = spotter.predict_batch(split.test_known, split.test_unknown)
    metrics = compute_metrics(results)

    fpr, tpr, _, auc_score = compute_roc(
        results, score_key='distances', score_type='distance'
    )
    metrics['auc'] = auc_score

    timing = spotter.get_timing_info()

    return {
        'metrics': metrics, 'results': results,
        'roc': (fpr, tpr, auc_score), 'timing': timing,
        'keywords': split.known_keywords
    }


def run_hmm_experiment(config: Config, split, k_shot: int) -> dict:
    """Run HMM/GMM baseline on one split."""
    print(f"\n{'='*60}")
    print(f"  HMM/GMM Baseline - {k_shot}-shot")
    print(f"{'='*60}")

    spotter = HMMKeywordSpotter(config)
    spotter.enroll(split.known_keywords, split.enrollment, split)
    spotter.tune_threshold(split)

    results = spotter.predict_batch(split.test_known, split.test_unknown)
    metrics = compute_metrics(results)

    fpr, tpr, _, auc_score = compute_roc(
        results, score_key='scores', score_type='score'
    )
    metrics['auc'] = auc_score

    timing = spotter.get_timing_info()

    return {
        'metrics': metrics, 'results': results,
        'roc': (fpr, tpr, auc_score), 'timing': timing,
        'keywords': split.known_keywords
    }


def run_all_experiments(config: Config):
    """Run all baseline experiments."""
    all_metrics = {}
    all_rocs = {}
    all_timing = {}

    for k_shot in config.data.k_shots:
        print(f"\n{'#'*60}")
        print(f"  Running {k_shot}-shot experiments")
        print(f"  ({config.eval.n_trials} trials)")
        print(f"{'#'*60}")

        dtw_trial_metrics = []
        hmm_trial_metrics = []

        for trial in range(config.eval.n_trials):
            seed = config.data.seed + trial
            print(f"\n--- Trial {trial+1}/{config.eval.n_trials} (seed={seed}) ---")

            split = create_few_shot_split(config, k_shot, seed)

            dtw_result = run_dtw_experiment(config, split, k_shot)
            dtw_trial_metrics.append(dtw_result['metrics'])

            hmm_result = run_hmm_experiment(config, split, k_shot)
            hmm_trial_metrics.append(hmm_result['metrics'])

        def average_metrics(trial_list):
            avg = {}
            for key in trial_list[0]:
                values = [t[key] for t in trial_list]
                avg[f'{key}_mean'] = np.mean(values)
                avg[f'{key}_std'] = np.std(values)
                avg[key] = np.mean(values)
            return avg

        dtw_key = f"DTW ({k_shot}-shot)"
        hmm_key = f"HMM/GMM ({k_shot}-shot)"

        all_metrics[dtw_key] = average_metrics(dtw_trial_metrics)
        all_metrics[hmm_key] = average_metrics(hmm_trial_metrics)

        all_rocs[dtw_key] = dtw_result['roc']
        all_rocs[hmm_key] = hmm_result['roc']
        all_timing[dtw_key] = dtw_result['timing']
        all_timing[hmm_key] = hmm_result['timing']

        # Plots for last trial
        output_dir = config.eval.output_dir
        for name, result, score_key in [
            (dtw_key, dtw_result, 'distances'),
            (hmm_key, hmm_result, 'scores')
        ]:
            safe_name = name.replace(' ', '_').replace('/', '_')
            plot_confusion_matrix(
                result['results'], result['keywords'],
                method_name=name,
                save_path=os.path.join(output_dir, f"cm_{safe_name}.png")
            )
            plot_distance_distributions(
                result['results'], method_name=name,
                score_key=score_key,
                save_path=os.path.join(output_dir, f"dist_{safe_name}.png")
            )
    from evaluation import compute_operating_points, print_operating_points_table

    all_op_points = {}

    for k_shot in config.data.k_shots:
        seed = config.data.seed + config.eval.n_trials - 1  # last trial
        split = create_few_shot_split(config, k_shot, seed)

        # DTW
        dtw_spotter = DTWKeywordSpotter(config)
        dtw_spotter.enroll(split.known_keywords, split.enrollment)
        dtw_spotter.rejection_threshold = float('inf')  
        dtw_results = dtw_spotter.predict_batch(split.test_known, split.test_unknown)
        dtw_op = compute_operating_points(dtw_results, 'distances', 'distance')
        all_op_points[f"DTW ({k_shot}-shot)"] = dtw_op

        # HMM
        hmm_spotter = HMMKeywordSpotter(config)
        hmm_spotter.enroll(split.known_keywords, split.enrollment, split)
        hmm_spotter.rejection_threshold = float('-inf')  
        hmm_results = hmm_spotter.predict_batch(split.test_known, split.test_unknown)
        hmm_op = compute_operating_points(hmm_results, 'scores', 'score')
        all_op_points[f"HMM/GMM ({k_shot}-shot)"] = hmm_op

    print_operating_points_table(all_op_points)

    op_save = {}
    for method, ops in all_op_points.items():
        op_save[method] = {
            k: {kk: float(vv) for kk, vv in v.items()} 
            for k, v in ops.items()
        }
    with open(os.path.join(config.eval.output_dir, "operating_points.json"), 'w') as f:
        json.dump(op_save, f, indent=2)

    print_results_table(all_metrics, all_timing)

    plot_roc(
        all_rocs,
        save_path=os.path.join(config.eval.output_dir, "roc_comparison.png")
    )

    save_results = {
        key: {k: float(v) for k, v in metrics.items()}
        for key, metrics in all_metrics.items()
    }
    with open(os.path.join(config.eval.output_dir, "results.json"), 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nAll results saved to {config.eval.output_dir}/")
    return all_metrics, all_rocs, all_timing


def main():
    parser = argparse.ArgumentParser(description="Run KWS baselines")
    parser.add_argument('--gsc_root', type=str,
                        default="./data/SpeechCommands/speech_commands_v0.02")
    parser.add_argument('--k_shots', nargs='+', type=int, default=[5, 10])
    parser.add_argument('--n_trials', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default="./results")
    parser.add_argument('--download', action='store_true')

    args = parser.parse_args()

    config = Config()
    config.data.gsc_root = args.gsc_root
    config.data.k_shots = args.k_shots
    config.eval.n_trials = args.n_trials
    config.eval.output_dir = args.output_dir

    if args.download:
        download_gsc("./data")

    run_all_experiments(config)


if __name__ == "__main__":
    main()