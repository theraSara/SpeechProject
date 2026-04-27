import csv
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns



def compute_metrics(results: Dict) -> Dict:
    """
    Compute all required evaluation metrics.
    
    Metrics:
      - Known keyword accuracy (among known test samples)
      - Unknown rejection accuracy (among unknown test samples)
      - False Acceptance Rate (FAR): unknown classified as any known keyword
      - False Rejection Rate (FRR): known keyword classified as unknown
      - False Positive Rate between known classes (known-to-known confusion)
      - Overall accuracy
    """
    predictions = np.array(results['predictions'])
    true_labels = np.array(results['true_labels'])
    is_known = np.array(results['is_known'])
    
    metrics = {}
    
    # --- Known keyword accuracy ---
    # Among samples that are truly known keywords:
    # what fraction are correctly classified (to the RIGHT keyword)?
    known_mask = is_known
    if known_mask.sum() > 0:
        known_preds = predictions[known_mask]
        known_true = true_labels[known_mask]
        
        # Correct if predicted == true label
        known_correct = (known_preds == known_true).sum()
        metrics['known_accuracy'] = known_correct / known_mask.sum()
        
        # Known classified as unknown (false rejection)
        known_rejected = (known_preds == "unknown").sum()
        metrics['false_rejection_rate'] = known_rejected / known_mask.sum()
        
        # Known-to-known confusion: classified as wrong known keyword
        known_confused = (
            (known_preds != known_true) & (known_preds != "unknown")
        ).sum()
        metrics['known_confusion_rate'] = known_confused / known_mask.sum()
    
    # --- Unknown rejection accuracy ---
    # Among samples that are truly unknown:
    # what fraction are correctly rejected?
    unknown_mask = ~is_known
    if unknown_mask.sum() > 0:
        unknown_preds = predictions[unknown_mask]
        
        # Correctly rejected
        unknown_rejected = (unknown_preds == "unknown").sum()
        metrics['unknown_rejection_accuracy'] = unknown_rejected / unknown_mask.sum()
        
        # False acceptance: unknown classified as any known keyword
        false_accepted = (unknown_preds != "unknown").sum()
        metrics['false_acceptance_rate'] = false_accepted / unknown_mask.sum()
    
    # --- Overall accuracy ---
    # Correct = known correctly classified + unknown correctly rejected
    correct = 0
    correct += ((predictions[known_mask] == true_labels[known_mask]).sum() 
                if known_mask.sum() > 0 else 0)
    correct += ((predictions[unknown_mask] == "unknown").sum() 
                if unknown_mask.sum() > 0 else 0)
    metrics['overall_accuracy'] = correct / len(predictions)
    
    # --- Balanced accuracy (average of known acc and unknown rejection) ---
    metrics['balanced_accuracy'] = (
        metrics.get('known_accuracy', 0) + 
        metrics.get('unknown_rejection_accuracy', 0)
    ) / 2
    
    return metrics


def compute_roc(
    results: Dict, 
    score_key: str = 'distances',
    score_type: str = 'distance'  # 'distance' (lower=known) or 'score' (higher=known)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve for open-set detection.
    
    Binary task: known (positive) vs unknown (negative)
    
    Returns:
        fpr, tpr, thresholds, auc_score
    """
    scores = np.array(results[score_key], dtype=float)
    is_known = np.array(results['is_known'])
    
    # For ROC, we need: higher score = more likely known
    if score_type == 'distance':
        # Distance: lower = more likely known, so negate
        scores = -scores
    
    fpr, tpr, thresholds = roc_curve(is_known.astype(int), scores)
    auc_score = auc(fpr, tpr)
    
    return fpr, tpr, thresholds, auc_score


def plot_roc(
    roc_data: Dict[str, Tuple],
    save_path: str = None
):
    """
    Plot ROC curves for multiple methods.
    
    Args:
        roc_data: method_name -> (fpr, tpr, auc_score)
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    for name, (fpr, tpr, auc_score) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve: Known vs Unknown Detection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC plot saved to {save_path}")
    
    return fig


def plot_confusion_matrix(
    results: Dict,
    keywords: List[str],
    method_name: str = "",
    save_path: str = None
):
    """Plot confusion matrix including 'unknown' class."""
    all_labels = keywords + ["unknown"]
    predictions = results['predictions']
    true_labels = results['true_labels']
    
    cm = confusion_matrix(true_labels, predictions, labels=all_labels)
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_normalized, annot=True, fmt='.2f',
        xticklabels=all_labels, yticklabels=all_labels,
        cmap='Blues', ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix - {method_name}')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_distance_distributions(
    results: Dict,
    method_name: str = "",
    score_key: str = 'distances',
    save_path: str = None
):
    """
    Plot score/distance distributions for known vs unknown.
    Helps visualize how separable they are.
    """
    scores = np.array(results[score_key], dtype=float)
    is_known = np.array(results['is_known'])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.hist(scores[is_known], bins=50, alpha=0.6, label='Known', density=True)
    ax.hist(scores[~is_known], bins=50, alpha=0.6, label='Unknown', density=True)
    ax.set_xlabel('Score / Distance')
    ax.set_ylabel('Density')
    ax.set_title(f'Score Distribution - {method_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def find_threshold_for_target_far(
    known_values: np.ndarray,
    unknown_values: np.ndarray,
    target_far: float,
    score_type: str = "distance",
) -> Dict[str, float]:
    """
    Choose the most permissive threshold that still satisfies FAR <= target_far.
    For DTW:
        accept if distance <= threshold
    For HMM:
        accept if score >= threshold
    """
    known = np.asarray(known_values, dtype=float)
    unknown = np.asarray(unknown_values, dtype=float)

    if known.size == 0 or unknown.size == 0:
        raise ValueError("known_values and unknown_values must be non-empty")

    candidates = np.unique(np.concatenate([known, unknown]))
    eps = 1e-8

    if score_type == "distance":
        candidates = np.concatenate(([candidates.min() - eps], candidates, [candidates.max() + eps]))
        far_fn = lambda t: float(np.mean(unknown <= t))
        known_accept_fn = lambda t: float(np.mean(known <= t))
    elif score_type == "score":
        candidates = np.concatenate(([candidates.max() + eps], candidates[::-1], [candidates.min() - eps]))
        far_fn = lambda t: float(np.mean(unknown >= t))
        known_accept_fn = lambda t: float(np.mean(known >= t))
    else:
        raise ValueError(f"Unknown score_type: {score_type}")

    best = None
    for t in candidates:
        far = far_fn(t)
        known_accept = known_accept_fn(t)
        record = {
            "threshold": float(t),
            "target_far": float(target_far),
            "far": float(far),
            "frr": float(1.0 - known_accept),
            "known_accept_rate": float(known_accept),
            "unknown_reject_rate": float(1.0 - far),
            "balanced_acc": float((known_accept + (1.0 - far)) / 2.0),
        }

        if far <= target_far + 1e-12:
            if best is None:
                best = record
            else:
                better_known = record["known_accept_rate"] > best["known_accept_rate"] + 1e-12
                equal_known = abs(record["known_accept_rate"] - best["known_accept_rate"]) <= 1e-12
                better_far = record["far"] < best["far"] - 1e-12
                if better_known or (equal_known and better_far):
                    best = record

    if best is None:
        # conservative fallback
        t = float(candidates[0])
        far = far_fn(t)
        known_accept = known_accept_fn(t)
        best = {
            "threshold": t,
            "target_far": float(target_far),
            "far": float(far),
            "frr": float(1.0 - known_accept),
            "known_accept_rate": float(known_accept),
            "unknown_reject_rate": float(1.0 - far),
            "balanced_acc": float((known_accept + (1.0 - far)) / 2.0),
        }

    return best

def apply_threshold_to_results(
    results: Dict,
    threshold: float,
    score_key: str = "distances",
    score_type: str = "distance",
) -> Dict:
    """
    Take raw predictions (with rejection disabled) and apply a threshold afterward.
    """
    out = {}
    for k, v in results.items():
        if isinstance(v, list):
            out[k] = list(v)
        else:
            out[k] = v

    scores = np.array(results[score_key], dtype=float)
    base_preds = np.array(results["predictions"], dtype=object)

    if score_type == "distance":
        accepted = scores <= threshold
    elif score_type == "score":
        accepted = scores >= threshold
    else:
        raise ValueError(f"Unknown score_type: {score_type}")

    final_preds = np.where(accepted, base_preds, "unknown")

    out["predictions"] = final_preds.tolist()
    out["accepted"] = accepted.tolist()
    out["threshold"] = float(threshold)
    return out


def compute_fixed_far_metrics(
    results: Dict,
    target_far: float = 0.05,
    score_key: str = "distances",
    score_type: str = "distance",
) -> Dict:
    """
    Compute:
      - ACC+_5%  -> known accuracy at threshold fixed by FAR <= 5%
      - AUROC    -> ROC AUC for known vs unknown
      - FRR+_5%  -> false rejection rate at the same threshold
    """
    scores = np.array(results[score_key], dtype=float)
    is_known = np.array(results["is_known"], dtype=bool)
    true_labels = np.array(results["true_labels"], dtype=object)

    if is_known.sum() == 0 or (~is_known).sum() == 0:
        raise ValueError("Need both known and unknown samples to compute fixed-FAR metrics.")

    # Threshold chosen from positives + negatives under FAR constraint
    stats = find_threshold_for_target_far(
        known_values=scores[is_known],
        unknown_values=scores[~is_known],
        target_far=target_far,
        score_type=score_type,
    )

    threshold = float(stats["threshold"])
    thresholded = apply_threshold_to_results(
        results,
        threshold=threshold,
        score_key=score_key,
        score_type=score_type,
    )

    preds = np.array(thresholded["predictions"], dtype=object)

    known_mask = is_known
    unknown_mask = ~is_known

    acc_plus_5 = float(np.mean(preds[known_mask] == true_labels[known_mask]))
    frr_plus_5 = float(np.mean(preds[known_mask] == "unknown"))
    far_at_5 = float(np.mean(preds[unknown_mask] != "unknown"))

    _, _, _, auroc = compute_roc(results, score_key=score_key, score_type=score_type)

    return {
        "acc_plus_5": acc_plus_5,
        "auroc": float(auroc),
        "frr_plus_5": frr_plus_5,
        "far_at_5": far_at_5,
        "threshold_5": threshold,
        "thresholded_results": thresholded,
    }
def print_results_table(
    all_results: Dict[str, Dict],
    timing_info: Dict[str, Dict] = None
):
    has_official_metrics = any(
        ("acc_plus_5" in metrics or "auroc" in metrics or "frr_plus_5" in metrics)
        for metrics in all_results.values()
    )

    if has_official_metrics:
        print("\n" + "=" * 110)
        print("RESULTS COMPARISON (OFFICIAL METRICS)")
        print("=" * 110)

        header = (
            f"{'Method':<25} {'ACC+_5%':>10} {'AUROC':>10} "
            f"{'FRR+_5%':>10} {'FAR@thr':>10} {'Thr@5%':>12}"
        )
        print(header)
        print("-" * 110)

        for method, metrics in all_results.items():
            row = (
                f"{method:<25} "
                f"{metrics.get('acc_plus_5', metrics.get('known_accuracy', 0)) * 100:>9.1f}% "
                f"{metrics.get('auroc', metrics.get('auc', 0)):>9.3f} "
                f"{metrics.get('frr_plus_5', metrics.get('false_rejection_rate', 0)) * 100:>9.1f}% "
                f"{metrics.get('far_at_5', metrics.get('false_acceptance_rate', 0)) * 100:>9.1f}% "
                f"{metrics.get('threshold_5', metrics.get('threshold', 0)):>11.4f}"
            )
            print(row)
    else:
        print("\n" + "=" * 108)
        print("RESULTS COMPARISON")
        print("=" * 108)

        header = (
            f"{'Method':<25} {'Known Acc':>10} {'Unk Rej':>10} "
            f"{'FAR':>8} {'FRR':>8} {'Bal Acc':>10} {'AUC':>8} {'Val FAR':>9}"
        )
        print(header)
        print("-" * 108)

        for method, metrics in all_results.items():
            row = (
                f"{method:<25} "
                f"{metrics.get('known_accuracy', 0) * 100:>9.1f}% "
                f"{metrics.get('unknown_rejection_accuracy', 0) * 100:>9.1f}% "
                f"{metrics.get('false_acceptance_rate', 0) * 100:>7.1f}% "
                f"{metrics.get('false_rejection_rate', 0) * 100:>7.1f}% "
                f"{metrics.get('balanced_accuracy', 0) * 100:>9.1f}% "
                f"{metrics.get('auc', 0):>7.3f} "
                f"{metrics.get('val_far', 0) * 100:>8.1f}%"
            )
            print(row)

    if timing_info:
        print("\n" + "-" * 110)
        print(f"{'Method':<25} {'Params':>12} {'Storage (MB)':>14} {'Inference (ms)':>15}")
        print("-" * 110)
        for method, info in timing_info.items():
            storage_mb = info.get("storage_bytes", 0) / (1024 ** 2)
            print(
                f"{method:<25} {info.get('model_params', 0):>12} "
                f"{storage_mb:>13.3f} {info.get('mean_inference_time_ms', 0):>14.1f}"
            )

    print("=" * 110)

def compute_operating_points(results: Dict, score_key: str = 'distances',
                              score_type: str = 'distance') -> Dict:
    """
    Compute metrics at several meaningful operating points.
    More informative than a single threshold.
    """
    scores = np.array(results[score_key], dtype=float)
    true_labels = np.array(results['true_labels'])
    predictions = np.array(results['predictions'])
    is_known = np.array(results['is_known'])

    if score_type == 'distance':
        # Lower distance = more likely known
        confidence = -scores
    else:
        confidence = scores

    known_conf = confidence[is_known]
    unknown_conf = confidence[~is_known]

    operating_points = {}

    # Sweep thresholds and find specific operating points
    all_conf = np.concatenate([known_conf, unknown_conf])
    thresholds = np.linspace(np.min(all_conf), np.max(all_conf), 500)

    records = []
    for t in thresholds:
        known_accepted = np.mean(known_conf >= t)  # True positive rate
        unknown_rejected = np.mean(unknown_conf < t)  # True negative rate
        far = 1 - unknown_rejected
        frr = 1 - known_accepted

        # Known accuracy among accepted known samples
        known_mask_accepted = is_known & (confidence >= t)
        if known_mask_accepted.sum() > 0:
            # Among known samples that pass threshold, how many are correct?
            correct_known = (predictions[known_mask_accepted] ==
                           true_labels[known_mask_accepted]).sum()
            known_acc_at_threshold = correct_known / is_known.sum()
        else:
            known_acc_at_threshold = 0

        records.append({
            'threshold': t,
            'known_accept_rate': known_accepted,
            'unknown_reject_rate': unknown_rejected,
            'far': far,
            'frr': frr,
            'balanced_acc': (known_accepted + unknown_rejected) / 2,
            'known_acc': known_acc_at_threshold,
        })

    records = sorted(records, key=lambda x: x['balanced_acc'], reverse=True)

    # Extract key operating points
    import pandas as pd
    df = pd.DataFrame(records)

    # 1. Best balanced accuracy
    best_bal = df.iloc[0]
    operating_points['best_balanced'] = best_bal.to_dict()

    # 2. FAR = 10% (allow 10% of unknowns through)
    close_to_far10 = df.iloc[(df['far'] - 0.10).abs().argsort()[:1]]
    operating_points['far_10pct'] = close_to_far10.iloc[0].to_dict()

    # 3. FAR = 20%
    close_to_far20 = df.iloc[(df['far'] - 0.20).abs().argsort()[:1]]
    operating_points['far_20pct'] = close_to_far20.iloc[0].to_dict()

    # 4. FAR = 50%
    close_to_far50 = df.iloc[(df['far'] - 0.50).abs().argsort()[:1]]
    operating_points['far_50pct'] = close_to_far50.iloc[0].to_dict()

    # 5. EER (Equal Error Rate: FAR ≈ FRR)
    df['eer_diff'] = (df['far'] - df['frr']).abs()
    eer_row = df.iloc[df['eer_diff'].argsort()[:1]]
    operating_points['eer'] = eer_row.iloc[0].to_dict()

    return operating_points


def print_operating_points_table(all_op_points: Dict[str, Dict]):
    """Print a nice table of operating points for all methods."""
    print("\n" + "=" * 90)
    print("OPERATING POINTS ANALYSIS")
    print("=" * 90)

    for method, op_points in all_op_points.items():
        print(f"\n{'─'*90}")
        print(f"  {method}")
        print(f"{'─'*90}")
        header = f"  {'Operating Point':<20} {'Known Acc':>10} {'Unk Rej':>10} {'FAR':>8} {'FRR':>8} {'Bal Acc':>10}"
        print(header)
        print(f"  {'-'*66}")

        point_names = {
            'best_balanced': 'Best Balanced',
            'eer': 'EER',
            'far_10pct': 'FAR ≈ 10%',
            'far_20pct': 'FAR ≈ 20%',
            'far_50pct': 'FAR ≈ 50%',
        }

        for key, name in point_names.items():
            if key in op_points:
                p = op_points[key]
                print(f"  {name:<20} "
                      f"{p.get('known_accept_rate', 0)*100:>9.1f}% "
                      f"{p.get('unknown_reject_rate', 0)*100:>9.1f}% "
                      f"{p.get('far', 0)*100:>7.1f}% "
                      f"{p.get('frr', 0)*100:>7.1f}% "
                      f"{p.get('balanced_acc', 0)*100:>9.1f}%")

    print("=" * 90)
    
def save_metrics_csv(
    all_results: Dict[str, Dict],
    timing_info: Dict[str, Dict] = None,
    save_path: str = "./results/baseline_results_table.csv"
):
    rows = []
    for method, metrics in all_results.items():
        row = {
            "method": method,
            "known_accuracy": metrics.get("known_accuracy", 0.0),
            "unknown_rejection_accuracy": metrics.get("unknown_rejection_accuracy", 0.0),
            "false_acceptance_rate": metrics.get("false_acceptance_rate", 0.0),
            "false_rejection_rate": metrics.get("false_rejection_rate", 0.0),
            "known_confusion_rate": metrics.get("known_confusion_rate", 0.0),
            "overall_accuracy": metrics.get("overall_accuracy", 0.0),
            "balanced_accuracy": metrics.get("balanced_accuracy", 0.0),
            "auc": metrics.get("auc", 0.0),
            "auroc": metrics.get("auroc", metrics.get("auc", 0.0)),
            "acc_plus_5": metrics.get("acc_plus_5", metrics.get("known_accuracy", 0.0)),
            "frr_plus_5": metrics.get("frr_plus_5", metrics.get("false_rejection_rate", 0.0)),
            "far_at_5": metrics.get("far_at_5", metrics.get("false_acceptance_rate", 0.0)),
            "threshold": metrics.get("threshold", 0.0),
            "threshold_5": metrics.get("threshold_5", metrics.get("threshold", 0.0)),
            "val_far": metrics.get("val_far", 0.0),
            "val_frr": metrics.get("val_frr", 0.0),
        }
        if timing_info and method in timing_info:
            row["mean_inference_time_ms"] = timing_info[method].get("mean_inference_time_ms", 0.0)
            row["std_inference_time_ms"] = timing_info[method].get("std_inference_time_ms", 0.0)
            row["model_params"] = timing_info[method].get("model_params", 0)
            row["storage_bytes"] = timing_info[method].get("storage_bytes", 0)
        rows.append(row)

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results table saved to {save_path}")