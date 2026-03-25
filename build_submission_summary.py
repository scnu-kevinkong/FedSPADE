import csv
import glob
import os
import argparse
from typing import Dict, List

import numpy as np


def latest_mechanism_npz(run_dir: str, target_round: int = None):
    files = sorted(glob.glob(os.path.join(run_dir, "mechanism", "round_*.npz")))
    if not files:
        return None
    if target_round is None:
        return np.load(files[-1], allow_pickle=False)
    chosen = None
    chosen_round = -1
    for fp in files:
        base = os.path.basename(fp)
        try:
            r = int(base.replace("round_", "").replace(".npz", ""))
        except ValueError:
            continue
        if r <= target_round and r > chosen_round:
            chosen = fp
            chosen_round = r
    if chosen is None:
        return None
    return np.load(chosen, allow_pickle=False)


def resolve_method_run(candidates: List[str], target_round: int = None) -> str:
    best_path = ""
    best_round = -1
    for path in candidates:
        files = sorted(glob.glob(os.path.join(path, "mechanism", "round_*.npz")))
        if not files:
            continue
        if target_round is None:
            last_name = os.path.basename(files[-1])
            try:
                r = int(last_name.replace("round_", "").replace(".npz", ""))
            except ValueError:
                r = -1
        else:
            r = -1
            for f in files:
                try:
                    rr = int(os.path.basename(f).replace("round_", "").replace(".npz", ""))
                except ValueError:
                    continue
                if rr <= target_round and rr > r:
                    r = rr
        if r > best_round:
            best_round = r
            best_path = path
    return best_path


def safe_mean(arr):
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(arr.mean())


def summarize_method(run_dir: str, target_round: int = None) -> Dict[str, float]:
    d = latest_mechanism_npz(run_dir, target_round=target_round)
    if d is None:
        return {}
    return {
        "source_run": os.path.basename(run_dir),
        "round": int(d["round"][0]),
        "ece_mean": safe_mean(d["ece_client"]),
        "entropy_gap_mean": safe_mean(d["entropy_gap"]),
        "curvature_probe": safe_mean(d["curvature"][d["probe_mask"].astype(bool)]),
        "alpha_unseen_probe": safe_mean(d["alpha_unseen_mean"][d["probe_mask"].astype(bool)]),
        "collapse_probe": safe_mean(d["collapse_rate"][d["probe_mask"].astype(bool)]),
    }


def prior_error(records, key):
    oracle = records["beta_oracle_benign"]
    est = records[key]
    return float(np.nanmean(np.abs(est - oracle).sum(axis=1)))


def summarize_byz(run_dir: str, target_round: int = None) -> Dict[str, float]:
    fp = os.path.join(run_dir, "byzantine", "records.npz")
    d = np.load(fp, allow_pickle=False)
    idx = d["rounds"].shape[0] - 1
    if target_round is not None:
        valid = np.where(d["rounds"] <= target_round)[0]
        if valid.size == 0:
            return {}
        idx = int(valid[-1])
    return {
        "ratio": float(np.nanmean(d["ratio"])),
        "rounds": int(d["rounds"][idx]),
        "ece_last": float(d["ece_mean"][idx]),
        "auroc_last": float(d["auroc_epistemic"][idx]),
        "r_beta_weighted": prior_error(d, "beta_weighted"),
        "r_beta_trimmed": prior_error(d, "beta_trimmed"),
        "r_beta_median": prior_error(d, "beta_median"),
        "r_beta_geomedian": prior_error(d, "beta_geomedian"),
        "r_beta_krum": prior_error(d, "beta_krum"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-round", type=int, default=None)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    method_run_candidates = {
        "Full": [
            "results/cifar10_FedUgV2_0.5_full_topsec200",
            "results/cifar10_FedUgV2_0.5_full_topsec_gpu",
        ],
        "NoFSA": [
            "results/cifar10_FedUgV2_0.5_nofsa_topsec200",
            "results/cifar10_FedUgV2_0.5_nofsa_topsec_gpu",
        ],
        "NoGC": [
            "results/cifar10_FedUgV2_0.5_nogc_topsec200",
            "results/cifar10_FedUgV2_0.5_nogc_topsec_gpu",
        ],
    }
    byz_runs = [
        "results/cifar10_FedUgV2_0.5_byz_r00_topsec200",
        "results/cifar10_FedUgV2_0.5_byz_r01_topsec200",
        "results/cifar10_FedUgV2_0.5_byz_r02_topsec200",
        "results/cifar10_FedUgV2_0.5_byz_r03_topsec200",
    ]

    out_dir = "results/submission_summary"
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    method_csv = os.path.join(out_dir, f"rq_method_summary{suffix}.csv")
    byz_csv = os.path.join(out_dir, f"rq_byzantine_summary{suffix}.csv")
    md_path = os.path.join(out_dir, f"topsec_submission_summary{suffix}.md")

    method_rows: List[Dict[str, float]] = []
    for name, candidates in method_run_candidates.items():
        path = resolve_method_run(candidates, target_round=args.target_round)
        if not path:
            continue
        row = summarize_method(path, target_round=args.target_round)
        if row:
            row["method"] = name
            method_rows.append(row)

    byz_rows: List[Dict[str, float]] = []
    for path in byz_runs:
        if os.path.exists(os.path.join(path, "byzantine", "records.npz")):
            row = summarize_byz(path, target_round=args.target_round)
            if not row:
                continue
            row["run"] = os.path.basename(path)
            byz_rows.append(row)

    with open(method_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "source_run", "round", "ece_mean", "entropy_gap_mean", "curvature_probe", "alpha_unseen_probe", "collapse_probe"],
        )
        writer.writeheader()
        writer.writerows(method_rows)

    with open(byz_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "ratio",
                "rounds",
                "ece_last",
                "auroc_last",
                "r_beta_weighted",
                "r_beta_trimmed",
                "r_beta_median",
                "r_beta_geomedian",
                "r_beta_krum",
            ],
        )
        writer.writeheader()
        writer.writerows(byz_rows)

    lines = []
    title = "TopSec Submission Summary"
    if args.target_round is not None:
        title += f" (target round <= {args.target_round})"
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## RQ1/RQ2 Core Methods")
    lines.append("")
    lines.append("| Method | Source Run | Round | Mean ECE | Mean Entropy Gap | Probe Curvature | Probe Unseen Alpha | Probe Collapse |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in method_rows:
        lines.append(
            f"| {r['method']} | {r['source_run']} | {r['round']} | {r['ece_mean']:.4f} | {r['entropy_gap_mean']:.4f} | {r['curvature_probe']:.6f} | {r['alpha_unseen_probe']:.4f} | {r['collapse_probe']:.4f} |"
        )
    lines.append("")
    lines.append("## RQ4 Byzantine Robustness")
    lines.append("")
    lines.append("| Run | Ratio | Logged Rounds | Last ECE | Last AUROC | Rβ Weighted | Rβ Trimmed | Rβ Median | Rβ Geomedian | Rβ Krum |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(byz_rows, key=lambda x: x["ratio"]):
        lines.append(
            f"| {r['run']} | {r['ratio']:.2f} | {r['rounds']} | {r['ece_last']:.4f} | {r['auroc_last']:.4f} | {r['r_beta_weighted']:.4f} | {r['r_beta_trimmed']:.4f} | {r['r_beta_median']:.4f} | {r['r_beta_geomedian']:.4f} | {r['r_beta_krum']:.4f} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(method_csv)
    print(byz_csv)
    print(md_path)


if __name__ == "__main__":
    main()
