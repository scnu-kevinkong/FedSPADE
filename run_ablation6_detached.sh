#!/usr/bin/env bash
set -euo pipefail

cd /home/gyxc_linux/fedcure_local/pFedFDA-main
export DATA_PATH="${DATA_PATH:-/home/gyxc_linux/fedcure_local/pFedFDA-main/data}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/home/gyxc_linux/fedcure_local/pFedFDA-main/.mplconfig}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/home/gyxc_linux/fedcure_local/pFedFDA-main/.torchinductor_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/home/gyxc_linux/fedcure_local/pFedFDA-main/.cache}"

target_round=200
method_name="FedUg"
common_args=(
  --method "${method_name}"
  --dataset cifar10
  --num_classes 10
  --partition_path cifar10_c100_dir01
  --num_clients 100
  --sampling_prob 0.3
  --global_rounds 200
  --local_epochs 5
  --eval_gap 10
  --batch_size 64
  --lr 0.005
  --wd 5e-4
  --model_name cnn
  --uncertainty
  --ood_dataset svhn
  --use_global_prior_regularization true
  --use_ood_reg true
  --use_data_augmentation true
  --edl_kl_global_prior_lambda 1.0
  --edl_kl_lambda_personalization 0.5
  --lr_personalization 0.001
  --personalization_epochs_on_eval 5
  --mechanism_profile_clients_per_eval 20
  --mechanism_hessian_max_batches 1
  --mechanism_hessian_max_samples_per_batch 1
  --byzantine_ratio 0.0
)

is_done() {
  local exp_name="$1"
  python - "$exp_name" "$target_round" <<'PY'
import os,sys,re
exp_name=sys.argv[1]
target=int(sys.argv[2])
log_path=os.path.join("results","submission_summary",f"{exp_name}.log")
if not os.path.exists(log_path):
    print("0")
    raise SystemExit(0)
pat=re.compile(r"Evaluating models at round\s+(\d+)")
last=0
with open(log_path,"r",encoding="utf-8",errors="ignore") as f:
    for line in f:
        m=pat.search(line)
        if m:
            last=max(last,int(m.group(1)))
print("1" if last>=target else "0")
PY
}

run_variant() {
  local exp_name="$1"
  shift
  local done
  done="$(is_done "$exp_name")"
  if [[ "$done" == "1" ]]; then
    echo "[SKIP] ${exp_name} already reached round ${target_round}"
    return 0
  fi
  echo "[RUN] ${exp_name}"
  python main.py --exp_name "${exp_name}" "${common_args[@]}" "$@"
}

launch_variant() {
  local gpu="$1"
  local exp_name="$2"
  shift 2
  local done
  done="$(is_done "$exp_name")"
  if [[ "$done" == "1" ]]; then
    echo "[SKIP] ${exp_name} already reached round ${target_round}"
    return
  fi
  local log_file="results/submission_summary/${exp_name}.log"
  echo "[LAUNCH][GPU${gpu}] ${exp_name}"
  CUDA_VISIBLE_DEVICES="${gpu}" python main.py --device cuda:0 --exp_name "${exp_name}" "${common_args[@]}" "$@" > "${log_file}" 2>&1 &
  pids+=("$!")
  names+=("${exp_name}")
}

wait_all_current() {
  for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    name="${names[$i]}"
    if wait "${pid}"; then
      echo "[DONE] ${name}"
    else
      echo "[FAIL] ${name}"
    fi
  done
}

pids=()
names=()
launch_variant 0 ablation7_gpr_only_200r --use_global_prior_regularization true --use_ood_reg false --use_data_augmentation false
launch_variant 0 ablation7_ood_only_200r --use_global_prior_regularization false --use_ood_reg true --use_data_augmentation false
launch_variant 0 ablation7_aug_only_200r --use_global_prior_regularization false --use_ood_reg false --use_data_augmentation true
launch_variant 1 ablation7_gpr_ood_200r --use_global_prior_regularization true --use_ood_reg true --use_data_augmentation false
launch_variant 1 ablation7_gpr_aug_200r --use_global_prior_regularization true --use_ood_reg false --use_data_augmentation true
launch_variant 1 ablation7_ood_aug_200r --use_global_prior_regularization false --use_ood_reg true --use_data_augmentation true
launch_variant 1 ablation7_fedug_full_200r --use_global_prior_regularization true --use_ood_reg true --use_data_augmentation true
wait_all_current

python mechanism_study_controlled.py --export_ablation_table --ablation_output_dir results/submission_summary --ablation_runs \
  "GPR only=results/submission_summary/ablation7_gpr_only_200r.log" \
  "OOD only=results/submission_summary/ablation7_ood_only_200r.log" \
  "AUG only=results/submission_summary/ablation7_aug_only_200r.log" \
  "GPR + OOD=results/submission_summary/ablation7_gpr_ood_200r.log" \
  "GPR + AUG=results/submission_summary/ablation7_gpr_aug_200r.log" \
  "OOD + AUG=results/submission_summary/ablation7_ood_aug_200r.log" \
  "FedUg Full=results/submission_summary/ablation7_fedug_full_200r.log"

echo "[DONE] ablation7 and table export completed"
