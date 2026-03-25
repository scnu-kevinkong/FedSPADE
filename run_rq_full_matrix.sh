#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=${DATA_PATH:-/home/gyxc_linux/fedcure_local/pFedFDA-main/data}
export DATA_PATH

COMMON_ARGS=(
  --method FedUgV2
  --dataset cifar10
  --num_classes 10
  --partition_path cifar10_c100_dir01_1
  --num_clients 100
  --sampling_prob 0.1
  --global_rounds 200
  --local_epochs 5
  --eval_gap 5
  --batch_size 64
  --lr 0.01
  --wd 5e-4
  --model_name cnn
  --uncertainty
  --ood_dataset svhn
  --device gpu
  --mechanism_log_gap 5
  --use_byzantine_robust
  --robust_prior_method coord_median
)

python main.py --exp_name full_real_15r "${COMMON_ARGS[@]}" --use_fsa true  --byzantine_ratio 0.0
python main.py --exp_name nofsa_real_15r "${COMMON_ARGS[@]}" --use_fsa false --byzantine_ratio 0.0
python main.py --exp_name nogc_real_15r "${COMMON_ARGS[@]}" --use_fsa true  --global_consistency_weight 0.0 --byzantine_ratio 0.0

python main.py --exp_name byz_r00_15r "${COMMON_ARGS[@]}" --use_fsa true --byzantine_ratio 0.0
python main.py --exp_name byz_r01_15r "${COMMON_ARGS[@]}" --use_fsa true --byzantine_ratio 0.1 --byzantine_attack_type gaussian_noise
python main.py --exp_name byz_r02_15r "${COMMON_ARGS[@]}" --use_fsa true --byzantine_ratio 0.2 --byzantine_attack_type gaussian_noise
python main.py --exp_name byz_r03_15r "${COMMON_ARGS[@]}" --use_fsa true --byzantine_ratio 0.3 --byzantine_attack_type gaussian_noise

python plot_rq_core_assets_real.py \
  --run Full=results/cifar10_FedUgV2_0.5_full_real_15r \
  --run NoFSA=results/cifar10_FedUgV2_0.5_nofsa_real_15r \
  --run NoGC=results/cifar10_FedUgV2_0.5_nogc_real_15r \
  --r2-no-fsa results/cifar10_FedUgV2_0.5_nofsa_real_15r \
  --r2-full results/cifar10_FedUgV2_0.5_full_real_15r \
  --byz-run results/cifar10_FedUgV2_0.5_byz_r00_15r \
  --byz-run results/cifar10_FedUgV2_0.5_byz_r01_15r \
  --byz-run results/cifar10_FedUgV2_0.5_byz_r02_15r \
  --byz-run results/cifar10_FedUgV2_0.5_byz_r03_15r \
  --out-dir fig

echo "All experiments and figure generation completed."
