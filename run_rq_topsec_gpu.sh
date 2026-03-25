#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=${DATA_PATH:-/home/gyxc_linux/fedcure_local/pFedFDA-main/data}
export DATA_PATH

DEVICE=${DEVICE:-cuda:0}

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
  --lr 0.005
  --wd 5e-4
  --model_name cnn
  --uncertainty
  --ood_dataset svhn
  --device "${DEVICE}"
  --mechanism_log_gap 5
  --use_byzantine_robust
  --robust_prior_method coord_median
  --temperature 2.0
  --max_evidence 5.0
  --edl_weight 0.1
  --personal_lr_factor 0.1
  --feature_weight 0.4
  --global_consistency_weight 0.5
  --use_adv_ood
  --adv_ood_reg_weight 0.3
  --use_temperature_ensemble false
)

python main.py --exp_name full_topsec_gpu  "${COMMON_ARGS[@]}" --use_fsa true  --byzantine_ratio 0.0
python main.py --exp_name nofsa_topsec_gpu "${COMMON_ARGS[@]}" --use_fsa false --byzantine_ratio 0.0
python main.py --exp_name nogc_topsec_gpu  "${COMMON_ARGS[@]}" --use_fsa true  --global_consistency_weight 0.0 --byzantine_ratio 0.0

python main.py --exp_name byz_r00_topsec_gpu "${COMMON_ARGS[@]}" --use_fsa true --byzantine_ratio 0.0 --byzantine_attack_type gaussian_noise
python main.py --exp_name byz_r01_topsec_gpu "${COMMON_ARGS[@]}" --use_fsa true --byzantine_ratio 0.1 --byzantine_attack_type gaussian_noise
python main.py --exp_name byz_r02_topsec_gpu "${COMMON_ARGS[@]}" --use_fsa true --byzantine_ratio 0.2 --byzantine_attack_type gaussian_noise
python main.py --exp_name byz_r03_topsec_gpu "${COMMON_ARGS[@]}" --use_fsa true --byzantine_ratio 0.3 --byzantine_attack_type gaussian_noise

python plot_rq_core_assets_real.py \
  --run Full=results/cifar10_FedUgV2_0.5_full_topsec_gpu \
  --run NoFSA=results/cifar10_FedUgV2_0.5_nofsa_topsec_gpu \
  --run NoGC=results/cifar10_FedUgV2_0.5_nogc_topsec_gpu \
  --r2-no-fsa results/cifar10_FedUgV2_0.5_nofsa_topsec_gpu \
  --r2-full results/cifar10_FedUgV2_0.5_full_topsec_gpu \
  --byz-run results/cifar10_FedUgV2_0.5_byz_r00_topsec_gpu \
  --byz-run results/cifar10_FedUgV2_0.5_byz_r01_topsec_gpu \
  --byz-run results/cifar10_FedUgV2_0.5_byz_r02_topsec_gpu \
  --byz-run results/cifar10_FedUgV2_0.5_byz_r03_topsec_gpu \
  --out-dir fig

echo "TopSec RQ experiments complete."
