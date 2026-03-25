#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# FedUG V2 敏感性分析实验
mkdir -p sensitivity_logs
export DATA_PATH="/home/xiongzc/Desktop/pFedFDA-main/data"

# 添加最小批量大小参数，避免维度不匹配问题
MIN_BATCH_SIZE=2

# 创建结果目录
RESULTS_DIR="results/sensitivity_analysis"
mkdir -p ${RESULTS_DIR}

# 1. 全局一致性权重敏感性分析
echo "===== 分析全局一致性权重(GCW)的影响 ====="
for GCW in 0.0 0.5 1.0 2.0
do
    echo "Running with global_consistency_weight = $GCW"
    nohup python main.py \
        --method FedUgV2 \
        --dataset cifar10 \
        --partition_path cifar10_c100_dir01 \
        --model_name cifaredl \
        --num_clients 100 \
        --sampling_prob 0.1 \
        --global_rounds 200 \
        --local_epochs 5 \
        --lr 0.01 \
        --wd 1e-4 \
        --device cuda:0 \
        --eval_gap 1 \
        --uncertainty \
        --ood_dataset svhn \
        --use_adv_ood \
        --adv_epsilon 0.1 \
        --exp_name "fedug_v2_gcw_${GCW}" \
        --global_consistency_weight $GCW \
        --plot_dir "${RESULTS_DIR}/gcw_${GCW}" > sensitivity_logs/fedug_v2_gcw_${GCW}.log 2>&1 &
    
    sleep 10
done

# # 2. 温度参数敏感性分析
# echo "===== 分析温度参数(Temperature)的影响 ====="
# for TEMP in 1.0 1.5 2.0 3.0
# do
#     echo "Running with temperature = $TEMP"
#     nohup python main.py \
#         --method FedUgV2 \
#         --dataset cifar10 \
#         --partition_path cifar10_c100_dir01 \
#         --model_name cifaredl \
#         --num_clients 100 \
#         --sampling_prob 0.1 \
#         --global_rounds 200 \
#         --local_epochs 5 \
#         --lr 0.01 \
#         --wd 1e-4 \
#         --device cuda:1 \
#         --eval_gap 1 \
#         --uncertainty \
#         --use_adv_ood \
#         --adv_epsilon 0.1 \
#         --ood_dataset svhn \
#         --exp_name "fedug_v2_temp_${TEMP}" \
#         --temperature $TEMP \
#         --plot_dir "${RESULTS_DIR}/temp_${TEMP}" > sensitivity_logs/fedug_v2_temp_${TEMP}.log 2>&1 &
    
#     sleep 10
# done

# # 3. EDL权重敏感性分析
# echo "===== 分析EDL权重(EDL Weight)的影响 ====="
# for EDL_W in 0.1 0.2 0.5 1.0
# do
#     echo "Running with edl_weight = $EDL_W"
#     nohup python main.py \
#         --method FedUgV2 \
#         --dataset cifar10 \
#         --partition_path cifar10_c100_dir01 \
#         --model_name cifaredl \
#         --num_clients 100 \
#         --sampling_prob 0.1 \
#         --global_rounds 200 \
#         --local_epochs 5 \
#         --lr 0.01 \
#         --wd 1e-4 \
#         --device cuda:0 \
#         --eval_gap 1 \
#         --uncertainty \
#         --use_adv_ood \
#         --adv_epsilon 0.1 \
#         --ood_dataset svhn \
#         --exp_name "fedug_v2_edl_${EDL_W}" \
#         --edl_weight $EDL_W \
#         --plot_dir "${RESULTS_DIR}/edl_${EDL_W}" > sensitivity_logs/fedug_v2_edl_${EDL_W}.log 2>&1 &
    
#     sleep 10
# done

# # 4. 最大Evidence值敏感性分析
# echo "===== 分析最大Evidence值(Max Evidence)的影响 ====="
# for MAX_EV in 1.0 2.0 5.0 10.0
# do
#     echo "Running with max_evidence = $MAX_EV"
#     nohup python main.py \
#         --method FedUgV2 \
#         --dataset cifar10 \
#         --partition_path cifar10_c100_dir01 \
#         --model_name cifaredl \
#         --num_clients 100 \
#         --sampling_prob 0.1 \
#         --global_rounds 200 \
#         --local_epochs 5 \
#         --lr 0.01 \
#         --wd 1e-4 \
#         --device cuda:1 \
#         --eval_gap 1 \
#         --uncertainty \
#         --use_adv_ood \
#         --adv_epsilon 0.1 \
#         --ood_dataset svhn \
#         --exp_name "fedug_v2_maxev_${MAX_EV}" \
#         --max_evidence $MAX_EV \
#         --plot_dir "${RESULTS_DIR}/maxev_${MAX_EV}" > sensitivity_logs/fedug_v2_maxev_${MAX_EV}.log 2>&1 &
    
#     sleep 10
# done

# # 5. 添加结果分析脚本
# echo "===== 所有敏感性分析任务已在后台启动 ====="
# echo "完成后将自动生成分析报告"

# # 等待所有任务完成
# wait

# # 生成分析报告
# python analyze_sensitivity.py --results_dir ${RESULTS_DIR} --output_file "fedug_v2_sensitivity_analysis.pdf"

# echo "敏感性分析完成，结果保存在 ${RESULTS_DIR} 目录"
