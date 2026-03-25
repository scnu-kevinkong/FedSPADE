# TopSec Submission Summary

## RQ1/RQ2 Core Methods

| Method | Source Run | Round | Mean ECE | Mean Entropy Gap | Probe Curvature | Probe Unseen Alpha | Probe Collapse |
|---|---|---:|---:|---:|---:|---:|---:|
| Full | cifar10_FedUgV2_0.5_full_topsec_gpu | 15 | 0.3031 | -0.0006 | 0.000008 | 1.7004 | 0.0141 |
| NoFSA | cifar10_FedUgV2_0.5_nofsa_topsec_gpu | 15 | 0.2392 | 0.0001 | 0.000004 | 1.7009 | 0.0004 |
| NoGC | cifar10_FedUgV2_0.5_nogc_topsec_gpu | 15 | 0.2709 | 0.0035 | 0.000005 | 1.7201 | 0.0095 |

## RQ4 Byzantine Robustness

| Run | Ratio | Logged Rounds | Last ECE | Last AUROC | Rβ Weighted | Rβ Trimmed | Rβ Median | Rβ Geomedian | Rβ Krum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cifar10_FedUgV2_0.5_byz_r00_topsec200 | 0.00 | 70 | 0.4612 | 0.6493 | 0.0000 | 0.6520 | 0.7265 | 0.6778 | 0.8078 |
| cifar10_FedUgV2_0.5_byz_r01_topsec200 | 0.10 | 65 | 0.4712 | 0.7203 | 0.7745 | 0.5985 | 0.7006 | 0.6793 | 0.7936 |
| cifar10_FedUgV2_0.5_byz_r02_topsec200 | 0.20 | 60 | 0.4459 | 0.6171 | 1.3330 | 0.7527 | 0.7453 | 0.7295 | 0.8305 |
| cifar10_FedUgV2_0.5_byz_r03_topsec200 | 0.30 | 65 | 0.4834 | 0.7798 | 1.9926 | 0.9853 | 0.7704 | 0.7893 | 0.8614 |
