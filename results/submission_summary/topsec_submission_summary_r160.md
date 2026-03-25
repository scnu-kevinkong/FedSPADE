# TopSec Submission Summary (target round <= 160)

## RQ1/RQ2 Core Methods

| Method | Source Run | Round | Mean ECE | Mean Entropy Gap | Probe Curvature | Probe Unseen Alpha | Probe Collapse |
|---|---|---:|---:|---:|---:|---:|---:|
| Full | cifar10_FedUgV2_0.5_full_topsec200 | 155 | 0.4415 | -0.0034 | 0.000003 | 14.6932 | 0.1080 |
| NoFSA | cifar10_FedUgV2_0.5_nofsa_topsec200 | 155 | 0.4356 | -0.0087 | 0.000001 | 23.6992 | 0.0888 |
| NoGC | cifar10_FedUgV2_0.5_nogc_topsec_gpu | 15 | 0.2709 | 0.0035 | 0.000005 | 1.7201 | 0.0095 |

## RQ4 Byzantine Robustness

| Run | Ratio | Logged Rounds | Last ECE | Last AUROC | Rβ Weighted | Rβ Trimmed | Rβ Median | Rβ Geomedian | Rβ Krum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cifar10_FedUgV2_0.5_byz_r00_topsec200 | 0.00 | 70 | 0.4612 | 0.6493 | 0.0000 | 0.6520 | 0.7265 | 0.6778 | 0.8078 |
| cifar10_FedUgV2_0.5_byz_r01_topsec200 | 0.10 | 65 | 0.4712 | 0.7203 | 0.7745 | 0.5985 | 0.7006 | 0.6793 | 0.7936 |
| cifar10_FedUgV2_0.5_byz_r02_topsec200 | 0.20 | 60 | 0.4459 | 0.6171 | 1.3330 | 0.7527 | 0.7453 | 0.7295 | 0.8305 |
| cifar10_FedUgV2_0.5_byz_r03_topsec200 | 0.30 | 65 | 0.4834 | 0.7798 | 1.9926 | 0.9853 | 0.7704 | 0.7893 | 0.8614 |
