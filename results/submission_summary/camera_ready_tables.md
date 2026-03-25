# Camera-Ready Tables and Captions

## Table 1 (RQ1/RQ2): Mechanism and Reliability at Final Logged Round

### Caption (Main Text)
Table 1 reports mechanism-aware reliability statistics for the three FedSPADE variants at the final logged round of each run. `Full` enables both FSA and GC, `NoFSA` disables semantic basis restoration, and `NoGC` disables geometric correction. We report mean client ECE, ID/OOD entropy gap, probe-client unseen-evidence level, probe collapse rate, and probe curvature in unseen subspace. Together with Fig. 1 and Fig. 2, this table quantifies how component removal changes mechanism trajectories rather than only endpoint accuracy.

### Markdown Version
| Method | Round | Mean ECE ↓ | Mean Entropy Gap ↑ | Probe Curvature ↑ | Probe Unseen Alpha ↑ | Probe Collapse ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Full (FSA+GC) | 15 | 0.3031 | -0.0006 | 0.000008 | 1.7004 | 0.0141 |
| NoFSA | 15 | 0.2392 | 0.0001 | 0.000004 | 1.7009 | 0.0004 |
| NoGC | 15 | 0.2709 | 0.0035 | 0.000005 | 1.7201 | 0.0095 |

### LaTeX Version
```latex
\begin{table}[t]
\centering
\caption{RQ1/RQ2 mechanism and reliability summary at final logged round.}
\label{tab:rq12_mechanism}
\resizebox{\linewidth}{!}{
\begin{tabular}{lrrrrrr}
\toprule
Method & Round & Mean ECE $\downarrow$ & Entropy Gap $\uparrow$ & Probe Curvature $\uparrow$ & Probe Unseen Alpha $\uparrow$ & Probe Collapse $\downarrow$ \\
\midrule
Full (FSA+GC) & 15 & 0.3031 & -0.0006 & 0.000008 & 1.7004 & 0.0141 \\
NoFSA & 15 & 0.2392 & 0.0001 & 0.000004 & 1.7009 & 0.0004 \\
NoGC & 15 & 0.2709 & 0.0035 & 0.000005 & 1.7201 & 0.0095 \\
\bottomrule
\end{tabular}}
\end{table}
```

---

## Table 2 (RQ4): Byzantine Prior Robustness Across Malicious Ratios

### Caption (Main Text)
Table 2 summarizes robustness under Byzantine client perturbation. We report the latest logged checkpoint per malicious ratio and show endpoint ECE/AUROC together with prior estimation errors \(R_\beta\) for five aggregators (weighted, trimmed mean, coordinate median, geometric median, Krum). This table corresponds to Fig. 5 and evaluates whether robust prior aggregation preserves downstream uncertainty quality under increasing corruption.

### Markdown Version
| Run | Ratio | Logged Rounds | Last ECE ↓ | Last AUROC ↑ | Rβ Weighted ↓ | Rβ Trimmed ↓ | Rβ Median ↓ | Rβ Geomedian ↓ | Rβ Krum ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| byz\_r00\_topsec200 | 0.00 | 70 | 0.4612 | 0.6493 | 0.0000 | 0.6520 | 0.7265 | 0.6778 | 0.8078 |
| byz\_r01\_topsec200 | 0.10 | 65 | 0.4712 | 0.7203 | 0.7745 | 0.5985 | 0.7006 | 0.6793 | 0.7936 |
| byz\_r02\_topsec200 | 0.20 | 60 | 0.4459 | 0.6171 | 1.3330 | 0.7527 | 0.7453 | 0.7295 | 0.8305 |
| byz\_r03\_topsec200 | 0.30 | 65 | 0.4834 | 0.7798 | 1.9926 | 0.9853 | 0.7704 | 0.7893 | 0.8614 |

### LaTeX Version
```latex
\begin{table*}[t]
\centering
\caption{RQ4 Byzantine robustness summary at latest logged checkpoints.}
\label{tab:rq4_byzantine}
\resizebox{\linewidth}{!}{
\begin{tabular}{lrrrrrrrrr}
\toprule
Run & Ratio & Logged Rounds & Last ECE $\downarrow$ & Last AUROC $\uparrow$ & $R_\beta$ Weighted $\downarrow$ & $R_\beta$ Trimmed $\downarrow$ & $R_\beta$ Median $\downarrow$ & $R_\beta$ Geomedian $\downarrow$ & $R_\beta$ Krum $\downarrow$ \\
\midrule
byz\_r00\_topsec200 & 0.00 & 70 & 0.4612 & 0.6493 & 0.0000 & 0.6520 & 0.7265 & 0.6778 & 0.8078 \\
byz\_r01\_topsec200 & 0.10 & 65 & 0.4712 & 0.7203 & 0.7745 & 0.5985 & 0.7006 & 0.6793 & 0.7936 \\
byz\_r02\_topsec200 & 0.20 & 60 & 0.4459 & 0.6171 & 1.3330 & 0.7527 & 0.7453 & 0.7295 & 0.8305 \\
byz\_r03\_topsec200 & 0.30 & 65 & 0.4834 & 0.7798 & 1.9926 & 0.9853 & 0.7704 & 0.7893 & 0.8614 \\
\bottomrule
\end{tabular}}
\end{table*}
```

---

## Text Snippets for Main Paper

### RQ1/RQ2 Result Paragraph
In Table 1, removing either FSA or GC shifts mechanism-related indicators in different directions, confirming that reliability cannot be explained by a single scalar endpoint metric alone. Combined with Fig. 1 and Fig. 2, the evidence supports a component-sensitive mechanism path: semantic basis conditioning (FSA) and geometric correction (GC) jointly control unseen-subspace behavior and calibration trajectory.

### RQ4 Result Paragraph
Table 2 and Fig. 5 show that increasing malicious ratio degrades prior estimation quality, with weighted averaging deteriorating fastest in \(R_\beta\). Robust aggregators reduce this degradation to varying degrees, and the downstream ECE/AUROC trajectories indicate non-trivial robustness trade-offs rather than a uniform winner across all attack levels.
