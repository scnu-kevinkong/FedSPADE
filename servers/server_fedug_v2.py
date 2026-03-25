import torch
import numpy as np
import time
import os # ### SUGGESTION 1: Added for saving plots
from copy import deepcopy
from servers.server_base import Server
from clients.client_fedug_v2 import ClientFedUgV2
from utils.util import AverageMeter
### SUGGESTION 1: UNCERTAINTY QUANTIFICATION ###
from utils.uncertainty_metrics import calculate_brier_score, calculate_ece, plot_reliability_diagram, calculate_ood_auc
from sklearn.metrics import average_precision_score # 用于 AUC-PR
from scipy.stats import spearmanr # 用于认知不确定性保真度
from scipy.stats import ttest_ind # 用于偶然不确定性一致性
import matplotlib.pyplot as plt

class ServerFedUgV2(Server):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.clients = [ClientFedUgV2(args, i) for i in range(args.num_clients)]
        
        self.global_prior_alpha = torch.ones(self.num_classes) * 2.0
        
        self.round_times = []
        self.train_times = []

        ### SUGGESTION 1: Create directory for saving evaluation results
        exp_suffix = getattr(args, "exp_name", "default")
        self.results_dir = f"results/{args.dataset}_{args.method}_{args.dir_alpha}_{exp_suffix}"
        os.makedirs(self.results_dir, exist_ok=True)
        self.mechanism_dir = os.path.join(self.results_dir, "mechanism")
        os.makedirs(self.mechanism_dir, exist_ok=True)
        self.byzantine_dir = os.path.join(self.results_dir, "byzantine")
        os.makedirs(self.byzantine_dir, exist_ok=True)
        self.mechanism_log_gap = int(getattr(args, "mechanism_log_gap", 10))
        self.byzantine_log_gap = int(getattr(args, "byzantine_log_gap", 1))
        self.mechanism_probe_top_frac = float(getattr(args, "mechanism_probe_top_frac", 0.25))
        self.collapse_tau = float(getattr(args, "collapse_tau", 0.2))
        self.save_mechanism_assets = bool(getattr(args, "save_mechanism_assets", True))
        self.save_byzantine_assets = bool(getattr(args, "save_byzantine_assets", True))
        self.save_eval_plots = bool(getattr(args, "save_eval_plots", True))
        self.client_missing_counts = np.array([int(np.sum(np.array(c.label_distribution) <= 0)) for c in self.clients], dtype=np.int32)
        self.client_entropy_skew = np.array([self._compute_entropy_skew(np.array(c.label_distribution)) for c in self.clients], dtype=np.float32)
        self.client_severity = (self.client_missing_counts / max(1, self.num_classes)) + self.client_entropy_skew
        top_k = max(1, int(np.ceil(self.num_clients * self.mechanism_probe_top_frac)))
        self.probe_client_ids = np.argsort(-self.client_severity)[:top_k].astype(np.int32)
        self.latest_eval_snapshot = {}
        self.byzantine_records = []
        
        print(f"Initialized FedSPADE Server with {len(self.clients)} clients")

    def _compute_entropy_skew(self, label_count):
        label_count = label_count.astype(np.float64)
        total = float(np.sum(label_count))
        if total <= 0:
            return 1.0
        p = label_count / total
        entropy = -np.sum(p * np.log(p + 1e-12))
        return float(1.0 - entropy / np.log(self.num_classes))

    def _projected_min_nonzero_eig(self, fisher):
        n = fisher.shape[0]
        if n <= 1:
            return 0.0
        proj = np.eye(n) - np.ones((n, n)) / n
        f_proj = proj @ fisher @ proj
        eigvals = np.linalg.eigvalsh(f_proj)
        positive = eigvals[eigvals > 1e-10]
        if positive.size == 0:
            return 0.0
        return float(positive.min())

    def _compute_aggregated_prior_methods(self, reports, weights):
        if len(reports) == 0:
            default = np.ones(self.num_classes, dtype=np.float32) * 1.5
            return {
                "weighted": default,
                "trimmed_mean": default,
                "coord_median": default,
                "geometric_median": default,
                "krum": default,
            }
        stacked = np.stack([r.detach().cpu().numpy().astype(np.float64) for r in reports], axis=0)
        w = np.array(weights, dtype=np.float64)
        if np.sum(w) <= 0:
            w = np.ones_like(w)
        w = w / np.sum(w)
        weighted = np.average(stacked, axis=0, weights=w)
        sorted_vals = np.sort(stacked, axis=0)
        n = sorted_vals.shape[0]
        trim = int(np.floor(0.2 * n))
        if n - 2 * trim <= 0:
            trimmed_mean = weighted.copy()
        else:
            trimmed_mean = sorted_vals[trim:n - trim].mean(axis=0)
        coord_median = np.median(stacked, axis=0)
        gmed = stacked.mean(axis=0)
        for _ in range(100):
            d = np.linalg.norm(stacked - gmed[None, :], axis=1)
            d = np.clip(d, 1e-8, None)
            ww = 1.0 / d
            gmed_next = (stacked * ww[:, None]).sum(axis=0) / ww.sum()
            if np.linalg.norm(gmed_next - gmed) < 1e-7:
                gmed = gmed_next
                break
            gmed = gmed_next
        byz_count = int(round(self.args.byzantine_ratio * len(reports)))
        m = max(1, len(reports) - byz_count - 2)
        scores = []
        for i in range(len(reports)):
            dist = np.sum((stacked - stacked[i][None, :]) ** 2, axis=1)
            scores.append(np.sort(dist)[1:m + 1].sum())
        krum = stacked[int(np.argmin(scores))]
        return {
            "weighted": np.clip(weighted, 1.1, 5.0).astype(np.float32),
            "trimmed_mean": np.clip(trimmed_mean, 1.1, 5.0).astype(np.float32),
            "coord_median": np.clip(coord_median, 1.1, 5.0).astype(np.float32),
            "geometric_median": np.clip(gmed, 1.1, 5.0).astype(np.float32),
            "krum": np.clip(krum, 1.1, 5.0).astype(np.float32),
        }

    def _apply_byzantine_attack(self, reports):
        n = len(reports)
        if n == 0:
            return reports, np.zeros(0, dtype=bool)
        byz_ratio = float(getattr(self.args, "byzantine_ratio", 0.0))
        byz_count = int(round(byz_ratio * n))
        byz_count = min(byz_count, n)
        malicious_mask = np.zeros(n, dtype=bool)
        if byz_count == 0:
            return reports, malicious_mask
        rng = np.random.default_rng(2026 + len(self.byzantine_records))
        idx = rng.choice(np.arange(n), size=byz_count, replace=False)
        malicious_mask[idx] = True
        attack_type = getattr(self.args, "byzantine_attack_type", "random")
        attacked = []
        for i, rep in enumerate(reports):
            vec = rep.detach().clone().float()
            if malicious_mask[i]:
                if attack_type == "random":
                    vec = torch.empty_like(vec).uniform_(0.1, 6.0)
                elif attack_type == "label_flip":
                    vec = torch.flip(vec, dims=[0]) * 1.5
                elif attack_type == "gaussian_noise":
                    vec = vec + torch.randn_like(vec) * 1.5
                elif attack_type == "sign_flip":
                    vec = 4.0 - vec
                vec = torch.clamp(vec, min=0.1, max=8.0)
            attacked.append(vec)
        return attacked, malicious_mask

    def _save_mechanism_round(self, round_idx, per_client_outputs, ood_outputs):
        if not self.save_mechanism_assets:
            return
        rows = sorted(per_client_outputs.keys())
        num_clients = len(self.clients)
        unseen_mask = np.zeros((num_clients, self.num_classes), dtype=np.int8)
        alpha_unseen_mean = np.full(num_clients, np.nan, dtype=np.float32)
        collapse_rate = np.full(num_clients, np.nan, dtype=np.float32)
        curvature = np.full(num_clients, np.nan, dtype=np.float32)
        ece_client = np.full(num_clients, np.nan, dtype=np.float32)
        entropy_gap = np.full(num_clients, np.nan, dtype=np.float32)
        bin_conf = np.full((num_clients, 10), np.nan, dtype=np.float32)
        bin_acc = np.full((num_clients, 10), np.nan, dtype=np.float32)
        centroids = np.full((num_clients, self.num_classes, self.D), np.nan, dtype=np.float32)
        for cid in rows:
            client = self.clients[cid]
            counts = np.array(client.label_distribution, dtype=np.float32)
            unseen = np.where(counts <= 0)[0]
            unseen_mask[cid, unseen] = 1
            out = per_client_outputs[cid]
            probs = out["probs"]
            labels = out["labels"].astype(int)
            evidence = out["evidence"]
            if unseen.size > 0:
                alpha_unseen = evidence[:, unseen] + 1.0
                alpha_unseen_mean[cid] = float(alpha_unseen.mean())
                collapse_rate[cid] = float(np.mean(alpha_unseen <= (1.0 + self.collapse_tau)))
                y = np.zeros_like(probs)
                y[np.arange(labels.shape[0]), labels] = 1.0
                g = probs[:, unseen] - y[:, unseen]
                fisher = (g.T @ g) / max(1, g.shape[0])
                curvature[cid] = self._projected_min_nonzero_eig(fisher)
            ece_client[cid] = float(out.get("ece", np.nan))
            conf = np.max(probs, axis=1)
            pred = np.argmax(probs, axis=1)
            corr = (pred == labels).astype(np.float32)
            for bi in range(10):
                lo = bi / 10.0
                hi = (bi + 1) / 10.0
                if bi == 9:
                    mask = (conf >= lo) & (conf <= hi)
                else:
                    mask = (conf >= lo) & (conf < hi)
                if np.any(mask):
                    bin_conf[cid, bi] = float(conf[mask].mean())
                    bin_acc[cid, bi] = float(corr[mask].mean())
            if cid in ood_outputs and ood_outputs[cid] is not None:
                ood_p = ood_outputs[cid]["probs"]
                h_id = -np.sum(probs * np.log(probs + 1e-8), axis=1).mean()
                h_ood = -np.sum(ood_p * np.log(ood_p + 1e-8), axis=1).mean()
                entropy_gap[cid] = float(h_ood - h_id)
            ccent = client.class_centroids
            if ccent is not None:
                arr = ccent.detach().cpu().numpy().astype(np.float32)
                d = min(arr.shape[1], self.D)
                centroids[cid, :, :d] = arr[:, :d]
            else:
                feats = out.get("features", None)
                labels_np = out.get("labels", None)
                if feats is not None and labels_np is not None:
                    feats = np.asarray(feats, dtype=np.float32)
                    labels_np = np.asarray(labels_np, dtype=np.int32)
                    if feats.ndim == 2 and feats.shape[0] == labels_np.shape[0]:
                        d = min(feats.shape[1], self.D)
                        for k in range(self.num_classes):
                            mk = labels_np == k
                            if np.any(mk):
                                centroids[cid, k, :d] = feats[mk, :d].mean(axis=0)
        global_anchor = np.full((self.num_classes, self.D), np.nan, dtype=np.float32)
        for k in range(self.num_classes):
            valid = np.where(~np.isnan(centroids[:, k, 0]))[0]
            if valid.size > 0:
                global_anchor[k] = np.nanmean(centroids[valid, k, :], axis=0)
        probe_mask = np.zeros(num_clients, dtype=np.int8)
        probe_mask[self.probe_client_ids] = 1
        save_path = os.path.join(self.mechanism_dir, f"round_{round_idx:04d}.npz")
        np.savez_compressed(
            save_path,
            round=np.array([round_idx], dtype=np.int32),
            client_ids=np.arange(num_clients, dtype=np.int32),
            probe_mask=probe_mask,
            missing_counts=self.client_missing_counts.astype(np.int32),
            severity=self.client_severity.astype(np.float32),
            unseen_mask=unseen_mask,
            alpha_unseen_mean=alpha_unseen_mean,
            collapse_rate=collapse_rate,
            curvature=curvature,
            ece_client=ece_client,
            entropy_gap=entropy_gap,
            reliability_bin_conf=bin_conf,
            reliability_bin_acc=bin_acc,
            centroids=centroids,
            global_anchor=global_anchor,
        )
        self.logger.info(f"[Mechanism Log] saved {save_path}")

    def _append_byzantine_record(self, round_idx, reports_honest, reports_attacked, malicious_mask, client_data_sizes):
        if not self.save_byzantine_assets:
            return
        if not (round_idx % self.byzantine_log_gap == 0 or round_idx == self.global_rounds):
            return
        weights = np.array(client_data_sizes, dtype=np.float64)
        methods_attack = self._compute_aggregated_prior_methods(reports_attacked, weights)
        honest_only_reports = [r for r, m in zip(reports_honest, malicious_mask) if not m]
        honest_only_weights = [w for w, m in zip(client_data_sizes, malicious_mask) if not m]
        if len(honest_only_reports) == 0:
            honest_only_reports = reports_honest
            honest_only_weights = client_data_sizes
        oracle = self._compute_aggregated_prior_methods(honest_only_reports, np.array(honest_only_weights, dtype=np.float64))["weighted"]
        rec = {
            "round": int(round_idx),
            "ratio": float(getattr(self.args, "byzantine_ratio", 0.0)),
            "malicious_count": int(malicious_mask.sum()),
            "beta_oracle_benign": oracle.astype(np.float32),
            "beta_weighted": methods_attack["weighted"].astype(np.float32),
            "beta_trimmed": methods_attack["trimmed_mean"].astype(np.float32),
            "beta_median": methods_attack["coord_median"].astype(np.float32),
            "beta_geomedian": methods_attack["geometric_median"].astype(np.float32),
            "beta_krum": methods_attack["krum"].astype(np.float32),
            "is_malicious": malicious_mask.astype(np.int8),
            "alpha_reports_honest": np.stack([r.detach().cpu().numpy() for r in reports_honest]).astype(np.float32),
            "alpha_reports_attacked": np.stack([r.detach().cpu().numpy() for r in reports_attacked]).astype(np.float32),
            "ece_mean": float(self.latest_eval_snapshot.get("ece_mean", np.nan)),
            "auroc_epistemic": float(self.latest_eval_snapshot.get("auroc_epistemic", np.nan)),
            "acc_mean": float(self.latest_eval_snapshot.get("acc_mean", np.nan)),
            "brier_mean": float(self.latest_eval_snapshot.get("brier_mean", np.nan)),
            "fpr95": float(self.latest_eval_snapshot.get("fpr95", np.nan)),
        }
        self.byzantine_records.append(rec)
        save_path = os.path.join(self.byzantine_dir, "records.npz")
        np.savez_compressed(
            save_path,
            rounds=np.array([r["round"] for r in self.byzantine_records], dtype=np.int32),
            ratio=np.array([r["ratio"] for r in self.byzantine_records], dtype=np.float32),
            malicious_count=np.array([r["malicious_count"] for r in self.byzantine_records], dtype=np.int32),
            beta_oracle_benign=np.stack([r["beta_oracle_benign"] for r in self.byzantine_records]).astype(np.float32),
            beta_weighted=np.stack([r["beta_weighted"] for r in self.byzantine_records]).astype(np.float32),
            beta_trimmed=np.stack([r["beta_trimmed"] for r in self.byzantine_records]).astype(np.float32),
            beta_median=np.stack([r["beta_median"] for r in self.byzantine_records]).astype(np.float32),
            beta_geomedian=np.stack([r["beta_geomedian"] for r in self.byzantine_records]).astype(np.float32),
            beta_krum=np.stack([r["beta_krum"] for r in self.byzantine_records]).astype(np.float32),
            ece_mean=np.array([r["ece_mean"] for r in self.byzantine_records], dtype=np.float32),
            auroc_epistemic=np.array([r["auroc_epistemic"] for r in self.byzantine_records], dtype=np.float32),
            acc_mean=np.array([r["acc_mean"] for r in self.byzantine_records], dtype=np.float32),
            brier_mean=np.array([r["brier_mean"] for r in self.byzantine_records], dtype=np.float32),
            fpr95=np.array([r["fpr95"] for r in self.byzantine_records], dtype=np.float32),
        )
        self.logger.info(f"[Byzantine Log] saved {save_path}")

    def _update_byzantine_eval_snapshot(self, round_idx):
        if len(self.byzantine_records) == 0:
            return
        updated = False
        for rec in reversed(self.byzantine_records):
            if rec["round"] == int(round_idx):
                rec["ece_mean"] = float(self.latest_eval_snapshot.get("ece_mean", np.nan))
                rec["auroc_epistemic"] = float(self.latest_eval_snapshot.get("auroc_epistemic", np.nan))
                rec["acc_mean"] = float(self.latest_eval_snapshot.get("acc_mean", np.nan))
                rec["brier_mean"] = float(self.latest_eval_snapshot.get("brier_mean", np.nan))
                rec["fpr95"] = float(self.latest_eval_snapshot.get("fpr95", np.nan))
                updated = True
                break
        if not updated:
            return
        save_path = os.path.join(self.byzantine_dir, "records.npz")
        np.savez_compressed(
            save_path,
            rounds=np.array([r["round"] for r in self.byzantine_records], dtype=np.int32),
            ratio=np.array([r["ratio"] for r in self.byzantine_records], dtype=np.float32),
            malicious_count=np.array([r["malicious_count"] for r in self.byzantine_records], dtype=np.int32),
            beta_oracle_benign=np.stack([r["beta_oracle_benign"] for r in self.byzantine_records]).astype(np.float32),
            beta_weighted=np.stack([r["beta_weighted"] for r in self.byzantine_records]).astype(np.float32),
            beta_trimmed=np.stack([r["beta_trimmed"] for r in self.byzantine_records]).astype(np.float32),
            beta_median=np.stack([r["beta_median"] for r in self.byzantine_records]).astype(np.float32),
            beta_geomedian=np.stack([r["beta_geomedian"] for r in self.byzantine_records]).astype(np.float32),
            beta_krum=np.stack([r["beta_krum"] for r in self.byzantine_records]).astype(np.float32),
            ece_mean=np.array([r["ece_mean"] for r in self.byzantine_records], dtype=np.float32),
            auroc_epistemic=np.array([r["auroc_epistemic"] for r in self.byzantine_records], dtype=np.float32),
            acc_mean=np.array([r["acc_mean"] for r in self.byzantine_records], dtype=np.float32),
            brier_mean=np.array([r["brier_mean"] for r in self.byzantine_records], dtype=np.float32),
            fpr95=np.array([r["fpr95"] for r in self.byzantine_records], dtype=np.float32),
        )
    
    def send_models(self):
        for client in self.clients:
            client.model.load_state_dict(self.model.state_dict())
            client.model_global.load_state_dict(self.model.state_dict())
            client.set_global_prior_alpha(self.global_prior_alpha)
    
    def train_round(self, round_num):
        self.logger.info(f"--- Global Round {round_num}/{self.global_rounds} ---")
        self.sample_active_clients()
        active_clients = self.active_clients
        
        self.send_models()
        
        client_updates = []
        client_alpha_reports = []
        client_data_sizes = []
        
        kl_div_times = AverageMeter()

        for client in active_clients:
            client.set_current_global_round(round_num)
            stats = client.train(round_num)
            
            kl_div_times.update(stats.get('kl_div_time', 0))

            model_weights, alpha_report, data_size = client.get_update()
            
            client_updates.append(model_weights)
            client_alpha_reports.append(alpha_report)
            client_data_sizes.append(data_size)

        # 报告开销分析
        avg_kl_time = kl_div_times.avg * 1000
        self.logger.info(f"[Overhead Analysis] Avg. KL-Div computation time per batch: {avg_kl_time:.4f} ms")
        if client_alpha_reports:
            report_size_bytes = client_alpha_reports[0].element_size() * client_alpha_reports[0].nelement()
            self.logger.info(f"[Overhead Analysis] Communication overhead per client for alpha_report: {report_size_bytes} bytes")

        if client_updates:
            self.aggregate_models(client_updates, client_data_sizes)
            honest_reports = [r.detach().clone() for r in client_alpha_reports]
            attacked_reports, malicious_mask = self._apply_byzantine_attack(client_alpha_reports)
            if hasattr(self.args, 'use_byzantine_robust') and self.args.use_byzantine_robust:
                self.logger.info("Using Byzantine-robust prior aggregation")
                self.aggregate_alpha_priors(attacked_reports, client_data_sizes)
            else:
                self.aggregate_alpha_reports(attacked_reports, client_data_sizes)
            self._append_byzantine_record(round_num, honest_reports, attacked_reports, malicious_mask, client_data_sizes)
        
        return 0.0, 0.0
    
    def aggregate_models(self, client_updates, client_data_sizes):
        if not client_updates: return
        total_data_size = sum(client_data_sizes)
        if total_data_size == 0: return
        
        global_state_dict = self.model.state_dict()
        aggregated_state_dict = {name: torch.zeros_like(param) for name, param in global_state_dict.items()}
        
        for i, client_state_dict in enumerate(client_updates):
            weight = client_data_sizes[i] / total_data_size
            for name in aggregated_state_dict:
                if 'num_batches_tracked' not in name:
                    aggregated_state_dict[name] += weight * client_state_dict[name]
        
        self.model.load_state_dict(aggregated_state_dict)
    
    def aggregate_alpha_reports(self, alpha_reports, client_data_sizes):
        """聚合客户端的alpha报告"""
        if not alpha_reports: return
        
        total_size = sum(client_data_sizes)
        if total_size == 0: return
        
        weighted_sum = torch.zeros_like(alpha_reports[0])
        for report, size in zip(alpha_reports, client_data_sizes):
            weighted_sum += (size / total_size) * report.to(weighted_sum.device)
        
        ### SUGGESTION 2: ABLATION STUDY ###
        # 条件应用平滑和限制
        if hasattr(self.args, 'use_prior_smoothing') and self.args.use_prior_smoothing:
            smoothing_factor = 0.1
            uniform_prior = torch.ones_like(weighted_sum) * 2.0
            aggregated_alpha = (1 - smoothing_factor) * weighted_sum + smoothing_factor * uniform_prior
        else:
            aggregated_alpha = weighted_sum
        
        if hasattr(self.args, 'use_prior_clamping') and self.args.use_prior_clamping:
            self.global_prior_alpha = torch.clamp(aggregated_alpha, min=1.1, max=5.0)
        else:
            self.global_prior_alpha = aggregated_alpha
        
        # 记录alpha值统计
        self.logger.info(f"Global Prior Alpha Stats: Min={self.global_prior_alpha.min().item():.2f}, "
                        f"Max={self.global_prior_alpha.max().item():.2f}, "
                        f"Mean={self.global_prior_alpha.mean().item():.2f}")

    def evaluate_uncertainty(self, round_idx):
        """评估不确定性性能"""
        self.logger.info(f"--- Evaluating models at round {round_idx} with Advanced Uncertainty Metrics ---")
        
        # 评估指标
        personal_accs = []
        personal_briers = []
        personal_eces = []
        fpr95 = np.nan
        epistemic_auroc = np.nan
        epistemic_fidelities = []
        aleatoric_consistencies = []
        
        # 收集OOD评估数据
        all_id_uncertainties = []
        all_ood_uncertainties = []
        all_id_energies = []
        all_ood_energies = []
        all_id_total = []
        all_ood_total = []
        per_client_outputs = {}
        per_client_ood_outputs = {}
        
        # 评估每个客户端
        for client in self.clients:
            # ID数据评估
            id_results = client.get_eval_output(use_personal=True, dataset='test')
            if id_results is None:
                continue
            per_client_outputs[client.client_idx] = id_results
                
            # 计算准确率
            preds = np.argmax(id_results['probs'], axis=1)
            acc = np.mean(preds == id_results['labels']) * 100
            personal_accs.append(acc)
            
            # 计算Brier分数
            if 'brier_score' in id_results:
                personal_briers.append(id_results['brier_score'])
            else:
                y_one_hot = np.zeros((len(id_results['labels']), id_results['probs'].shape[1]))
                for i, label in enumerate(id_results['labels']):
                    y_one_hot[i, label] = 1
                brier = np.mean(np.sum((id_results['probs'] - y_one_hot) ** 2, axis=1))
                personal_briers.append(brier)
            
            # 计算ECE
            if 'ece' in id_results:
                personal_eces.append(id_results['ece'])
            else:
                confidences = np.max(id_results['probs'], axis=1)
                ece = calculate_ece(confidences, preds == id_results['labels'])
                personal_eces.append(ece)
            
            # 计算认知不确定性与错误的相关性
            is_correct = (preds == id_results['labels']).astype(float)
            if len(np.unique(is_correct)) > 1:  # 确保有正确和错误的预测
                epistemic_corr, _ = spearmanr(id_results['epistemic_uncertainties'], 1 - is_correct)
                if not np.isnan(epistemic_corr):
                    epistemic_fidelities.append(epistemic_corr)
            
            # 计算偶然不确定性一致性
            if 'aleatoric_uncertainties' in id_results:
                correct_mask = is_correct == 1
                incorrect_mask = is_correct == 0
                
                if np.sum(correct_mask) > 0 and np.sum(incorrect_mask) > 0:
                    # 使用t检验比较正确和错误预测的偶然不确定性
                    t_stat, p_value = ttest_ind(
                        id_results['aleatoric_uncertainties'][correct_mask],
                        id_results['aleatoric_uncertainties'][incorrect_mask],
                        equal_var=False
                    )
                    # 一致性指标：p值越小，差异越显著
                    aleatoric_consistencies.append(1 - min(p_value, 1.0))
            
            # 收集ID不确定性
            all_id_uncertainties.extend(id_results['epistemic_uncertainties'])
            if 'energy_uncertainties' in id_results:
                all_id_energies.extend(id_results['energy_uncertainties'])
            if 'total_uncertainties' in id_results:
                all_id_total.extend(id_results['total_uncertainties'])
            
            # OOD数据评估
            ood_results = client.get_eval_output(use_personal=True, dataset='ood')
            if ood_results is not None:
                per_client_ood_outputs[client.client_idx] = ood_results
                all_ood_uncertainties.extend(ood_results['epistemic_uncertainties'])
                if 'energy_uncertainties' in ood_results:
                    all_ood_energies.extend(ood_results['energy_uncertainties'])
                if 'total_uncertainties' in ood_results:
                    all_ood_total.extend(ood_results['total_uncertainties'])
        
        # 计算平均指标
        mean_acc = 0
        if personal_accs:
            mean_acc = np.mean(personal_accs)
            self.logger.info(f"[Metric 1] Personalized Accuracy | Mean: {mean_acc:.2f}%, Std: {np.std(personal_accs):.2f}, Min: {np.min(personal_accs):.2f}, Max: {np.max(personal_accs):.2f}")
            self.logger.info(f"[Metric 1] Personalized Brier Score | Mean: {np.mean(personal_briers):.4f}, Std: {np.std(personal_briers):.4f}")
            self.logger.info(f"[Metric 1] Personalized ECE | Mean: {np.mean(personal_eces):.4f}, Std: {np.std(personal_eces):.4f}")
        
        if epistemic_fidelities:
            self.logger.info(f"[Metric 2] Epistemic Fidelity (Spearman Corr) | Mean: {np.mean(epistemic_fidelities):.4f}, Std: {np.std(epistemic_fidelities):.4f}")
        
        if aleatoric_consistencies:
            self.logger.info(f"[Metric 3] Aleatoric Consistency (U) | Mean: {np.mean(aleatoric_consistencies):.4f}, Std: {np.std(aleatoric_consistencies):.4f}")
        
        # OOD检测性能
        if all_id_uncertainties and all_ood_uncertainties:
            # 认知不确定性的OOD检测
            epistemic_auroc = calculate_ood_auc(all_id_uncertainties, all_ood_uncertainties)
            self.logger.info(f"[OOD Eval] OOD AUC-ROC (Epistemic): {epistemic_auroc:.4f}")
            
            # FPR@95%TPR
            fpr95 = self._calculate_fpr_at_tpr(all_id_uncertainties, all_ood_uncertainties, tpr_threshold=0.95)
            self.logger.info(f"[Metric 4] OOD FPR@95%TPR: {fpr95:.4f}")
            
            # AUPRC
            auprc = self._calculate_auprc(all_id_uncertainties, all_ood_uncertainties)
            self.logger.info(f"[Metric 5] OOD AUC-PR: {auprc:.4f}")
            
            # 能量不确定性的OOD检测
            if all_id_energies and all_ood_energies:
                energy_auroc = calculate_ood_auc(all_id_energies, all_ood_energies)
                self.logger.info(f"[OOD Eval] OOD AUC-ROC (Energy): {energy_auroc:.4f}")
            
            # 总不确定性的OOD检测
            if all_id_total and all_ood_total:
                total_auroc = calculate_ood_auc(all_id_total, all_ood_total)
                self.logger.info(f"[OOD Eval] OOD AUC-ROC (Total): {total_auroc:.4f}")
            
            # 绘制不确定性分布图
            if self.save_eval_plots:
                self.plot_uncertainty_distributions(
                    all_id_uncertainties, 
                    all_id_total[:len(all_id_uncertainties)] if all_id_total else None,
                    all_ood_uncertainties, 
                    all_ood_total[:len(all_ood_uncertainties)] if all_ood_total else None,
                    round_idx, 
                    self.results_dir
                )
        
        # 返回平均准确率，用于保存最佳模型
        self.latest_eval_snapshot = {
            "acc_mean": float(np.mean(personal_accs)) if len(personal_accs) > 0 else np.nan,
            "brier_mean": float(np.mean(personal_briers)) if len(personal_briers) > 0 else np.nan,
            "ece_mean": float(np.mean(personal_eces)) if len(personal_eces) > 0 else np.nan,
            "auroc_epistemic": float(epistemic_auroc) if all_id_uncertainties and all_ood_uncertainties else np.nan,
            "fpr95": float(fpr95) if all_id_uncertainties and all_ood_uncertainties else np.nan,
            "round": int(round_idx),
        }
        self._update_byzantine_eval_snapshot(round_idx)
        if self.save_mechanism_assets and (round_idx % self.mechanism_log_gap == 0 or round_idx == self.global_rounds):
            self._save_mechanism_round(round_idx, per_client_outputs, per_client_ood_outputs)
        return mean_acc

    def train(self):
        self.logger.info("Starting FedSPADE Training Process")
        best_acc = 0.0
        
        for round_num in range(1, self.global_rounds + 1):
            start_time = time.time()
            self.train_round(round_num)
            round_time = time.time() - start_time
            self.round_times.append(round_time)
            
            if round_num % self.args.eval_gap == 0 or round_num == self.global_rounds:
                # 使用evaluate_uncertainty方法替代evaluate_models
                test_acc = self.evaluate_uncertainty(round_num)
                if test_acc > best_acc:
                    best_acc = test_acc
                    self.logger.info(f"New best personalized accuracy: {best_acc:.2f}%")
                    
                    # 保存最佳模型
                    if hasattr(self.args, 'save_models') and self.args.save_models:
                        self._save_best_model(round_num, best_acc)
        
        self.logger.info(f"Training completed. Best personalized accuracy: {best_acc:.2f}%")
        return best_acc

    def _save_best_model(self, round_num, acc):
        """保存最佳模型"""
        save_path = os.path.join(self.results_dir, f"best_model_round{round_num}_acc{acc:.2f}.pt")
        torch.save({
            'model': self.model.state_dict(),
            'global_prior_alpha': self.global_prior_alpha,
            'round': round_num,
            'acc': acc
        }, save_path)
        self.logger.info(f"Saved best model to {save_path}")

    def plot_uncertainty_distributions(self, id_epistemic, id_aleatoric, ood_epistemic, ood_aleatoric, round_num, save_dir):
        """绘制不确定性分布图"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(id_epistemic, bins=50, alpha=0.7, label='ID Epistemic', density=True)
        if ood_epistemic is not None and len(ood_epistemic) > 0:
            plt.hist(ood_epistemic, bins=50, alpha=0.7, label='OOD Epistemic', density=True)
        plt.title(f'Epistemic Uncertainty (Round {round_num})')
        plt.xlabel('Uncertainty')
        plt.ylabel('Density')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(id_aleatoric, bins=50, alpha=0.7, label='ID Total', density=True)
        if ood_aleatoric is not None and len(ood_aleatoric) > 0:
            plt.hist(ood_aleatoric, bins=50, alpha=0.7, label='OOD Total', density=True)
        plt.title(f'Total Uncertainty (Round {round_num})')
        plt.xlabel('Uncertainty')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"uncertainty_dist_r{round_num}.png"))
        plt.close()

    def plot_error_vs_uncertainty(self, epistemic, aleatoric, errors, round_num, save_dir):
        """绘制错误与不确定性关系图"""
        correct_mask = ~errors
        
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(epistemic[correct_mask], aleatoric[correct_mask], alpha=0.1, s=5, label='Correct')
        plt.scatter(epistemic[errors], aleatoric[errors], alpha=0.1, s=5, label='Incorrect')
        plt.xlabel('Epistemic Uncertainty')
        plt.ylabel('Total Uncertainty')
        plt.title(f'Uncertainty of Predictions (Round {round_num})')
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')

        plt.subplot(1, 2, 2)
        plt.boxplot([epistemic[correct_mask], epistemic[errors]], labels=['Correct', 'Incorrect'])
        plt.title('Epistemic Uncertainty vs. Error')
        plt.ylabel('Epistemic Uncertainty')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"error_vs_uncertainty_r{round_num}.png"))
        plt.close()

    def _expected_calibration_error(self, confidences, accuracies, num_bins=10):
        """计算期望校准误差 (ECE)"""
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # 计算落在这个bin中的样本
            in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

    def _spearman_correlation(self, x, y):
        """计算Spearman相关系数"""
        from scipy.stats import spearmanr
        corr, _ = spearmanr(x, y)
        return corr if not np.isnan(corr) else 0.0

    def _calculate_auroc(self, id_values, ood_values):
        """计算AUC-ROC"""
        from sklearn.metrics import roc_auc_score
        # 确保ID为负类(0)，OOD为正类(1)
        y_true = np.concatenate([np.zeros(len(id_values)), np.ones(len(ood_values))])
        y_score = np.concatenate([id_values, ood_values])
        return roc_auc_score(y_true, y_score)

    def _calculate_fpr_at_tpr(self, id_values, ood_values, tpr_threshold=0.95):
        """计算在给定TPR阈值下的FPR"""
        from sklearn.metrics import roc_curve
        # 确保ID为负类(0)，OOD为正类(1)
        y_true = np.concatenate([np.zeros(len(id_values)), np.ones(len(ood_values))])
        y_score = np.concatenate([id_values, ood_values])
        
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        
        # 找到最接近目标TPR的索引
        idx = np.argmin(np.abs(tpr - tpr_threshold))
        return fpr[idx]

    def _calculate_auprc(self, id_values, ood_values):
        """计算AUC-PR曲线下面积"""
        from sklearn.metrics import average_precision_score
        # 确保ID为负类(0)，OOD为正类(1)
        y_true = np.concatenate([np.zeros(len(id_values)), np.ones(len(ood_values))])
        y_score = np.concatenate([id_values, ood_values])
        return average_precision_score(y_true, y_score)

    def aggregate_alpha_priors(self, alpha_reports, client_data_sizes):
        if not alpha_reports:
            self.global_prior_alpha = torch.ones(self.num_classes) * 1.5
            return
        methods = self._compute_aggregated_prior_methods(alpha_reports, client_data_sizes)
        robust_name = getattr(self.args, "robust_prior_method", "coord_median")
        if robust_name not in methods:
            robust_name = "coord_median"
        robust_prior = torch.tensor(methods[robust_name], dtype=torch.float32)
        self.global_prior_alpha = torch.clamp(robust_prior, min=1.1, max=5.0)
        self.logger.info(
            f"Byzantine-robust aggregation ({robust_name}): "
            f"Min={self.global_prior_alpha.min().item():.2f}, "
            f"Max={self.global_prior_alpha.max().item():.2f}, "
            f"Mean={self.global_prior_alpha.mean().item():.2f}"
        )
