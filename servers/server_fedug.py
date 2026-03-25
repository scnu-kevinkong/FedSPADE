import time
from copy import deepcopy
import torch
import numpy as np
import traceback
import os
import csv
from utils.util import AverageMeter # Assuming utility exists
from servers.server_base import Server # Assuming base class exists
from clients.client_fedug import ClientFedUg # Ensure this path is correct
import torch.nn.functional as F

class ServerFedUg(Server):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        
        # Ensured logger checks before usage throughout the class
        self.log_func_info = self.logger.info if hasattr(self, 'logger') and self.logger else print
        self.log_func_warning = self.logger.warning if hasattr(self, 'logger') and self.logger else print
        self.log_func_error = self.logger.error if hasattr(self, 'logger') and self.logger else print

        if not hasattr(self, 'device'):
            self.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and hasattr(args, 'gpu_id') and args.gpu_id is not None else "cpu")
            self.log_func_info(f"Server device explicitly set to: {self.device}")

        if hasattr(self, 'model') and self.model is not None:
            self.model.to(self.device)
            self.log_func_info(f"Server model moved to device: {self.device}")
        else:
            self.log_func_error("Server model 'self.model' not found or not initialized after base class __init__.")
        
        self.clients = []
        for client_idx in range(self.num_clients):
            # Pass a reference to the server's logger to the client if needed, or client uses its own
            c = ClientFedUg(args, client_idx) # ClientFedUg will instantiate its own logger if base Client does
            self.clients.append(c)
        
        default_strength_from_args = getattr(args, 'default_prior_strength', None)
        num_classes_val = self.num_classes if hasattr(self, 'num_classes') and self.num_classes > 0 else 10.0

        if default_strength_from_args is not None:
            try:
                self.default_prior_strength = float(default_strength_from_args)
                if self.default_prior_strength <= 0:
                     self.log_func_warning(f"default_prior_strength ({self.default_prior_strength}) must be positive. Using num_classes ({num_classes_val}).")
                     self.default_prior_strength = float(num_classes_val)
            except (ValueError, TypeError):
                self.log_func_warning(
                    f"Invalid value for args.default_prior_strength: '{default_strength_from_args}'. "
                    f"Using default based on num_classes: {float(num_classes_val)}."
                )
                self.default_prior_strength = float(num_classes_val)
        else:
            self.default_prior_strength = float(num_classes_val)
        
        self.global_aggregated_evidence_prior = torch.ones(num_classes_val, device=self.device) * self.default_prior_strength
        
        if self.args.use_fedavg_momentum:
            self.server_momentum_beta = self.args.server_momentum_beta
            if hasattr(self.model, 'parameters') and any(True for _ in self.model.parameters()): 
                 self.server_momentum_buffer = {name: torch.zeros_like(param.data, device=self.device)
                                               for name, param in self.model.named_parameters() if param.requires_grad}
            else:
                 self.log_func_error("Server model not found or has no parameters for FedAvgM buffer.")
                 self.args.use_fedavg_momentum = False
        self.mechanism_clients_per_eval = int(getattr(args, "mechanism_profile_clients_per_eval", self.num_clients))
        self.mechanism_max_batches = int(getattr(args, "mechanism_hessian_max_batches", 1))
        self.mechanism_max_samples = int(getattr(args, "mechanism_hessian_max_samples_per_batch", 1))
        exp_name = getattr(args, "exp_name", "default")
        out_dir = os.path.join("results", "submission_summary")
        os.makedirs(out_dir, exist_ok=True)
        self.curvature_rounds_csv = os.path.join(out_dir, f"{exp_name}_curvature_rounds.csv")
        self.curvature_eigs_csv = os.path.join(out_dir, f"{exp_name}_curvature_eigs.csv")
        for p in [self.curvature_rounds_csv, self.curvature_eigs_csv]:
            if os.path.exists(p):
                os.remove(p)

    def _projected_curvature_metrics(self, H):
        H = np.asarray(H, dtype=np.float64)
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            return {"lambda_min_pos": 0.0, "trace": 0.0, "effective_rank": 0.0, "condition_number": float("inf"), "eigenvalues": np.array([], dtype=np.float64)}
        H = np.nan_to_num(H, nan=0.0, posinf=1e6, neginf=-1e6)
        H = 0.5 * (H + H.T)
        k = H.shape[0]
        P = np.eye(k, dtype=np.float64) - np.ones((k, k), dtype=np.float64) / float(k)
        PH = P.T @ H @ P
        PH = 0.5 * (PH + PH.T)
        eigvals = None
        for eps in [0.0, 1e-10, 1e-8, 1e-6]:
            try:
                eigvals = np.linalg.eigvalsh(PH + (eps * np.eye(k, dtype=np.float64) if eps > 0 else 0.0))
                break
            except np.linalg.LinAlgError:
                continue
        if eigvals is None:
            eigvals = np.zeros(k, dtype=np.float64)
        pos = eigvals[eigvals > 1e-10]
        lam = float(np.min(pos)) if pos.size > 0 else 0.0
        tr = float(np.trace(PH)) if np.isfinite(PH).all() else 0.0
        if pos.size > 0 and np.sum(pos) > 0:
            p = pos / np.sum(pos)
            er = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
            cn = float(np.max(pos) / max(np.min(pos), 1e-12))
        else:
            er = 0.0
            cn = float("inf")
        return {"lambda_min_pos": lam, "trace": tr, "effective_rank": er, "condition_number": cn, "eigenvalues": eigvals}

    def _client_hessian_unseen_block(self, client, global_state):
        label_count = np.array(getattr(client, "label_distribution", []))
        if label_count.size == 0:
            return None
        unseen = np.where(label_count <= 0)[0].astype(np.int64)
        if unseen.shape[0] < 2:
            return None
        client.model_global.load_state_dict(global_state)
        client.model_global.eval()
        client.model_global.to(client.device)
        loader = client.load_test_data()
        if not loader:
            client.model_global.to("cpu")
            return None
        k = unseen.shape[0]
        H = torch.zeros(k, k, device=client.device)
        total = 0
        idx = torch.tensor(unseen, device=client.device, dtype=torch.long)
        for b_idx, (x, y) in enumerate(loader):
            if b_idx >= self.mechanism_max_batches:
                break
            x = x.to(client.device)
            y = y.to(client.device)
            with torch.no_grad():
                logits_batch = client.model_global(x)
            take = min(self.mechanism_max_samples, logits_batch.size(0))
            for j in range(take):
                z_ref = logits_batch[j].detach()
                target = int(y[j].item())
                u0 = z_ref[idx].detach().clone().requires_grad_(True)
                def sample_loss(u):
                    z = z_ref.clone()
                    z[idx] = u
                    alpha = F.softplus(z) + 1.0
                    S = torch.sum(alpha)
                    return -(torch.digamma(alpha[target]) - torch.digamma(S))
                try:
                    Hij = torch.autograd.functional.hessian(sample_loss, u0)
                except RuntimeError:
                    continue
                if not torch.isfinite(Hij).all():
                    continue
                H = H + Hij.detach()
                total += 1
        client.model_global.to("cpu")
        if total == 0:
            return None
        H = H / total
        H = 0.5 * (H + H.t())
        H = torch.nan_to_num(H, nan=0.0, posinf=1e6, neginf=-1e6)
        return {"H": H.cpu().numpy().astype(np.float64, copy=False), "unseen": unseen}

    def _append_mechanism_profile(self, current_round):
        global_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        round_rows = []
        eig_rows = []
        max_clients = max(1, min(self.mechanism_clients_per_eval, len(self.clients)))
        for cid, client in enumerate(self.clients[:max_clients]):
            out = self._client_hessian_unseen_block(client, global_state)
            if out is None:
                continue
            metrics = self._projected_curvature_metrics(out["H"])
            round_rows.append({
                "round": int(current_round),
                "client_idx": int(cid),
                "unseen_count": int(out["unseen"].shape[0]),
                "lambda_min_pos": float(metrics["lambda_min_pos"]),
                "trace": float(metrics["trace"]),
                "effective_rank": float(metrics["effective_rank"]),
                "condition_number": float(metrics["condition_number"]),
            })
            eigvals = metrics["eigenvalues"]
            for rank, val in enumerate(eigvals[::-1], start=1):
                eig_rows.append({
                    "round": int(current_round),
                    "client_idx": int(cid),
                    "eig_rank_desc": int(rank),
                    "eig_value": float(val),
                })
        if len(round_rows) == 0:
            return
        write_round_header = not os.path.exists(self.curvature_rounds_csv)
        with open(self.curvature_rounds_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["round", "client_idx", "unseen_count", "lambda_min_pos", "trace", "effective_rank", "condition_number"])
            if write_round_header:
                w.writeheader()
            w.writerows(round_rows)
        if len(eig_rows) > 0:
            write_eig_header = not os.path.exists(self.curvature_eigs_csv)
            with open(self.curvature_eigs_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["round", "client_idx", "eig_rank_desc", "eig_value"])
                if write_eig_header:
                    w.writeheader()
                w.writerows(eig_rows)


    def train_round(self, r):
        self.log_func_info(f"--- Global Round {r}/{self.args.global_rounds} ---")
        round_start_time = time.time()

        num_active_clients = max(1, int(self.num_clients * self.args.sampling_prob))
        active_client_indices = np.random.choice(range(self.num_clients), num_active_clients, replace=False)
        self.active_clients = [self.clients[i] for i in active_client_indices]
        self.log_func_info(f"Selected {len(self.active_clients)} active clients for round {r}.")


        client_updates_model_weights = []
        client_alpha_reports_list = []
        client_data_sizes_list = []
        
        round_train_stats_meter = {
            'loss': AverageMeter(), 'acc': AverageMeter(),
            'nll': AverageMeter(), 'kl': AverageMeter(), 'avg_S': AverageMeter(),
            'prox_loss': AverageMeter() 
        }

        global_model_params_for_clients = None
        if hasattr(self.args, 'use_fedprox') and self.args.use_fedprox and self.args.fedprox_mu > 0: # check mu > 0
            global_model_params_for_clients = [param.detach().clone() for param in self.model.parameters()]

        for client_idx_in_selection, client in enumerate(self.active_clients):
            self.log_func_info(f"Training client {client.client_idx} ({client_idx_in_selection+1}/{len(self.active_clients)})...")
            client.set_model(self.model)
            client.set_global_prior_alpha(self.global_aggregated_evidence_prior)
            client.set_current_global_round(r)

            if global_model_params_for_clients is not None:
                client.set_global_model_params(global_model_params_for_clients)

            try:
                client_train_stats = client.train()
                if not client_train_stats: 
                    self.log_func_warning(f"Client {client.client_idx} returned empty stats. Skipping.")
                    continue

                model_weights, avg_alpha_report, data_size = client.get_update()
                
                client_updates_model_weights.append(model_weights)
                client_alpha_reports_list.append(avg_alpha_report)
                client_data_sizes_list.append(data_size)

                for key in round_train_stats_meter:
                    if key in client_train_stats and not (isinstance(client_train_stats[key], float) and np.isnan(client_train_stats[key])):
                        round_train_stats_meter[key].update(client_train_stats[key], data_size if data_size > 0 else 1) # Avoid weight 0
                
                log_loss = client_train_stats.get('loss', float('nan'))
                log_nll = client_train_stats.get('nll', float('nan'))
                log_kl = client_train_stats.get('kl', float('nan'))
                log_S = client_train_stats.get('avg_S', float('nan'))

                self.log_func_info(f"Client {client.client_idx}: Acc={client_train_stats.get('acc',0):.2f}, "
                                 f"Loss={log_loss if np.isnan(log_loss) else f'{log_loss:.4f}'}, "
                                 f"NLL={log_nll if np.isnan(log_nll) else f'{log_nll:.4f}'}, "
                                 f"KL={log_kl if np.isnan(log_kl) else f'{log_kl:.4f}'}, "
                                 f"AvgS={log_S if np.isnan(log_S) else f'{log_S:.2f}'}, "
                                 f"ProxLoss={client_train_stats.get('prox_loss',0):.4f}")

            except Exception as e:
                self.log_func_error(f"Error training client {client.client_idx} in round {r}: {e}")
                traceback.print_exc()
                continue
        
        if not client_updates_model_weights:
            self.log_func_warning(f"Round {r}: No successful client updates received. Skipping aggregation.")
            round_time = time.time() - round_start_time
            if hasattr(self, 'round_times') and isinstance(self.round_times, list): # Ensure round_times exists
                 self.round_times.append(round_time)
            self.log_func_info(f"Round Time (no aggregation): {round_time:.2f}s")
            if r % self.args.eval_gap == 0 or r == self.args.global_rounds:
                self.evaluate_models(current_round=r)
            
            return

        self.aggregate_client_model_weights(client_updates_model_weights, client_data_sizes_list)
        self.aggregate_client_alpha_reports(client_alpha_reports_list, client_data_sizes_list)

        round_time = time.time() - round_start_time
        if hasattr(self, 'round_times') and isinstance(self.round_times, list):
            self.round_times.append(round_time) 

        log_loss_avg = round_train_stats_meter['loss'].avg if round_train_stats_meter['loss'].count > 0 else float('nan')
        log_nll_avg = round_train_stats_meter['nll'].avg if round_train_stats_meter['nll'].count > 0 else float('nan')
        log_kl_avg = round_train_stats_meter['kl'].avg if round_train_stats_meter['kl'].count > 0 else float('nan')
        log_S_avg = round_train_stats_meter['avg_S'].avg if round_train_stats_meter['avg_S'].count > 0 else float('nan')


        log_message = (f"Round [{r}/{self.args.global_rounds}] Summary | Avg Client Train Stats:\t"
                       f"Loss [{log_loss_avg if np.isnan(log_loss_avg) else f'{log_loss_avg:.4f}'}]\t Acc [{round_train_stats_meter['acc'].avg:.2f}]\t"
                       f"NLL [{log_nll_avg if np.isnan(log_nll_avg) else f'{log_nll_avg:.4f}'}]\t KL [{log_kl_avg if np.isnan(log_kl_avg) else f'{log_kl_avg:.4f}'}]\t"
                       f"AvgS [{log_S_avg if np.isnan(log_S_avg) else f'{log_S_avg:.2f}'}]\t ProxLoss [{round_train_stats_meter['prox_loss'].avg:.4f}]")
        
        prior_display = self.global_aggregated_evidence_prior.cpu().numpy().round(4).tolist()
        log_message += f"\nGlobalPriorAlpha (first 10): {prior_display[:10]}" \
                       f"{'...' if len(prior_display) > 10 else ''}"
        log_message += f"\nRound Time: {round_time:.2f}s"
        self.log_func_info(log_message)

        if r % self.args.eval_gap == 0 or r == self.args.global_rounds:
            self.evaluate_models(current_round=r)


    def aggregate_client_model_weights(self, client_model_weights_list, client_data_sizes):
        if not client_model_weights_list: return
        
        total_data_size = sum(client_data_sizes)
        if total_data_size == 0:
            self.log_func_warning("Total data size from clients is 0. Cannot aggregate model weights.")
            return

        global_state_dict = self.model.state_dict()
        aggregated_state_dict = {name: torch.zeros_like(param_data, device=self.device)
                                 for name, param_data in global_state_dict.items()}

        for i, client_state_dict in enumerate(client_model_weights_list):
            weight = client_data_sizes[i] / total_data_size if total_data_size > 0 else 1.0/len(client_model_weights_list) # Handle total_data_size=0
            for name in aggregated_state_dict:
                if name in client_state_dict: # Ensure key exists
                    # Skip non-trainable parameters like num_batches_tracked
                    if 'num_batches_tracked' in name:
                        # For num_batches_tracked, just use the first client's value
                        if i == 0:
                            aggregated_state_dict[name] = client_state_dict[name].to(self.device)
                        continue
                    
                    # For trainable parameters, aggregate normally
                    client_param = client_state_dict[name].to(self.device)
                    if client_param.dtype != aggregated_state_dict[name].dtype:
                        # Convert to float for aggregation if needed
                        client_param = client_param.float()
                        if aggregated_state_dict[name].dtype != torch.float32:
                            aggregated_state_dict[name] = aggregated_state_dict[name].float()
                    
                    aggregated_state_dict[name] += client_param * weight
        
        if self.args.use_fedavg_momentum and hasattr(self, 'server_momentum_buffer'):
            new_global_state_dict = {}
            for name, avg_param_val in aggregated_state_dict.items():
                current_global_param = global_state_dict[name]
                if name in self.server_momentum_buffer:
                    update_step = avg_param_val - current_global_param # Difference from current global model
                    self.server_momentum_buffer[name] = self.server_momentum_beta * self.server_momentum_buffer[name] + \
                                                       (1 - self.server_momentum_beta) * update_step # Standard momentum update
                    
                    # Apply server learning rate to the momentum (often 1.0 for FedAvgM style)
                    new_global_state_dict[name] = current_global_param + self.args.server_lr * self.server_momentum_buffer[name]
                else: 
                    new_global_state_dict[name] = avg_param_val # Fallback if not in buffer (should not happen for trainable params)
                    if name in global_state_dict and global_state_dict[name].requires_grad:
                         self.log_func_warning(f"Parameter {name} requires grad but not found in server momentum buffer.")
            self.model.load_state_dict(new_global_state_dict)
        else:
            self.model.load_state_dict(aggregated_state_dict)


    def aggregate_client_alpha_reports(self, client_alpha_reports_list, client_data_sizes):
        num_classes_val = self.num_classes if hasattr(self, 'num_classes') and self.num_classes > 0 else 10

        if not client_alpha_reports_list:
            self.log_func_warning("No client alpha reports to aggregate.")
            self.global_aggregated_evidence_prior = torch.ones(num_classes_val, device=self.device) * self.default_prior_strength
            return

        total_data_size = sum(client_data_sizes)
        # Filter out None or non-tensor reports and corresponding sizes
        valid_reports_data = [(report, size) for report, size in zip(client_alpha_reports_list, client_data_sizes) 
                              if isinstance(report, torch.Tensor) and report.numel() == num_classes_val and size >= 1 and torch.all(report > 0)]
        
        if not valid_reports_data:
            self.log_func_warning(f"No valid client alpha reports (all None, non-tensor, or wrong size for {num_classes_val} classes). Resetting to default prior.")
            self.global_aggregated_evidence_prior = torch.ones(num_classes_val, device=self.device) * self.default_prior_strength
            return
        
        valid_alpha_reports, valid_data_sizes = zip(*valid_reports_data)
        # 权重加下限，极小样本客户端权重平滑
        min_weight = 5
        valid_data_sizes = [max(size, min_weight) for size in valid_data_sizes]
        total_valid_data_size = sum(valid_data_sizes)

        if total_valid_data_size == 0: 
            self.log_func_warning("Total valid data size for alpha reports is 0. Resetting to default prior.")
            self.global_aggregated_evidence_prior = torch.ones(num_classes_val, device=self.device) * self.default_prior_strength
            return

        weighted_alpha_sum = torch.zeros_like(valid_alpha_reports[0], device=self.device, dtype=torch.float32) # Use first valid report for shape
        
        for i, alpha_report in enumerate(valid_alpha_reports):
            weight = valid_data_sizes[i] / total_valid_data_size
            weighted_alpha_sum += weight * alpha_report.to(self.device) 
        
        current_S0_tensor = weighted_alpha_sum.sum()

        target_S0_from_args = getattr(self.args, 'target_global_prior_strength', None)
        target_S0 = self.default_prior_strength # Default to default_prior_strength

        if target_S0_from_args is not None:
            try:
                target_S0 = float(target_S0_from_args)
                if target_S0 <= 0:
                    self.log_func_warning(
                        f"args.target_global_prior_strength ({target_S0}) must be positive. "
                        f"Using default_prior_strength: {self.default_prior_strength}."
                    )
                    target_S0 = self.default_prior_strength
            except (ValueError, TypeError):
                self.log_func_warning(
                    f"Invalid value for args.target_global_prior_strength: '{target_S0_from_args}'. "
                    f"Using default_prior_strength: {self.default_prior_strength}."
                )
                target_S0 = self.default_prior_strength
        
        if current_S0_tensor.item() > 1e-6 and target_S0 > 0: 
            new_prior = (weighted_alpha_sum / current_S0_tensor) * target_S0
            new_prior = torch.clamp(new_prior, min=1e-6)
            max_div_min = (new_prior.max() / (new_prior.min() + 1e-6)).item()
            threshold = 10.0
            lam = min(0.5, max_div_min / threshold)
            if lam > 0:
                uniform = torch.ones_like(new_prior) * new_prior.mean()
                new_prior = (1-lam)*new_prior + lam*uniform
                self.log_func_warning(f"Global prior extremely imbalanced (max/min={max_div_min:.1f}), applied adaptive smoothing lam={lam:.2f}.")
            self.global_aggregated_evidence_prior = new_prior
        else:
            self.log_func_warning(f"Aggregated alpha sum {current_S0_tensor.item():.4f} too small or target_S0 {target_S0} not positive. Resetting to default prior.")
            self.global_aggregated_evidence_prior = torch.ones(num_classes_val, device=self.device) * self.default_prior_strength

        self.global_aggregated_evidence_prior = torch.clamp(self.global_aggregated_evidence_prior, min=1e-6) 


    def evaluate_models(self, current_round):
        self.log_func_info(f"--- Evaluating models at round {current_round} ---")
        num_classes_val = self.num_classes if hasattr(self, 'num_classes') and self.num_classes > 0 else 10
        # 全局模型评估
        global_model_avg_acc = AverageMeter()
        global_model_avg_loss = AverageMeter()
        global_model_avg_nll = AverageMeter()
        global_model_avg_kl = AverageMeter()
        global_model_avg_S = AverageMeter()
        global_model_avg_acc_softmax = AverageMeter()
        global_model_avg_ece = AverageMeter()
        global_model_avg_brier = AverageMeter()
        global_model_avg_auroc = AverageMeter()
        global_model_avg_fpr95 = AverageMeter()
        for client in self.clients:
            edl_stats = client.evaluate_edl_stats(use_personal=False)
            client_weight = client.num_test_data if hasattr(client, 'num_test_data') and client.num_test_data > 0 else 1
            if not (isinstance(edl_stats['test_loss'], float) and np.isnan(edl_stats['test_loss'])):
                global_model_avg_loss.update(edl_stats['test_loss'], client_weight)
            if not (isinstance(edl_stats['test_nll'], float) and np.isnan(edl_stats['test_nll'])):
                global_model_avg_nll.update(edl_stats['test_nll'], client_weight)
            if not (isinstance(edl_stats['test_kl'], float) and np.isnan(edl_stats['test_kl'])):
                kl_val = edl_stats['test_kl'] if isinstance(edl_stats['test_kl'], (int, float)) else edl_stats['test_kl'].item()
                global_model_avg_kl.update(kl_val, client_weight)
            global_model_avg_acc.update(edl_stats['test_acc'], client_weight)
            if 'test_acc_softmax' in edl_stats:
                global_model_avg_acc_softmax.update(edl_stats['test_acc_softmax'], client_weight)
            if 'test_ece' in edl_stats and np.isfinite(edl_stats['test_ece']):
                global_model_avg_ece.update(edl_stats['test_ece'], client_weight)
            if 'test_brier' in edl_stats and np.isfinite(edl_stats['test_brier']):
                global_model_avg_brier.update(edl_stats['test_brier'], client_weight)
            if 'test_ood_auroc' in edl_stats and np.isfinite(edl_stats['test_ood_auroc']):
                global_model_avg_auroc.update(edl_stats['test_ood_auroc'], client_weight)
            if 'test_ood_fpr95' in edl_stats and np.isfinite(edl_stats['test_ood_fpr95']):
                global_model_avg_fpr95.update(edl_stats['test_ood_fpr95'], client_weight)
            global_model_avg_S.update(edl_stats['test_avg_S'], client_weight)
        log_loss_avg = global_model_avg_loss.avg if global_model_avg_loss.count > 0 else float('nan')
        log_nll_avg = global_model_avg_nll.avg if global_model_avg_nll.count > 0 else float('nan')
        log_kl_avg = global_model_avg_kl.avg if global_model_avg_kl.count > 0 else float('nan')
        self.log_func_info(f"Global Model Evaluation (Round {current_round}): \n"
                         f"  Avg Test Acc: {global_model_avg_acc.avg:.2f}%\n"
                         f"  Avg Test Acc Softmax: {global_model_avg_acc_softmax.avg:.2f}%\n"
                         f"  Avg Test Loss: {log_loss_avg if np.isnan(log_loss_avg) else f'{log_loss_avg:.4f}'}\n"
                         f"  Avg Test NLL: {log_nll_avg if np.isnan(log_nll_avg) else f'{log_nll_avg:.4f}'}\n"
                         f"  Avg Test KL (vs flat prior): {log_kl_avg if np.isnan(log_kl_avg) else f'{log_kl_avg:.4f}'}\n"
                         f"  Avg Test S: {global_model_avg_S.avg:.2f}")
        # 个性化模型评估
        personalized_model_avg_acc = AverageMeter()
        personalized_model_avg_loss = AverageMeter()
        personalized_model_avg_nll = AverageMeter()
        personalized_model_avg_kl = AverageMeter()
        personalized_model_avg_S = AverageMeter()
        personalized_model_avg_acc_softmax = AverageMeter()
        personalized_model_avg_ece = AverageMeter()
        personalized_model_avg_brier = AverageMeter()
        personalized_model_avg_auroc = AverageMeter()
        personalized_model_avg_fpr95 = AverageMeter()
        for client in self.clients:
            edl_stats_personalized = client.evaluate_edl_stats(use_personal=True)
            client_weight = client.num_test_data if hasattr(client, 'num_test_data') and client.num_test_data > 0 else 1
            if not (isinstance(edl_stats_personalized['test_loss'], float) and np.isnan(edl_stats_personalized['test_loss'])):
                personalized_model_avg_loss.update(edl_stats_personalized['test_loss'], client_weight)
            if not (isinstance(edl_stats_personalized['test_nll'], float) and np.isnan(edl_stats_personalized['test_nll'])):
                personalized_model_avg_nll.update(edl_stats_personalized['test_nll'], client_weight)
            if not (isinstance(edl_stats_personalized['test_kl'], float) and np.isnan(edl_stats_personalized['test_kl'])):
                kl_val_p = edl_stats_personalized['test_kl'] if isinstance(edl_stats_personalized['test_kl'], (int, float)) else edl_stats_personalized['test_kl'].item()
                personalized_model_avg_kl.update(kl_val_p, client_weight)
            personalized_model_avg_acc.update(edl_stats_personalized['test_acc'], client_weight)
            if 'test_acc_softmax' in edl_stats_personalized:
                personalized_model_avg_acc_softmax.update(edl_stats_personalized['test_acc_softmax'], client_weight)
            if 'test_ece' in edl_stats_personalized and np.isfinite(edl_stats_personalized['test_ece']):
                personalized_model_avg_ece.update(edl_stats_personalized['test_ece'], client_weight)
            if 'test_brier' in edl_stats_personalized and np.isfinite(edl_stats_personalized['test_brier']):
                personalized_model_avg_brier.update(edl_stats_personalized['test_brier'], client_weight)
            if 'test_ood_auroc' in edl_stats_personalized and np.isfinite(edl_stats_personalized['test_ood_auroc']):
                personalized_model_avg_auroc.update(edl_stats_personalized['test_ood_auroc'], client_weight)
            if 'test_ood_fpr95' in edl_stats_personalized and np.isfinite(edl_stats_personalized['test_ood_fpr95']):
                personalized_model_avg_fpr95.update(edl_stats_personalized['test_ood_fpr95'], client_weight)
            personalized_model_avg_S.update(edl_stats_personalized['test_avg_S'], client_weight)
        log_loss_perso_avg = personalized_model_avg_loss.avg if personalized_model_avg_loss.count > 0 else float('nan')
        log_nll_perso_avg = personalized_model_avg_nll.avg if personalized_model_avg_nll.count > 0 else float('nan')
        log_kl_perso_avg = personalized_model_avg_kl.avg if personalized_model_avg_kl.count > 0 else float('nan')
        self.log_func_info(f"Personalized Model Evaluation (Round {current_round}): \n"
                            f"  Avg Test Acc: {personalized_model_avg_acc.avg:.2f}%\n"
                            f"  Avg Test Acc Softmax: {personalized_model_avg_acc_softmax.avg:.2f}%\n"
                            f"  Avg Test Loss: {log_loss_perso_avg if np.isnan(log_loss_perso_avg) else f'{log_loss_perso_avg:.4f}'}\n"
                            f"  Avg Test NLL: {log_nll_perso_avg if np.isnan(log_nll_perso_avg) else f'{log_nll_perso_avg:.4f}'}\n"
                            f"  Avg Test KL (vs flat prior): {log_kl_perso_avg if np.isnan(log_kl_perso_avg) else f'{log_kl_perso_avg:.4f}'}\n"
                            f"  Avg Test S: {personalized_model_avg_S.avg:.2f}")
        if personalized_model_avg_brier.count > 0:
            self.log_func_info(f"[Metric 1] Personalized Brier Score | Mean: {personalized_model_avg_brier.avg:.4f}")
        if personalized_model_avg_ece.count > 0:
            self.log_func_info(f"[Metric 1] Personalized ECE | Mean: {personalized_model_avg_ece.avg:.4f}")
        if personalized_model_avg_auroc.count > 0:
            self.log_func_info(f"[OOD Eval] OOD AUC-ROC (Epistemic): {personalized_model_avg_auroc.avg:.4f}")
        if personalized_model_avg_fpr95.count > 0:
            self.log_func_info(f"[Metric 4] OOD FPR@95%TPR: {personalized_model_avg_fpr95.avg:.4f}")
        self._append_mechanism_profile(current_round)

    def train(self):
        self.log_func_info("Starting Optimized FedUG Training Process")
        if not hasattr(self, 'round_times'): # Ensure round_times is initialized if not by base
            self.round_times = []
        for r in range(1, self.args.global_rounds + 1):
            self.train_round(r)
        self.log_func_info("Federated Training Finished.")
