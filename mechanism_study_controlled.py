import argparse
import os
import random
import re
from dataclasses import dataclass

os.environ.setdefault("DATA_PATH", os.path.join(os.getcwd(), "data"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(os.getcwd(), ".torchinductor_cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)
os.makedirs(os.path.join(os.environ["XDG_CACHE_HOME"], "torch", "kernels"), exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from clients.client_base import Client
from data.utils.loader import get_base_dataset
from models import model_dict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def kl_dirichlet(alpha, beta):
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    sum_beta = torch.sum(beta, dim=1, keepdim=True)
    t1 = torch.lgamma(sum_alpha) - torch.lgamma(sum_beta)
    t2 = torch.sum(torch.lgamma(beta) - torch.lgamma(alpha), dim=1, keepdim=True)
    t3 = torch.sum((alpha - beta) * (torch.digamma(alpha) - torch.digamma(sum_alpha)), dim=1, keepdim=True)
    return (t1 + t2 + t3).mean()


def build_runtime_args(cfg, seed):
    ns = argparse.Namespace()
    ns.dataset = "cifar10"
    ns.num_classes = 10
    ns.partition_path = cfg.partition_path
    ns.augmented = False
    ns.global_rounds = cfg.global_rounds
    ns.local_epochs = cfg.local_epochs
    ns.lr = cfg.lr
    ns.wd = cfg.wd
    ns.batch_size = cfg.batch_size
    ns.eval_gap = cfg.log_gap
    ns.train_prop = 1.0
    ns.method = "FedSPADE"
    ns.num_clients = cfg.num_clients
    ns.sampling_prob = cfg.sampling_prob
    ns.device = cfg.device
    ns.model_name = "cifaredl"
    ns.p_epochs = 1
    ns.single_beta = False
    ns.local_beta = False
    ns.exp_name = f"mechanism_seed_{seed}"
    ns.beta_tau = 50
    ns.min_samples = 20
    ns.in_channels = 3
    ns.D_h_taylor = 128
    ns.N_taylor = 3
    ns.taylor_activation = "gelu"
    ns.uncertainty = True
    ns.momentum = cfg.momentum
    ns.ood_dataset = ""
    ns.base_dataset = get_base_dataset(ns)
    ns.model = model_dict[ns.model_name](num_classes=ns.num_classes, in_channels=ns.in_channels)
    return ns


@dataclass
class StudyConfig:
    output_dir: str
    partition_path: str
    global_rounds: int
    local_epochs: int
    batch_size: int
    num_clients: int
    sampling_prob: float
    lr: float
    wd: float
    momentum: float
    log_gap: int
    seeds: list
    device: str
    fedprox_mu: float
    gc_weight: float
    fsa_weight: float
    afr_weight: float
    afr_noise_std: float
    max_hessian_batches: int
    max_hessian_samples_per_batch: int


class MechanismClient(Client):
    def __init__(self, args, client_idx, cfg):
        super().__init__(args, client_idx, is_corrupted=False)
        self.cfg = cfg

    def train(self):
        raise NotImplementedError

    def _forward_with_features(self, x):
        try:
            features, logits, _ = self.model(x, return_feat_alpha=True)
            return features, logits
        except TypeError:
            logits = self.model(x)
            return None, logits

    def _edl_nll(self, logits, y):
        alpha = F.softplus(logits) + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
        return -torch.sum(y_one_hot * (torch.digamma(alpha) - torch.digamma(S)), dim=1).mean()

    def _feature_compactness_loss(self, features, y):
        if features is None:
            return torch.tensor(0.0, device=self.device)
        f = features.view(features.size(0), -1)
        total = torch.tensor(0.0, device=self.device)
        count = 0
        for cls in torch.unique(y):
            m = (y == cls)
            if torch.sum(m) > 1:
                feat_cls = f[m]
                center = feat_cls.mean(dim=0, keepdim=True)
                total = total + ((feat_cls - center) ** 2).mean()
                count += 1
        if count == 0:
            return torch.tensor(0.0, device=self.device)
        return total / count

    def _afr_loss(self, x):
        noise = torch.randn_like(x) * self.cfg.afr_noise_std
        x_noisy = torch.clamp(x + noise, -3.0, 3.0)
        logits_noisy = self.model(x_noisy)
        alpha_noisy = F.softplus(logits_noisy) + 1.0
        uncertainty = self.num_classes / torch.sum(alpha_noisy, dim=1)
        return -uncertainty.mean()

    def local_update(self, global_state, method_name, global_prior_alpha):
        self.model.load_state_dict(global_state)
        self.model.train()
        self.model.to(self.device)
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.cfg.lr,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.wd,
        )
        initial_params = {k: v.detach().clone().to(self.device) for k, v in self.model.state_dict().items()}
        train_loader = self.load_train_data(drop_last=True)
        steps = 0
        for _ in range(self.cfg.local_epochs):
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                features, logits = self._forward_with_features(x)
                loss = self._edl_nll(logits, y)
                if method_name == "FedSPADE":
                    alpha = F.softplus(logits) + 1.0
                    prior = global_prior_alpha.unsqueeze(0).expand_as(alpha)
                    loss = loss + self.cfg.gc_weight * kl_dirichlet(alpha, prior)
                    loss = loss + self.cfg.fsa_weight * self._feature_compactness_loss(features, y)
                    loss = loss + self.cfg.afr_weight * self._afr_loss(x)
                if method_name == "FedProx-EDL":
                    prox = torch.tensor(0.0, device=self.device)
                    for k, p in self.model.named_parameters():
                        if k in initial_params:
                            prox = prox + ((p - initial_params[k]) ** 2).sum()
                    loss = loss + 0.5 * self.cfg.fedprox_mu * prox
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                steps += 1
        final_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        delta = {k: global_state[k].detach().cpu() - final_state[k] for k in final_state}
        self.model.to("cpu")
        alpha_report = self.compute_alpha_report(global_state)
        return {
            "client_idx": self.client_idx,
            "state": final_state,
            "delta": delta,
            "steps": max(steps, 1),
            "num_train": self.num_train,
            "alpha_report": alpha_report,
        }

    def compute_alpha_report(self, global_state):
        self.model.load_state_dict(global_state)
        self.model.eval()
        self.model.to(self.device)
        loader = self.load_train_data(drop_last=False)
        alpha_list = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                logits = self.model(x)
                alpha = F.softplus(logits) + 1.0
                alpha_list.append(alpha.cpu())
        self.model.to("cpu")
        if len(alpha_list) == 0:
            return torch.ones(self.num_classes) * 2.0
        return torch.cat(alpha_list, dim=0).mean(dim=0)

    def unseen_classes(self):
        label_count = np.array(self.label_distribution)
        return np.where(label_count <= 0)[0].astype(np.int64)

    def hessian_unseen_block(self, global_state):
        unseen = self.unseen_classes()
        if unseen.shape[0] < 2:
            return None
        self.model.load_state_dict(global_state)
        self.model.eval()
        self.model.to(self.device)
        loader = self.load_test_data()
        k = unseen.shape[0]
        H = torch.zeros(k, k, device=self.device)
        total = 0
        max_batches = self.cfg.max_hessian_batches
        max_samples = self.cfg.max_hessian_samples_per_batch
        idx = torch.tensor(unseen, device=self.device, dtype=torch.long)
        for b_idx, (x, y) in enumerate(loader):
            if b_idx >= max_batches:
                break
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                logits_batch = self.model(x)
            bsz = logits_batch.size(0)
            local_take = min(max_samples, bsz)
            for j in range(local_take):
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
        self.model.to("cpu")
        if total == 0:
            return None
        H = H / total
        H = 0.5 * (H + H.t())
        H = torch.nan_to_num(H, nan=0.0, posinf=1e6, neginf=-1e6)
        H = H.cpu().numpy().astype(np.float64, copy=False)
        return {"H": H, "unseen": unseen}


def aggregate_weighted(states, weights):
    out = {}
    keys = states[0].keys()
    for k in keys:
        acc = None
        for s, w in zip(states, weights):
            v = s[k].float() * w
            acc = v if acc is None else (acc + v)
        out[k] = acc
    return out


def aggregate_fednova(global_state, client_payloads):
    weights = np.array([p["num_train"] for p in client_payloads], dtype=np.float64)
    weights = weights / max(weights.sum(), 1.0)
    tau_eff = float(np.sum([w * p["steps"] for w, p in zip(weights, client_payloads)]))
    new_state = {}
    for k in global_state.keys():
        d_avg = None
        for w, p in zip(weights, client_payloads):
            comp = p["delta"][k].float() / float(p["steps"])
            d_avg = comp * w if d_avg is None else (d_avg + comp * w)
        new_state[k] = (global_state[k].float() - tau_eff * d_avg).detach().cpu()
    return new_state


def projected_curvature_metrics(H):
    H = np.asarray(H, dtype=np.float64)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        return {
            "lambda_min_pos": 0.0,
            "trace": 0.0,
            "effective_rank": 0.0,
            "condition_number": float("inf"),
            "eigenvalues": np.array([], dtype=np.float64),
        }
    H = np.nan_to_num(H, nan=0.0, posinf=1e6, neginf=-1e6)
    H = 0.5 * (H + H.T)
    k = H.shape[0]
    P = np.eye(k, dtype=np.float64) - np.ones((k, k), dtype=np.float64) / float(k)
    PH = P.T @ H @ P
    PH = 0.5 * (PH + PH.T)
    eigvals = None
    for eps in [0.0, 1e-10, 1e-8, 1e-6, 1e-4]:
        try:
            if eps > 0.0:
                eigvals = np.linalg.eigvalsh(PH + eps * np.eye(k, dtype=np.float64))
            else:
                eigvals = np.linalg.eigvalsh(PH)
            break
        except np.linalg.LinAlgError:
            continue
    if eigvals is None:
        eigvals = np.zeros(k, dtype=np.float64)
    pos = eigvals[eigvals > 1e-10]
    lambda_min_pos = float(np.min(pos)) if pos.size > 0 else 0.0
    tr = float(np.trace(PH)) if np.isfinite(PH).all() else 0.0
    if pos.size > 0 and np.sum(pos) > 0:
        p = pos / np.sum(pos)
        erank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
        cond = float(np.max(pos) / max(np.min(pos), 1e-12))
    else:
        erank = 0.0
        cond = float("inf")
    return {
        "lambda_min_pos": lambda_min_pos,
        "trace": tr,
        "effective_rank": erank,
        "condition_number": cond,
        "eigenvalues": eigvals,
    }


def run_single_method(cfg, method_name, seed, output_dir):
    round_path = os.path.join(output_dir, f"curvature_rounds_{method_name}_seed{seed}.csv")
    eig_path = os.path.join(output_dir, f"curvature_eigs_{method_name}_seed{seed}.csv")
    if os.path.exists(round_path) and os.path.exists(eig_path):
        print(f"Loading existing results for {method_name} seed {seed}")
        return pd.read_csv(round_path), pd.read_csv(eig_path)
    
    set_seed(seed)
    args = build_runtime_args(cfg, seed)
    clients = [MechanismClient(args, i, cfg) for i in range(cfg.num_clients)]
    global_model = model_dict["cifaredl"](num_classes=args.num_classes, in_channels=args.in_channels)
    global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
    global_prior_alpha = torch.ones(args.num_classes) * 2.0
    rng = np.random.default_rng(seed)
    round_rows = []
    eig_rows = []
    for round_idx in range(1, cfg.global_rounds + 1):
        bern = rng.binomial(1, cfg.sampling_prob, size=cfg.num_clients)
        active_ids = np.where(bern == 1)[0]
        if active_ids.size == 0:
            active_ids = np.array([int(rng.integers(0, cfg.num_clients))])
        payloads = []
        for cid in active_ids.tolist():
            payload = clients[cid].local_update(global_state, method_name, global_prior_alpha.to(cfg.device))
            payloads.append(payload)
        if method_name == "FedNova-EDL":
            global_state = aggregate_fednova(global_state, payloads)
        else:
            w = np.array([p["num_train"] for p in payloads], dtype=np.float64)
            w = w / max(w.sum(), 1.0)
            states = [p["state"] for p in payloads]
            global_state = aggregate_weighted(states, w)
        if method_name == "FedSPADE":
            w = np.array([p["num_train"] for p in payloads], dtype=np.float64)
            w = w / max(w.sum(), 1.0)
            reports = [p["alpha_report"] for p in payloads]
            gp = torch.zeros_like(reports[0])
            for wi, rep in zip(w, reports):
                gp += float(wi) * rep
            global_prior_alpha = torch.clamp(gp, min=1.05, max=50.0).cpu()
        if (round_idx % cfg.log_gap == 0) or (round_idx == cfg.global_rounds):
            # Evaluate metrics. Before doing so, let's keep the pre-aggregation global state
            # around if we need it for measuring drift? Wait, the loop evaluates AFTER aggregation.
            # But the 'payloads' contain the local state AFTER local update.
            # And 'global_state' is now the NEW aggregated state.
            # Drift is usually measured as || w_local - w_global_old || or || w_local - w_global_new ||.
            # Let's use || local_w - global_state ||
            for cid, client in enumerate(clients):
                out = client.hessian_unseen_block(global_state)
                if out is None:
                    continue
                metrics = projected_curvature_metrics(out["H"])
                unseen_count = int(out["unseen"].shape[0])
                
                # Compute unseen drift norm
                drift_norm = 0.0
                if "fc_evidence.weight" in global_state:
                    local_w = None
                    for p in payloads:
                        if p.get("client_idx", -1) == cid:
                            local_w = p["state"]["fc_evidence.weight"]
                            break
                    
                    if local_w is not None:
                        # Drift measured against the freshly aggregated global model
                        w_global = global_state["fc_evidence.weight"].cpu()
                        w_local_cpu = local_w.cpu()
                        # out["unseen"] is a list or tensor of indices. Make sure it's cpu tensor
                        unseen_idx = out["unseen"].cpu() if isinstance(out["unseen"], torch.Tensor) else out["unseen"]
                        drift = w_local_cpu[unseen_idx] - w_global[unseen_idx]
                        drift_norm = torch.norm(drift).item()
                        if drift_norm == 0.0:
                            # If they are exactly the same, maybe the client didn't update this layer much
                            drift_norm += 1e-6
                    else:
                        # If client wasn't active this round, we don't have its latest local_w.
                        # Let's just log it as nan
                        drift_norm = float('nan')

                round_rows.append(
                    {
                        "method": method_name,
                        "seed": seed,
                        "round": round_idx,
                        "client_idx": cid,
                        "unseen_count": unseen_count,
                        "lambda_min_pos": metrics["lambda_min_pos"],
                        "trace": metrics["trace"],
                        "effective_rank": metrics["effective_rank"],
                        "condition_number": metrics["condition_number"],
                        "unseen_drift_norm": drift_norm,
                    }
                )
                eigvals = metrics["eigenvalues"]
                for rank, val in enumerate(eigvals[::-1], start=1):
                    eig_rows.append(
                        {
                            "method": method_name,
                            "seed": seed,
                            "round": round_idx,
                            "client_idx": cid,
                            "eig_rank_desc": rank,
                            "eig_value": float(val),
                        }
                    )
    round_df = pd.DataFrame(round_rows)
    eig_df = pd.DataFrame(eig_rows)
    round_path = os.path.join(output_dir, f"curvature_rounds_{method_name}_seed{seed}.csv")
    eig_path = os.path.join(output_dir, f"curvature_eigs_{method_name}_seed{seed}.csv")
    round_df.to_csv(round_path, index=False)
    eig_df.to_csv(eig_path, index=False)
    return round_df, eig_df


def _method_color(name):
    if name == "FedSPADE":
        return "#1f77b4"
    if name == "FedProx-EDL":
        return "#d62728"
    return "#2ca02c"


def make_figure(all_round_df, all_eig_df, cfg, output_dir):
    plt.style.use("seaborn-v0_8-whitegrid")
    methods = ["FedProx-EDL", "FedNova-EDL", "FedSPADE"]
    final_round = int(all_round_df["round"].max())
    fdf = all_round_df[all_round_df["round"] == final_round]

    def draw_panel_a(ax):
        for m in methods:
            mdf = all_round_df[all_round_df["method"] == m]
            if mdf.empty:
                continue
            per_seed = mdf.groupby(["round", "seed"])["lambda_min_pos"].mean().reset_index()
            g = per_seed.groupby("round")["lambda_min_pos"]
            mean = g.mean()
            std = g.std(ddof=1).fillna(0.0)
            n = g.count().clip(lower=1)
            ci95 = 1.96 * std / np.sqrt(n)
            x = mean.index.to_numpy()
            y = mean.to_numpy()
            c = _method_color(m)
            ax.plot(x, y, label=m, color=c, linewidth=2.2)
            ax.fill_between(x, y - ci95.to_numpy(), y + ci95.to_numpy(), color=c, alpha=0.2)
        ax.set_xlabel("Communication Round")
        ax.set_ylabel(r"Mean $\lambda_{\min}^{+}$")
        ax.legend(frameon=True)

    def draw_panel_b(ax):
        valid_labels = []
        valid_data = []
        valid_colors = []
        for m in methods:
            d = fdf[fdf["method"] == m]["condition_number"].values.astype(np.float64)
            d = d[np.isfinite(d)]
            if d.size > 0:
                valid_labels.append(m)
                valid_data.append(d)
                valid_colors.append(_method_color(m))
        parts = ax.violinplot(valid_data, showmeans=True, showextrema=False)
        for pc, col in zip(parts["bodies"], valid_colors):
            pc.set_facecolor(col)
            pc.set_alpha(0.35)
        if "cmeans" in parts:
            parts["cmeans"].set_color("black")
        ax.set_xticks(np.arange(1, len(valid_labels) + 1))
        ax.set_xticklabels(valid_labels, rotation=12)
        ax.set_ylabel("Condition Number")
        ax.set_yscale("log")

    def draw_panel_c(ax):
        for m in methods:
            edf = all_eig_df[
                (all_eig_df["method"] == m)
                & (all_eig_df["round"] == final_round)
            ]
            if edf.empty:
                continue
            per_client = edf.groupby(["client_idx", "eig_rank_desc"])["eig_value"].mean().reset_index()
            pivot = per_client.pivot(index="client_idx", columns="eig_rank_desc", values="eig_value").sort_index(axis=1)
            if pivot.empty:
                continue
            x = pivot.columns.to_numpy(dtype=np.float64)
            mean_spec = pivot.mean(axis=0).to_numpy(dtype=np.float64)
            std_spec = pivot.std(axis=0, ddof=1).fillna(0.0).to_numpy(dtype=np.float64)
            n_clients = pivot.count(axis=0).to_numpy(dtype=np.float64)
            ci95 = 1.96 * std_spec / np.sqrt(np.clip(n_clients, 1.0, None))
            c = _method_color(m)
            ax.plot(x, mean_spec, linewidth=2.1, alpha=0.95, label=m, color=c)
            ax.fill_between(x, mean_spec - ci95, mean_spec + ci95, color=c, alpha=0.16)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Eigenvalue")
        ax.legend(fontsize=8, frameon=True)

    fig = plt.figure(figsize=(15.6, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.16)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    all_spec_ax = fig.add_subplot(gs[0, 2])
    draw_panel_a(ax)
    draw_panel_b(ax2)
    draw_panel_c(all_spec_ax)

    appendix_fig, appendix_ax = plt.subplots(1, 1, figsize=(5.2, 4.5))
    appendix_data = []
    appendix_labels = []
    appendix_colors = []
    for m in methods:
        d = fdf[fdf["method"] == m]["lambda_min_pos"].values.astype(np.float64)
        d = d[np.isfinite(d)]
        if d.size > 0:
            appendix_data.append(d)
            appendix_labels.append(m)
            appendix_colors.append(_method_color(m))
    if len(appendix_data) > 0:
        appendix_parts = appendix_ax.violinplot(appendix_data, showmeans=True, showextrema=False)
        for pc, col in zip(appendix_parts["bodies"], appendix_colors):
            pc.set_facecolor(col)
            pc.set_alpha(0.35)
        if "cmeans" in appendix_parts:
            appendix_parts["cmeans"].set_color("black")
        appendix_ax.set_xticks(np.arange(1, len(appendix_labels) + 1))
        appendix_ax.set_xticklabels(appendix_labels, rotation=10)
    appendix_ax.set_ylabel(r"$\lambda_{\min}^{+}$")
    appendix_ax.set_title(r"Appendix: Final-round $\lambda_{\min}^{+}$ violin")
    appendix_fig.tight_layout()
    panel_size = (5.2, 5.4)
    panel_a_fig, panel_a_ax = plt.subplots(1, 1, figsize=panel_size, constrained_layout=True)
    draw_panel_a(panel_a_ax)
    panel_b_fig, panel_b_ax = plt.subplots(1, 1, figsize=panel_size, constrained_layout=True)
    draw_panel_b(panel_b_ax)
    panel_c_fig, panel_c_ax = plt.subplots(1, 1, figsize=panel_size, constrained_layout=True)
    draw_panel_c(panel_c_ax)
    png = os.path.join(output_dir, "mechanism_curvature_3panel.png")
    pdf = os.path.join(output_dir, "mechanism_curvature_3panel.pdf")
    panel_a_png = os.path.join(output_dir, "mechanism_curvature_panel_a.png")
    panel_a_pdf = os.path.join(output_dir, "mechanism_curvature_panel_a.pdf")
    panel_b_png = os.path.join(output_dir, "mechanism_curvature_panel_b.png")
    panel_b_pdf = os.path.join(output_dir, "mechanism_curvature_panel_b.pdf")
    panel_c_png = os.path.join(output_dir, "mechanism_curvature_panel_c.png")
    panel_c_pdf = os.path.join(output_dir, "mechanism_curvature_panel_c.pdf")
    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(pdf, dpi=400, bbox_inches="tight")
    panel_a_fig.savefig(panel_a_png, dpi=400, bbox_inches="tight")
    panel_a_fig.savefig(panel_a_pdf, dpi=400, bbox_inches="tight")
    panel_b_fig.savefig(panel_b_png, dpi=400, bbox_inches="tight")
    panel_b_fig.savefig(panel_b_pdf, dpi=400, bbox_inches="tight")
    panel_c_fig.savefig(panel_c_png, dpi=400, bbox_inches="tight")
    panel_c_fig.savefig(panel_c_pdf, dpi=400, bbox_inches="tight")
    appendix_png = os.path.join(output_dir, "appendix_lambda_min_violin.png")
    appendix_pdf = os.path.join(output_dir, "appendix_lambda_min_violin.pdf")
    appendix_fig.savefig(appendix_png, dpi=400, bbox_inches="tight")
    appendix_fig.savefig(appendix_pdf, dpi=400, bbox_inches="tight")
    plt.close(appendix_fig)
    plt.close(panel_a_fig)
    plt.close(panel_b_fig)
    plt.close(panel_c_fig)
    plt.close(fig)
    return png, pdf


def interpretation_text(all_round_df):
    final_round = int(all_round_df["round"].max())
    f = all_round_df[all_round_df["round"] == final_round]
    summary = f.groupby("method")["lambda_min_pos"].mean()
    s_spade = float(summary.get("FedSPADE", np.nan))
    s_prox = float(summary.get("FedProx-EDL", np.nan))
    s_nova = float(summary.get("FedNova-EDL", np.nan))
    if np.isnan(s_spade) or np.isnan(s_prox) or np.isnan(s_nova):
        return "Insufficient final-round curvature statistics to draw a method-level conclusion."
    if s_spade > s_prox and s_spade > s_nova and s_spade > 1e-6:
        return (
            f"FedSPADE achieves the largest positive unseen-subspace curvature at the final round "
            f"(mean λ_min^+={s_spade:.4e}), while FedProx-EDL ({s_prox:.4e}) and FedNova-EDL ({s_nova:.4e}) "
            f"appear to stabilize optimization without fully restoring unseen evidential geometry."
        )
    return (
        f"FedSPADE does not show a clear final-round advantage in λ_min^+ "
        f"(FedSPADE {s_spade:.4e}, FedProx-EDL {s_prox:.4e}, FedNova-EDL {s_nova:.4e}); "
        f"additional rounds or stronger mechanism weights may be needed to confirm geometry repair."
    )


def run_study(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    methods = ["FedAvg-EDL", "FedProx-EDL", "FedNova-EDL", "FedSPADE"]
    all_round = []
    all_eigs = []
    for seed in cfg.seeds:
        for method_name in methods:
            round_df, eig_df = run_single_method(cfg, method_name, seed, cfg.output_dir)
            all_round.append(round_df)
            all_eigs.append(eig_df)
    all_round_df = pd.concat(all_round, ignore_index=True) if len(all_round) > 0 else pd.DataFrame()
    all_eig_df = pd.concat(all_eigs, ignore_index=True) if len(all_eigs) > 0 else pd.DataFrame()
    all_round_csv = os.path.join(cfg.output_dir, "curvature_rounds_all.csv")
    all_eigs_csv = os.path.join(cfg.output_dir, "curvature_eigs_all.csv")
    all_round_df.to_csv(all_round_csv, index=False)
    all_eig_df.to_csv(all_eigs_csv, index=False)
    fig_png, fig_pdf = make_figure(all_round_df, all_eig_df, cfg, cfg.output_dir)
    interp = interpretation_text(all_round_df)
    with open(os.path.join(cfg.output_dir, "interpretation.txt"), "w", encoding="utf-8") as f:
        f.write(interp + "\n")
    print(interp)
    print(f"Saved CSV logs to: {all_round_csv} and {all_eigs_csv}")
    print(f"Saved figures to: {fig_png} and {fig_pdf}")


def _parse_kv_pairs(items):
    out = {}
    for x in items:
        if "=" not in x:
            continue
        k, v = x.split("=", 1)
        k = k.strip()
        v = v.strip()
        if len(k) > 0 and len(v) > 0:
            out[k] = v
    return out


def _load_latest_variant_metrics(run_dir):
    rec_path = os.path.join(run_dir, "byzantine", "records.npz")
    metrics = {
        "Acc": np.nan,
        "ECE": np.nan,
        "Brier": np.nan,
        "AUROC": np.nan,
        "FPR95": np.nan,
        "Round": np.nan,
    }
    if os.path.exists(rec_path):
        d = np.load(rec_path, allow_pickle=False)
        if "rounds" in d and d["rounds"].size > 0:
            idx = -1
            metrics["Round"] = float(d["rounds"][idx])
            if "ece_mean" in d:
                metrics["ECE"] = float(d["ece_mean"][idx])
            if "auroc_epistemic" in d:
                metrics["AUROC"] = float(d["auroc_epistemic"][idx])
            if "acc_mean" in d:
                metrics["Acc"] = float(d["acc_mean"][idx])
            if "brier_mean" in d:
                metrics["Brier"] = float(d["brier_mean"][idx])
            if "fpr95" in d:
                metrics["FPR95"] = float(d["fpr95"][idx])
            return metrics
    log_path = run_dir if run_dir.endswith(".log") else os.path.join("results", "submission_summary", f"{os.path.basename(run_dir)}.log")
    if not os.path.exists(log_path):
        return metrics
    round_re = re.compile(r"Evaluating models at round\s+(\d+)")
    acc_re = re.compile(r"Personalized Accuracy \| Mean:\s*([0-9.]+)%")
    acc_soft_re = re.compile(r"Avg Test Acc Softmax:\s*([0-9.]+)%")
    acc_plain_re = re.compile(r"Avg Test Acc:\s*([0-9.]+)%")
    brier_re = re.compile(r"Personalized Brier Score \| Mean:\s*([0-9.]+)")
    ece_re = re.compile(r"Personalized ECE \| Mean:\s*([0-9.]+)")
    auroc_re = re.compile(r"OOD AUC-ROC \(Epistemic\):\s*([0-9.]+)")
    fpr_re = re.compile(r"OOD FPR@95%TPR:\s*([0-9.]+)")
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = round_re.search(line)
            if m:
                metrics["Round"] = float(m.group(1))
            m = acc_re.search(line)
            if m:
                metrics["Acc"] = float(m.group(1))
            m = acc_soft_re.search(line)
            if m:
                metrics["Acc"] = float(m.group(1))
            m = acc_plain_re.search(line)
            if m and not np.isfinite(metrics["Acc"]):
                metrics["Acc"] = float(m.group(1))
            m = brier_re.search(line)
            if m:
                metrics["Brier"] = float(m.group(1))
            m = ece_re.search(line)
            if m:
                metrics["ECE"] = float(m.group(1))
            m = auroc_re.search(line)
            if m:
                metrics["AUROC"] = float(m.group(1))
            m = fpr_re.search(line)
            if m:
                metrics["FPR95"] = float(m.group(1))
    return metrics


def _load_variant_mechanism_metrics(run_ref):
    metrics = {
        "LambdaMin+": np.nan,
        "CondNum": np.nan,
        "EigMean": np.nan,
    }
    log_path = run_ref if run_ref.endswith(".log") else os.path.join("results", "submission_summary", f"{os.path.basename(run_ref)}.log")
    base = os.path.splitext(log_path)[0]
    rounds_csv = base + "_curvature_rounds.csv"
    eigs_csv = base + "_curvature_eigs.csv"
    if not os.path.exists(rounds_csv):
        return metrics
    try:
        rdf = pd.read_csv(rounds_csv)
        if "lambda_min_pos" in rdf.columns and len(rdf) > 0:
            vals = pd.to_numeric(rdf["lambda_min_pos"], errors="coerce").dropna().values.astype(np.float64)
            if vals.size > 0:
                metrics["LambdaMin+"] = float(np.mean(vals))
        if "condition_number" in rdf.columns and len(rdf) > 0:
            vals = pd.to_numeric(rdf["condition_number"], errors="coerce").dropna().values.astype(np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                metrics["CondNum"] = float(np.mean(vals))
    except Exception:
        pass
    if os.path.exists(eigs_csv):
        try:
            edf = pd.read_csv(eigs_csv)
            if "eig_value" in edf.columns and len(edf) > 0:
                vals = pd.to_numeric(edf["eig_value"], errors="coerce").dropna().values.astype(np.float64)
                if vals.size > 0:
                    metrics["EigMean"] = float(np.mean(vals))
        except Exception:
            pass
    return metrics


def _fmt_metric(v, pct=False):
    if not np.isfinite(v):
        return "XX"
    if pct:
        return f"{v:.2f}"
    return f"{v:.4f}"


def export_ablation_table(output_dir, run_map):
    variants = [
        "GPR only",
        "OOD only",
        "AUG only",
        "GPR + OOD",
        "GPR + AUG",
        "OOD + AUG",
        "FedUg Full",
    ]
    component_flags = {
        "GPR only": {"GPR": "Y", "OOD": "N", "AUG": "N"},
        "OOD only": {"GPR": "N", "OOD": "Y", "AUG": "N"},
        "AUG only": {"GPR": "N", "OOD": "N", "AUG": "Y"},
        "GPR + OOD": {"GPR": "Y", "OOD": "Y", "AUG": "N"},
        "GPR + AUG": {"GPR": "Y", "OOD": "N", "AUG": "Y"},
        "OOD + AUG": {"GPR": "N", "OOD": "Y", "AUG": "Y"},
        "FedUg Full": {"GPR": "Y", "OOD": "Y", "AUG": "Y"},
    }
    rows = []
    for v in variants:
        run_dir = run_map.get(v, "")
        m = _load_latest_variant_metrics(run_dir) if run_dir else {
            "Acc": np.nan,
            "ECE": np.nan,
            "Brier": np.nan,
            "AUROC": np.nan,
            "FPR95": np.nan,
            "Round": np.nan,
        }
        mech = _load_variant_mechanism_metrics(run_dir) if run_dir else {
            "LambdaMin+": np.nan,
            "CondNum": np.nan,
            "EigMean": np.nan,
        }
        rows.append(
            {
                "Variant": v,
                "GPR": component_flags[v]["GPR"],
                "OOD": component_flags[v]["OOD"],
                "AUG": component_flags[v]["AUG"],
                "source_run": run_dir,
                "round": m["Round"],
                "Acc": m["Acc"],
                "ECE": m["ECE"],
                "Brier": m["Brier"],
                "AUROC": m["AUROC"],
                "FPR@95TPR": m["FPR95"],
                "MeanProjLambdaMin+": mech["LambdaMin+"],
                "ClientCondNum": mech["CondNum"],
                "MeanProjEigspec": mech["EigMean"],
            }
        )
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "ablation_main_table.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    valid_acc = [r["Acc"] for r in rows if np.isfinite(r["Acc"])]
    valid_ece = [r["ECE"] for r in rows if np.isfinite(r["ECE"])]
    valid_brier = [r["Brier"] for r in rows if np.isfinite(r["Brier"])]
    valid_auroc = [r["AUROC"] for r in rows if np.isfinite(r["AUROC"])]
    valid_fpr = [r["FPR@95TPR"] for r in rows if np.isfinite(r["FPR@95TPR"])]
    best_acc = max(valid_acc) if len(valid_acc) > 0 else np.nan
    best_ece = min(valid_ece) if len(valid_ece) > 0 else np.nan
    best_brier = min(valid_brier) if len(valid_brier) > 0 else np.nan
    best_auroc = max(valid_auroc) if len(valid_auroc) > 0 else np.nan
    best_fpr = min(valid_fpr) if len(valid_fpr) > 0 else np.nan
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Ablation of FedUg on CIFAR10 Dir(0.1), no attack.}",
        r"\label{tab:ablation_main}",
        r"\begin{tabular}{@{}lccc|ccccc|ccc@{}}",
        r"\toprule",
        r"Variant & GPR & OOD & AUG & Acc$\uparrow$ & ECE$\downarrow$ & Brier$\downarrow$ & AUROC$\uparrow$ & FPR@95TPR$\downarrow$ & Mean projected $\lambda_{min}^{+}$ & Client-wise condition number & Mean projected eigenspectrum \\",
        r"\midrule",
    ]
    for r in rows:
        acc = _fmt_metric(r["Acc"], pct=True)
        ece = _fmt_metric(r["ECE"])
        brier = _fmt_metric(r["Brier"])
        auroc = _fmt_metric(r["AUROC"])
        fpr = _fmt_metric(r["FPR@95TPR"])
        lam = _fmt_metric(r["MeanProjLambdaMin+"])
        cond = _fmt_metric(r["ClientCondNum"])
        eigm = _fmt_metric(r["MeanProjEigspec"])
        if np.isfinite(best_acc) and np.isfinite(r["Acc"]) and abs(r["Acc"] - best_acc) < 1e-12:
            acc = r"\textbf{" + acc + "}"
        if np.isfinite(best_ece) and np.isfinite(r["ECE"]) and abs(r["ECE"] - best_ece) < 1e-12:
            ece = r"\textbf{" + ece + "}"
        if np.isfinite(best_brier) and np.isfinite(r["Brier"]) and abs(r["Brier"] - best_brier) < 1e-12:
            brier = r"\textbf{" + brier + "}"
        if np.isfinite(best_auroc) and np.isfinite(r["AUROC"]) and abs(r["AUROC"] - best_auroc) < 1e-12:
            auroc = r"\textbf{" + auroc + "}"
        if np.isfinite(best_fpr) and np.isfinite(r["FPR@95TPR"]) and abs(r["FPR@95TPR"] - best_fpr) < 1e-12:
            fpr = r"\textbf{" + fpr + "}"
        lines.append(f'{r["Variant"]} & {r["GPR"]} & {r["OOD"]} & {r["AUG"]} & {acc} & {ece} & {brier} & {auroc} & {fpr} & {lam} & {cond} & {eigm} \\\\')
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = os.path.join(output_dir, "ablation_main_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved ablation summary CSV to: {csv_path}")
    print(f"Saved LaTeX table to: {tex_path}")


def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="results/mechanism_study_cifar10_dir01")
    p.add_argument("--partition_path", type=str, default="cifar10_c100_dir01")
    p.add_argument("--global_rounds", type=int, default=200)
    p.add_argument("--local_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--num_clients", type=int, default=100)
    p.add_argument("--sampling_prob", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--log_gap", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fedprox_mu", type=float, default=0.01)
    p.add_argument("--gc_weight", type=float, default=1.0)
    p.add_argument("--fsa_weight", type=float, default=0.1)
    p.add_argument("--afr_weight", type=float, default=0.1)
    p.add_argument("--afr_noise_std", type=float, default=0.03)
    p.add_argument("--max_hessian_batches", type=int, default=2)
    p.add_argument("--max_hessian_samples_per_batch", type=int, default=8)
    p.add_argument("--export_ablation_table", action="store_true")
    p.add_argument("--ablation_output_dir", type=str, default="results/submission_summary")
    p.add_argument(
        "--ablation_runs",
        type=str,
        nargs="+",
        default=[
            "GPR only=results/submission_summary/ablation7_gpr_only_200r.log",
            "OOD only=results/submission_summary/ablation7_ood_only_200r.log",
            "AUG only=results/submission_summary/ablation7_aug_only_200r.log",
            "GPR + OOD=results/submission_summary/ablation7_gpr_ood_200r.log",
            "GPR + AUG=results/submission_summary/ablation7_gpr_aug_200r.log",
            "OOD + AUG=results/submission_summary/ablation7_ood_aug_200r.log",
            "FedUg Full=results/submission_summary/ablation7_fedug_full_200r.log",
        ],
    )
    a = p.parse_args()
    cfg = StudyConfig(
        output_dir=a.output_dir,
        partition_path=a.partition_path,
        global_rounds=a.global_rounds,
        local_epochs=a.local_epochs,
        batch_size=a.batch_size,
        num_clients=a.num_clients,
        sampling_prob=a.sampling_prob,
        lr=a.lr,
        wd=a.wd,
        momentum=a.momentum,
        log_gap=a.log_gap,
        seeds=a.seeds,
        device=a.device,
        fedprox_mu=a.fedprox_mu,
        gc_weight=a.gc_weight,
        fsa_weight=a.fsa_weight,
        afr_weight=a.afr_weight,
        afr_noise_std=a.afr_noise_std,
        max_hessian_batches=a.max_hessian_batches,
        max_hessian_samples_per_batch=a.max_hessian_samples_per_batch,
    )
    return cfg, a


if __name__ == "__main__":
    cfg, args = parse_cli()
    if args.export_ablation_table:
        export_ablation_table(args.ablation_output_dir, _parse_kv_pairs(args.ablation_runs))
    else:
        run_study(cfg)
