import os
import sys
import argparse
import pickle
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, parent_dir)

from models.model_selector import get_model
from data.dataset_factory import DatasetFactory
from utils.utils import pulse_rate_from_power_spectral_density
from utils.gpu_utils import get_device
from losses.NegPearsonLoss import Neg_Pearson
from losses.SNRLoss import SNRLoss_dB_Signals
from utils.checkpoint_manager import CheckpointManager

try:
    from data.rf_processing import rotateIQ
except Exception:
    rotateIQ = None

try:
    from torch.nn.utils.stateless import functional_call
except Exception:
    def _resolve_attr(root: torch.nn.Module, qualname: str):
        parts = qualname.split(".")
        m = root
        for p in parts[:-1]:
            m = getattr(m, p)
        return m, parts[-1]

    def functional_call(module: torch.nn.Module, params_and_buffers: dict, args, kwargs=None):
        if kwargs is None:
            kwargs = {}
        orig_params = {}
        orig_buffers = {}
        for name, tensor in params_and_buffers.items():
            submod, attr = _resolve_attr(module, name)
            original = getattr(submod, attr)
            if isinstance(original, torch.nn.Parameter):
                orig_params[name] = (submod, attr, original.data.clone())
                original.data.copy_(tensor)
            elif isinstance(original, torch.Tensor) and not isinstance(original, torch.nn.Parameter):
                orig_buffers[name] = (submod, attr, original.data.clone())
                original.data.copy_(tensor)
            else:
                orig_params[name] = (submod, attr, original)
                try:
                    setattr(submod, attr, tensor)
                except (TypeError, AttributeError):
                    print(f"[WARN] Could not set {name}, skipping")
        try:
            return module(*args, **kwargs)
        finally:
            for name, (submod, attr, orig_val) in orig_params.items():
                current = getattr(submod, attr)
                if isinstance(current, torch.nn.Parameter):
                    current.data.copy_(orig_val)
                elif hasattr(current, 'data') and isinstance(orig_val, torch.Tensor):
                    current.data.copy_(orig_val)
                else:
                    setattr(submod, attr, orig_val)
            for name, (submod, attr, orig_val) in orig_buffers.items():
                current = getattr(submod, attr)
                if hasattr(current, 'data') and isinstance(orig_val, torch.Tensor):
                    current.data.copy_(orig_val)
                else:
                    setattr(submod, attr, orig_val)

DATASET_CONFIG = {
    "equipleth": {
        "data_dir": "./data/EquiPleth",
        "folds_path": "./data/EquiPleth/demo_fold2.pkl",
        "description": "EquiPleth dataset with RGB and RF data",
    },
}


def load_checkpoint_weights(model, ckpt_path: str, tag: str) -> bool:
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[WARN] {tag} checkpoint not found: {ckpt_path}")
        return False
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        elif isinstance(ckpt, dict) and all(hasattr(v, "shape") for v in ckpt.values()):
            state_dict = ckpt
        else:
            print(f"[WARN] Cannot find model-state keys in {tag} checkpoint.")
            return False
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                cleaned[k[7:]] = v
            else:
                cleaned[k] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        allowed_missing = {
            "rppg_head.bias1",
            "rppg_head.final_layer_hr.0.conv_block_3d.0.weight",
            "rppg_head.final_layer_hr.1.conv_block_3d.0.weight",
            "rppg_head.final_layer_hr.2.weight",
        }
        missing_set = set(missing)
        unexpected_set = set(unexpected)
        unexpected_allowed = missing_set - allowed_missing
        print(f"[INFO] {tag} checkpoint loaded: {ckpt_path}")
        if unexpected_set:
            print("[WARN] Unexpected keys not empty, load not fully clean.")
            return False
        if unexpected_allowed:
            print("[WARN] Missing keys outside allowed set:", sorted(unexpected_allowed))
            return False
        return True
    except Exception as e:
        print(f"[WARN] Failed to load {tag} checkpoint: {e}")
        return False


def extract_pred(output) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, torch.Tensor) and output.dim() == 3 and output.size(1) == 1:
        output = output.squeeze(1)
    return output


def build_student(args, device):
    args.channels = args.rgb_channels
    student = get_model(args.student_model, "rgb", args, device)
    return student.to(device)


def build_teacher(args, device):
    args.channels = args.rf_channels
    teacher = get_model(args.teacher_model, "rf", args, device)
    return teacher.to(device)


def forward_student(student, rgb_batch):
    return extract_pred(student(rgb_batch))


def forward_teacher(teacher, rf_batch, args):
    if args.target_domain == "equipleth" and rotateIQ is not None and rf_batch.dim() >= 3:
        try:
            rf_batch = rotateIQ(rf_batch)
        except Exception:
            pass
    return extract_pred(teacher(rf_batch))


def _align_1d(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    t = min(a.size(1), b.size(1))
    return a[:, :t], b[:, :t]


def compute_psd_features(signal: torch.Tensor, args, eps: float = 1e-8) -> Optional[Dict[str, torch.Tensor]]:
    if signal is None:
        return None
    if signal.dim() != 2:
        signal = signal.view(signal.size(0), -1)
    t = signal.size(1)
    if t < 8:
        return None
    signal = signal - signal.mean(dim=-1, keepdim=True)
    window = torch.hann_window(t, device=signal.device, dtype=signal.dtype).unsqueeze(0)
    signal = signal * window
    fft = torch.fft.rfft(signal, dim=-1)
    psd = (fft.real ** 2 + fft.imag ** 2)
    freqs = torch.fft.rfftfreq(t, d=1.0 / args.fs).to(signal.device)
    l_freq = args.l_freq_bpm / 60.0
    u_freq = args.u_freq_bpm / 60.0
    mask = (freqs >= l_freq) & (freqs <= u_freq)
    if mask.sum() == 0:
        return None
    psd_band = psd[:, mask]
    log_psd_band_raw = torch.log(psd_band + eps)
    log_psd_mean = log_psd_band_raw.mean(dim=1, keepdim=True)
    log_psd_std = log_psd_band_raw.std(dim=1, keepdim=True, unbiased=False)
    log_psd_band = (log_psd_band_raw - log_psd_mean) / (log_psd_std + eps)
    freqs_bpm = freqs[mask] * 60.0
    weights = torch.softmax(log_psd_band_raw, dim=1)
    peak_freq = (weights * freqs_bpm.unsqueeze(0)).sum(dim=1)
    band_len = psd_band.size(1)
    idx = torch.arange(band_len, device=signal.device, dtype=signal.dtype)
    peak_idx = (weights * idx.unsqueeze(0)).sum(dim=1)
    window_bins = max(1, int(0.02 * band_len))
    sigma = max(1.0, window_bins / 2.0)
    dist = idx.unsqueeze(0) - peak_idx.unsqueeze(1)
    peak_window = torch.exp(-0.5 * (dist / sigma) ** 2)
    band_energy = psd_band.sum(dim=1) + eps
    peak_energy = (psd_band * peak_window).sum(dim=1)
    band_energy_ratio = peak_energy / band_energy
    return {
        "log_psd": log_psd_band,
        "log_psd_raw": log_psd_band_raw,
        "psd_band": psd_band,
        "band_energy": band_energy,
        "peak_freq": peak_freq,
        "peak_idx": peak_idx,
        "peak_window": peak_window,
        "band_energy_ratio": band_energy_ratio,
    }


def build_policy_psd_input(
        feats_s: Dict[str, torch.Tensor],
        feats_t: Dict[str, torch.Tensor],
) -> torch.Tensor:
    log_s = feats_s["log_psd"]
    log_t = feats_t["log_psd"]
    f_min = min(log_s.size(1), log_t.size(1))
    if f_min <= 0:
        raise ValueError("Invalid PSD length for policy input.")
    if log_s.size(1) != f_min:
        log_s = log_s[:, :f_min]
    if log_t.size(1) != f_min:
        log_t = log_t[:, :f_min]
    return torch.stack([log_s, log_t, torch.abs(log_s - log_t)], dim=1)


def compute_pcd_components(
        feats_s: Dict[str, torch.Tensor],
        feats_t: Dict[str, torch.Tensor],
        args,
        eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Earliest-auto: 3 components band_peak / band_other / struct. L_struct = L_peak_idx + L_sharp."""
    del args
    diff = torch.abs(feats_s["log_psd"] - feats_t["log_psd"])
    peak_w = feats_t["peak_window"]
    other_w = torch.clamp(1.0 - peak_w, min=0.0)
    peak_norm = peak_w.sum(dim=1) + eps
    other_norm = other_w.sum(dim=1) + eps
    l_band_peak = (diff * peak_w).sum(dim=1) / peak_norm
    l_band_other = (diff * other_w).sum(dim=1) / other_norm
    l_peak_idx = F.smooth_l1_loss(feats_s["peak_idx"], feats_t["peak_idx"], reduction="none")
    p_s = torch.softmax(feats_s["log_psd"], dim=1)
    p_t = torch.softmax(feats_t["log_psd"], dim=1)
    ent_s = -(p_s * torch.log(p_s + eps)).sum(dim=1)
    ent_t = -(p_t * torch.log(p_t + eps)).sum(dim=1)
    l_sharp = F.smooth_l1_loss(ent_s, ent_t, reduction="none")
    l_struct = l_peak_idx + l_sharp
    return {
        "L_band_peak": l_band_peak,
        "L_band_other": l_band_other,
        "L_peak_idx": l_peak_idx,
        "L_sharp": l_sharp,
        "L_struct": l_struct,
    }


def aggregate_distill_loss(
        components: Dict[str, torch.Tensor],
        g: torch.Tensor,
        alpha: torch.Tensor,
        args,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Earliest-auto: comp order = [band_peak, band_other, struct], K=3."""
    if int(args.policy_k) != 3:
        raise ValueError("--policy-k must be 3 for RPM-Distill (peak/off/shape).")
    comp = torch.stack(
        [
            components["L_band_peak"],
            components["L_band_other"],
            components["L_struct"],
        ],
        dim=1,
    )
    distill_i = g * (alpha * comp).sum(dim=1)
    l_distill = distill_i.mean()
    stats: Dict[str, torch.Tensor] = {
        "g_mean": g.mean().detach(),
        "distill_mean": l_distill.detach(),
        "alpha_entropy": (-(alpha * torch.log(alpha + 1e-8)).sum(dim=1).mean()).detach(),
    }
    alpha_mean = alpha.mean(dim=0).detach()
    for idx in range(alpha_mean.numel()):
        stats[f"alpha_mean_{idx}"] = alpha_mean[idx]
    return l_distill, stats


def _make_group_norm(num_channels: int) -> nn.GroupNorm:
    groups = min(8, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class PolicyResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.gn1 = _make_group_norm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, dilation=1, bias=False)
        self.gn2 = _make_group_norm(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.gn2(x)
        return self.act(x + res)


class MatrixDecompositionDecoder(nn.Module):
    """Matrix-decomposition style decoder for policy heads."""

    def __init__(self, in_ch: int, rank: int, hidden: int, out_k: int, dropout: float):
        super().__init__()
        self.rank = max(2, int(rank))
        self.pre = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=1, bias=False),
            _make_group_norm(in_ch),
            nn.SiLU(inplace=True),
        )
        self.basis_proj = nn.Conv1d(in_ch, self.rank, kernel_size=1, bias=True)
        self.coeff_proj = nn.Conv1d(in_ch, self.rank, kernel_size=1, bias=True)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fuse = nn.Sequential(
            nn.Linear(in_ch * 2, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head_alpha = nn.Linear(hidden, out_k)
        self.head_g = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.pre(x)
        x_t = x.transpose(1, 2)  # [B, F, C]
        basis_attn = torch.softmax(self.basis_proj(x), dim=-1)  # [B, R, F]
        coeff_attn = torch.softmax(self.coeff_proj(x), dim=1)   # [B, R, F]
        basis_tokens = torch.bmm(basis_attn, x_t)  # [B, R, C]
        coeff_tokens = torch.bmm(coeff_attn, x_t)  # [B, R, C]
        md_vec = (0.5 * (basis_tokens + coeff_tokens)).mean(dim=1)  # [B, C]
        global_vec = self.global_pool(x).squeeze(-1)  # [B, C]
        h = self.fuse(torch.cat([md_vec, global_vec], dim=1))
        alpha = torch.softmax(self.head_alpha(h), dim=-1)
        g = torch.sigmoid(self.head_g(h)).squeeze(-1)
        return g, alpha


class PolicyConvNet(nn.Module):
    """Convolutional encoder + matrix-decomposition decoder policy."""

    def __init__(
            self,
            in_ch: int = 3,
            base_ch: int = 32,
            depth: int = 3,
            k: int = 3,
            dropout: float = 0.1,
            md_rank: int = 8,
            md_hidden: int = 32,
    ):
        super().__init__()
        self.k = k
        self.depth = max(1, int(depth))
        self.dilations = [1, 2, 4]
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base_ch, kernel_size=5, padding=2, bias=False),
            _make_group_norm(base_ch),
            nn.SiLU(inplace=True),
        )
        self.transition = None
        self.blocks = nn.ModuleList()
        ch = base_ch
        for idx in range(self.depth):
            if idx == 1:
                out_ch = base_ch * 2
                self.transition = nn.Sequential(
                    nn.Conv1d(ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                    _make_group_norm(out_ch),
                    nn.SiLU(inplace=True),
                )
                ch = out_ch
            dilation = self.dilations[min(idx, len(self.dilations) - 1)]
            self.blocks.append(PolicyResidualBlock(channels=ch, dilation=dilation))
        self.decoder = MatrixDecompositionDecoder(
            in_ch=ch,
            rank=md_rank,
            hidden=md_hidden,
            out_k=k,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        for idx, block in enumerate(self.blocks):
            if idx == 1 and self.transition is not None:
                x = self.transition(x)
            x = block(x)
        return self.decoder(x)


def _infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def _sup_loss(pred: torch.Tensor, target: torch.Tensor, criterion, args) -> torch.Tensor:
    pred, target = _align_1d(pred, target)
    return 0.01 * criterion[0](pred, target) + criterion[1](pred, target, Fs=args.fs)


def train_model(
        student: nn.Module,
        teacher: nn.Module,
        policy: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device,
        args,
) -> Tuple[nn.Module, Dict[str, Any]]:
    criterion = [Neg_Pearson(), SNRLoss_dB_Signals()]
    opt_theta = torch.optim.Adam(student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    opt_phi = torch.optim.Adam(policy.parameters(), lr=args.meta_lr, weight_decay=0.0)
    checkpoint_manager = CheckpointManager(base_dir=args.checkpoints_path)
    start_epoch = 1
    best_val_loss = float("inf")
    global_step = 0
    try:
        ckpt = checkpoint_manager.load_checkpoint(
            student, opt_theta, args.source_domain, "RPM-Distill", args.student_model,
            args.frame_length, args.fold, args.step, load_best=True
        )
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        global_step = ckpt.get("global_step", 0)
        policy_dict = ckpt.get("policy_state_dict") or ckpt.get("additional_info", {}).get("policy_state_dict")
        opt_phi_dict = ckpt.get("opt_phi_state_dict") or ckpt.get("additional_info", {}).get("opt_phi_state_dict")
        if policy_dict is not None:
            try:
                policy.load_state_dict(policy_dict)
                print("[INFO] Loaded policy state dict")
            except Exception as e:
                print(f"[WARN] Policy state-dict mismatch. Re-initializing policy: {e}")
        if opt_phi_dict is not None:
            try:
                opt_phi.load_state_dict(opt_phi_dict)
                print("[INFO] Loaded opt_phi state dict")
            except Exception as e:
                print(f"[WARN] Failed to load opt_phi state dict. Re-initializing optimizer: {e}")
        print(f"[INFO] Resumed checkpoint: epoch={start_epoch-1}, global_step={global_step}")
    except FileNotFoundError:
        print("[INFO] No existing checkpoint found, starting from scratch")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    val_iter = _infinite_loader(val_loader)
    last_epoch_summary: Dict[str, Any] = {}
    for epoch in range(start_epoch, args.epochs + 1):
        student.train()
        policy.train()
        sum_sup = 0.0
        sum_distill = 0.0
        sum_tr = 0.0
        sum_g = 0.0
        sum_alpha_entropy = 0.0
        sum_alpha = torch.zeros(args.policy_k, device=device)
        num_steps = 0
        meta_updates = 0
        for batch_idx, (rgb_data, rf_data, target_data) in enumerate(
                tqdm(train_loader, desc=f"Epoch Train {epoch}/{args.epochs}")
        ):
            if rgb_data.size(0) == 0 or rf_data.size(0) == 0:
                continue
            rgb_data = rgb_data.to(device).float()
            rf_data = rf_data.to(device).float()
            target_data = target_data.to(device).float()
            if (global_step % args.meta_interval) == 0:
                y_s = forward_student(student, rgb_data)
                with torch.no_grad():
                    y_t = forward_teacher(teacher, rf_data, args)
                y_s_d, y_t_d = _align_1d(y_s, y_t)
                feats_s = compute_psd_features(y_s_d, args)
                feats_t = compute_psd_features(y_t_d, args)
                if feats_s is not None and feats_t is not None:
                    x_pol_psd = build_policy_psd_input(feats_s, feats_t)
                    g, alpha = policy(x_pol_psd)
                    comps = compute_pcd_components(feats_s, feats_t, args)
                    l_distill, _ = aggregate_distill_loss(comps, g, alpha, args)
                else:
                    batch_size = rgb_data.size(0)
                    if y_s_d.numel() > 0:
                        l_distill = (y_s_d * 0.0).sum() * 1e-10
                    else:
                        l_distill = torch.tensor(0.0, device=device, requires_grad=True)
                    g = torch.zeros(batch_size, device=device)
                    alpha = torch.zeros(batch_size, args.policy_k, device=device)
                    alpha[:, 0] = 1.0
                if args.student_model in ["FactorizePhys", "iBVPNet"]:
                    target_sup = target_data[:, :-1]
                else:
                    target_sup = target_data
                l_sup = _sup_loss(y_s, target_sup, criterion, args)
                if args.train_mode == "sup_only":
                    l_tr = l_sup
                elif args.train_mode == "distill_only":
                    l_tr = args.lambda_distill * l_distill
                else:
                    l_tr = l_sup + args.lambda_distill * l_distill
                if args.gate_reg > 0:
                    l_tr = l_tr + args.gate_reg * (1.0 - torch.clamp(g, 0.0, 1.0)).mean()
                params = dict(student.named_parameters())
                grads = torch.autograd.grad(l_tr, params.values(), create_graph=True, allow_unused=True)
                theta_prime = {}
                for (n, p), g_ in zip(params.items(), grads):
                    if g_ is not None:
                        theta_prime[n] = p - args.virtual_lr * g_
                    else:
                        theta_prime[n] = p
                buffers = {n: b for n, b in student.named_buffers()}
                params_and_buffers = {**theta_prime, **buffers}
                rgb_val, _, target_val = next(val_iter)
                rgb_val = rgb_val.to(device).float()
                target_val = target_val.to(device).float()
                was_training = student.training
                student.eval()
                y_val_pred = extract_pred(functional_call(student, params_and_buffers, (rgb_val,)))
                if was_training:
                    student.train()
                if args.student_model in ["FactorizePhys", "iBVPNet"]:
                    target_val = target_val[:, :-1]
                l_meta = _sup_loss(y_val_pred, target_val, criterion, args)
                opt_phi.zero_grad(set_to_none=True)
                l_meta.backward()
                opt_phi.step()
                meta_updates += 1
                opt_theta.zero_grad(set_to_none=True)
            y_s = forward_student(student, rgb_data)
            with torch.no_grad():
                y_t = forward_teacher(teacher, rf_data, args)
            y_s_d, y_t_d = _align_1d(y_s, y_t)
            feats_s = compute_psd_features(y_s_d, args)
            feats_t = compute_psd_features(y_t_d, args)
            if args.student_model in ["FactorizePhys", "iBVPNet"]:
                target_sup = target_data[:, :-1]
            else:
                target_sup = target_data
            l_sup = _sup_loss(y_s, target_sup, criterion, args)
            if feats_s is not None and feats_t is not None:
                with torch.no_grad():
                    x_pol_psd = build_policy_psd_input(feats_s, feats_t)
                    if args.policy_detach_input:
                        x_pol_psd = x_pol_psd.detach()
                    g, alpha = policy(x_pol_psd)
                comps = compute_pcd_components(feats_s, feats_t, args)
                l_distill, dist_stats = aggregate_distill_loss(comps, g, alpha, args)
            else:
                g = torch.zeros(rgb_data.size(0), device=device)
                alpha = torch.zeros(rgb_data.size(0), args.policy_k, device=device)
                alpha[:, 0] = 1.0
                l_distill = torch.tensor(0.0, device=device)
                dist_stats = {
                    "g_mean": torch.tensor(0.0, device=device),
                    "distill_mean": l_distill.detach(),
                    "alpha_entropy": torch.tensor(0.0, device=device),
                }
                for k_idx in range(args.policy_k):
                    dist_stats[f"alpha_mean_{k_idx}"] = torch.tensor(0.0, device=device)
            if args.train_mode == "sup_only":
                l_tr = l_sup
            elif args.train_mode == "distill_only":
                l_tr = args.lambda_distill * l_distill
            else:
                l_tr = l_sup + args.lambda_distill * l_distill
            opt_theta.zero_grad(set_to_none=True)
            l_tr.backward()
            opt_theta.step()
            sum_sup += float(l_sup.detach().cpu())
            sum_distill += float(l_distill.detach().cpu())
            sum_tr += float(l_tr.detach().cpu())
            sum_g += float(dist_stats["g_mean"].detach().cpu())
            sum_alpha_entropy += float(dist_stats["alpha_entropy"].detach().cpu())
            sum_alpha += torch.stack([dist_stats[f"alpha_mean_{k_idx}"] for k_idx in range(args.policy_k)]).detach()
            num_steps += 1
            global_step += 1
        if num_steps == 0:
            print("[WARN] No valid training steps in this epoch.")
            continue
        avg_alpha = (sum_alpha / num_steps).detach().cpu().numpy().tolist()
        avg_train = {
            "sup_loss": sum_sup / num_steps,
            "distill_loss": sum_distill / num_steps,
            "tr_loss": sum_tr / num_steps,
            "g_mean": sum_g / num_steps,
            "policy_entropy_alpha": sum_alpha_entropy / num_steps,
            "alpha_means": avg_alpha,
            "meta_updates": meta_updates,
        }
        student.eval()
        policy.eval()
        val_sup_sum = 0.0
        val_tr_sum = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch_idx, (rgb_data, rf_data, target_data) in enumerate(
                    tqdm(val_loader, desc=f"Epoch Val {epoch}/{args.epochs}")
            ):
                if rgb_data.size(0) == 0 or rf_data.size(0) == 0:
                    continue
                rgb_data = rgb_data.to(device).float()
                rf_data = rf_data.to(device).float()
                target_data = target_data.to(device).float()
                y_s = forward_student(student, rgb_data)
                y_t = forward_teacher(teacher, rf_data, args)
                if args.student_model in ["FactorizePhys", "iBVPNet"]:
                    target_sup = target_data[:, :-1]
                else:
                    target_sup = target_data
                l_sup = _sup_loss(y_s, target_sup, criterion, args)
                y_s_d, y_t_d = _align_1d(y_s, y_t)
                feats_s = compute_psd_features(y_s_d, args)
                feats_t = compute_psd_features(y_t_d, args)
                if feats_s is not None and feats_t is not None:
                    x_pol_psd = build_policy_psd_input(feats_s, feats_t)
                    g, alpha = policy(x_pol_psd)
                    comps = compute_pcd_components(feats_s, feats_t, args)
                    l_distill, _ = aggregate_distill_loss(comps, g, alpha, args)
                else:
                    l_distill = torch.tensor(0.0, device=device)
                if args.train_mode == "sup_only":
                    l_tr = l_sup
                elif args.train_mode == "distill_only":
                    l_tr = args.lambda_distill * l_distill
                else:
                    l_tr = l_sup + args.lambda_distill * l_distill
                val_sup_sum += float(l_sup.detach().cpu())
                val_tr_sum += float(l_tr.detach().cpu())
                val_steps += 1
        avg_val_sup = val_sup_sum / max(1, val_steps)
        avg_val_tr = val_tr_sum / max(1, val_steps)
        print(f"\n[Epoch {epoch}/{args.epochs}] meta_updates={meta_updates}")
        print(f"  Train: L_sup={avg_train['sup_loss']:.6f}  L_distill={avg_train['distill_loss']:.6f}  L_tr={avg_train['tr_loss']:.6f}")
        print(f"  Train: g_mean={avg_train['g_mean']:.4f}  alpha_entropy={avg_train['policy_entropy_alpha']:.4f}")
        print(f"  Train: alpha_mean={['{:.3f}'.format(x) for x in avg_train['alpha_means']]}")
        print(f"  Val:   L_sup={avg_val_sup:.6f}  L_tr={avg_val_tr:.6f}")
        is_best = avg_val_sup < best_val_loss
        if is_best:
            best_val_loss = avg_val_sup
        additional_info = {
            "best_val_loss": best_val_loss,
            "global_step": global_step,
            "policy_state_dict": policy.state_dict(),
            "opt_phi_state_dict": opt_phi.state_dict(),
            "meta_cfg": {
                "train_mode": args.train_mode,
                "lambda_distill": args.lambda_distill,
                "meta_lr": args.meta_lr,
                "meta_interval": args.meta_interval,
                "virtual_lr": args.virtual_lr,
                "policy_k": args.policy_k,
                "policy_type": args.policy_type,
                "policy_base_ch": args.policy_base_ch,
                "policy_depth": args.policy_depth,
                "policy_dropout": args.policy_dropout,
                "policy_md_rank": args.policy_md_rank,
                "policy_md_hidden": args.policy_md_hidden,
                "policy_detach_input": args.policy_detach_input,
                "gate_reg": args.gate_reg,
            },
            "epoch_train_summary": avg_train,
            "epoch_val_sup": avg_val_sup,
            "epoch_val_tr": avg_val_tr,
        }
        checkpoint_manager.save_checkpoint(
            student, opt_theta, epoch, avg_val_sup,
            args.source_domain, "RPM-Distill", args.student_model,
            args.frame_length, args.fold, args.step,
            is_best=is_best, additional_info=additional_info
        )
        last_epoch_summary = {
            **avg_train,
            "avg_val_sup": avg_val_sup,
            "avg_val_tr": avg_val_tr,
            "best_val_loss": best_val_loss,
        }
    return student, last_epoch_summary


def evaluate_student(student, data_loader, device, args):
    student.eval()
    all_est_ppgs = []
    all_gt_ppgs = []
    all_hr_test = []
    peak_diff = []
    ratio_diff = []
    with torch.no_grad():
        for batch_idx, (rgb_data, rf_data, target_data) in enumerate(tqdm(data_loader, desc="Evaluating")):
            if rgb_data.size(0) == 0:
                continue
            rgb_data = rgb_data.to(device).float()
            target_data = target_data.to(device).float()
            if args.student_model in ["FactorizePhys", "iBVPNet"]:
                target_data = target_data[:, :-1]
            student_pred = forward_student(student, rgb_data)
            student_pred, target_data = _align_1d(student_pred, target_data)
            all_est_ppgs.extend(student_pred.cpu().numpy())
            all_gt_ppgs.extend(target_data.cpu().numpy())
            feats_s = compute_psd_features(student_pred, args)
            feats_g = compute_psd_features(target_data, args)
            if feats_s is not None and feats_g is not None:
                peak_diff.append(torch.abs(feats_s["peak_freq"] - feats_g["peak_freq"]).mean().item())
                ratio_diff.append(torch.abs(feats_s["band_energy_ratio"] - feats_g["band_energy_ratio"]).mean().item())
    for i in range(len(all_est_ppgs)):
        est_ppg = all_est_ppgs[i]
        gt_ppg = all_gt_ppgs[i]
        try:
            hr_est = pulse_rate_from_power_spectral_density(est_ppg, FS=args.fs, LL_PR=args.l_freq_bpm, UL_PR=args.u_freq_bpm)
            hr_gt = pulse_rate_from_power_spectral_density(gt_ppg, FS=args.fs, LL_PR=args.l_freq_bpm, UL_PR=args.u_freq_bpm)
            all_hr_test.append([hr_est, hr_gt])
        except Exception:
            continue
    if len(all_hr_test) == 0:
        return {
            "hr_mae_mean": 0.0, "hr_mae_std": 0.0, "hr_rmse_mean": 0.0, "hr_rmse_std": 0.0,
            "hr_corr_mean": 0.0, "hr_error_std": 0.0, "snr_mean": 0.0, "snr_std": 0.0, "num_samples": 0,
            "peak_freq_mae": 0.0, "band_ratio_mae": 0.0,
        }
    hr_est = np.array([item[0] for item in all_hr_test])
    hr_gt = np.array([item[1] for item in all_hr_test])
    hr_mae = np.mean(np.abs(hr_est - hr_gt))
    hr_rmse = np.sqrt(np.mean((hr_est - hr_gt) ** 2))
    hr_corr = np.corrcoef(hr_est, hr_gt)[0, 1] if len(hr_est) > 1 else 0.0
    snr_values = []
    for i in range(len(all_est_ppgs)):
        try:
            signal_power = np.mean(all_est_ppgs[i] ** 2)
            noise_power = np.mean((all_est_ppgs[i] - all_gt_ppgs[i]) ** 2)
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                snr_values.append(snr)
        except Exception:
            continue
    snr_mean = np.mean(snr_values) if snr_values else 0.0
    snr_std = np.std(snr_values) if len(snr_values) > 1 else 0.0
    hr_mae_std = np.std(np.abs(hr_est - hr_gt)) if len(hr_est) > 1 else 0.0
    hr_rmse_std = np.std((hr_est - hr_gt) ** 2) ** 0.5 if len(hr_est) > 1 else 0.0
    # Standard deviation of the error (Std): std(hr_est - hr_gt)
    hr_error_std = np.std(hr_est - hr_gt) if len(hr_est) > 1 else 0.0
    return {
        "hr_mae_mean": hr_mae, "hr_mae_std": hr_mae_std,
        "hr_rmse_mean": hr_rmse, "hr_rmse_std": hr_rmse_std,
        "hr_corr_mean": hr_corr, "hr_error_std": hr_error_std,
        "snr_mean": snr_mean, "snr_std": snr_std,
        "num_samples": len(all_hr_test),
        "peak_freq_mae": float(np.mean(peak_diff)) if peak_diff else 0.0,
        "band_ratio_mae": float(np.mean(ratio_diff)) if ratio_diff else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Train RPM-Distill")
    parser.add_argument("-s", "--source-domain", type=str, default="equipleth",
                        choices=["equipleth"])
    parser.add_argument("-t", "--target-domain", type=str, default="equipleth",
                        choices=["equipleth"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--folds-path", type=str, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--frame-length", type=int, default=256)
    parser.add_argument("--step", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--checkpoints_path", type=str, default=None)
    parser.add_argument('--rgb-checkpoint', type=str, default=None)
    parser.add_argument("--rf-checkpoint", type=str, default=None)
    parser.add_argument("--student-model", type=str, default="FactorizePhys")
    parser.add_argument("--teacher-model", type=str, default="RF_conv_decoder")
    parser.add_argument("--rgb-channels", type=int, default=3)
    parser.add_argument("--rf-channels", type=int, default=10)
    parser.add_argument("--video-length", type=int, default=300)
    parser.add_argument("--sampling-ratio", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--fs", type=int, default=30)
    parser.add_argument("--l-freq-bpm", type=int, default=45)
    parser.add_argument("--u-freq-bpm", type=int, default=180)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train-mode", type=str, default="sup_plus_distill",
                        choices=["sup_only", "distill_only", "sup_plus_distill"])
    parser.add_argument("--policy-type", type=str, default="conv", choices=["conv"])
    parser.add_argument("--policy-base-ch", type=int, default=32)
    parser.add_argument("--policy-depth", type=int, default=3)
    parser.add_argument("--policy-dropout", type=float, default=0.1)
    parser.add_argument("--policy-md-rank", type=int, default=8,
                        help="Rank in matrix-decomposition decoder.")
    parser.add_argument("--policy-md-hidden", type=int, default=32,
                        help="Hidden dim in matrix-decomposition decoder fusion head.")
    parser.add_argument("--policy-detach-input", dest="policy_detach_input", action="store_true")
    parser.add_argument("--no-policy-detach-input", dest="policy_detach_input", action="store_false")
    parser.set_defaults(policy_detach_input=True)
    parser.add_argument("--policy-k", type=int, default=3)
    parser.add_argument("--lambda-distill", type=float, default=1.0)
    parser.add_argument("--meta-lr", type=float, default=1e-4)
    parser.add_argument("--meta-interval", type=int, default=10)
    parser.add_argument("--virtual-lr", type=float, default=None)
    parser.add_argument("--gate-reg", type=float, default=0.0)
    parser.add_argument("--early-stopping", type=bool, default=True)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.001)
    args = parser.parse_args()
    if args.virtual_lr is None:
        args.virtual_lr = float(args.learning_rate)
    if args.policy_type != "conv":
        raise ValueError("--policy-type currently supports only 'conv'.")
    if int(args.policy_k) != 3:
        raise ValueError("--policy-k must be 3 for RPM-Distill (peak/off/shape).")
    if args.target_domain not in DATASET_CONFIG or args.source_domain not in DATASET_CONFIG:
        raise ValueError("Unsupported domain.")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    source_config = DATASET_CONFIG[args.source_domain]
    target_config = DATASET_CONFIG[args.target_domain]
    if args.data_dir is None:
        args.data_dir = source_config["data_dir"]
    if args.folds_path is None:
        args.folds_path = source_config["folds_path"]
    ckpt_mgr = CheckpointManager(base_dir=args.checkpoints_path)
    if args.rgb_checkpoint is None:
        args.rgb_checkpoint = ckpt_mgr.get_checkpoint_path(
            args.source_domain, "rgb", args.student_model, args.frame_length, args.fold, args.step, load_best=True
        )
    if args.rf_checkpoint is None:
        args.rf_checkpoint = ckpt_mgr.get_checkpoint_path(
            args.source_domain, "rf", args.teacher_model, args.frame_length, args.fold, args.step, load_best=True
        )
    print(f"Source domain: {args.source_domain}  Target: {args.target_domain}")
    print(f"student: {args.student_model}  teacher: {args.teacher_model}")
    print(
        f"RPM-Distill: policy_k=3 (peak/off/shape) policy_depth={args.policy_depth} "
        f"policy_base_ch={args.policy_base_ch} md_rank={args.policy_md_rank} md_hidden={args.policy_md_hidden}"
    )
    device = get_device(gpu_index=args.device, min_free_memory_gb=20)
    if args.source_domain == args.target_domain:
        with open(args.folds_path, "rb") as f:
            folds_data = pickle.load(f)
        files_in_fold = folds_data[args.fold]
        all_files = files_in_fold["train"] + files_in_fold["val"] + files_in_fold["test"]
        rng = np.random.default_rng(args.seed)
        all_files = [all_files[i] for i in rng.permutation(len(all_files))]
        total = len(all_files)
        train_end = int(total * 0.7)
        val_end = train_end + int(total * 0.1)
        source_train_files = all_files[:train_end]
        source_val_files = all_files[train_end:val_end]
        target_test_files = all_files[val_end:]
    else:
        with open(source_config["folds_path"], "rb") as f:
            source_folds_data = pickle.load(f)
        s_fold = source_folds_data[args.fold]
        source_train_files = s_fold["train"]
        source_val_files = s_fold["val"]
        with open(target_config["folds_path"], "rb") as f:
            target_folds_data = pickle.load(f)
        t_fold = target_folds_data[args.fold]
        target_test_files = t_fold["train"] + t_fold["val"] + t_fold["test"]
    dataset_factory = DatasetFactory()
    collate_fn = None
    train_dataset = dataset_factory.create_paired_distill_dataset(
        target_domain=args.source_domain, datapath=source_config["data_dir"], datapaths=source_train_files,
        video_length=args.video_length, frame_length=args.frame_length, step=args.step,
        sampling_ratio=args.sampling_ratio, window_size=args.window_size,
    )
    val_dataset = dataset_factory.create_paired_distill_dataset(
        target_domain=args.source_domain, datapath=source_config["data_dir"], datapaths=source_val_files,
        video_length=args.video_length, frame_length=args.frame_length, step=args.step,
        sampling_ratio=args.sampling_ratio, window_size=args.window_size,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)
    print(f"Train samples: {len(train_dataset)}  Val: {len(val_dataset)}")
    student = build_student(args, device)
    teacher = build_teacher(args, device)
    policy = PolicyConvNet(
        in_ch=3, base_ch=args.policy_base_ch, depth=args.policy_depth,
        k=args.policy_k, dropout=args.policy_dropout,
        md_rank=args.policy_md_rank, md_hidden=args.policy_md_hidden,
    ).to(device)
    #load_checkpoint_weights(student, args.rgb_checkpoint, tag="Student")
    load_checkpoint_weights(teacher, args.rf_checkpoint, tag="Teacher")
    test_collate_fn = None
    test_dataset = dataset_factory.create_paired_distill_dataset(
        target_domain=args.target_domain, datapath=target_config["data_dir"], datapaths=target_test_files,
        video_length=args.video_length, frame_length=args.frame_length, step=args.step,
        sampling_ratio=args.sampling_ratio, window_size=args.window_size,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                             collate_fn=test_collate_fn)
    print("Pre-training evaluation...")
    pre_results = evaluate_student(student, test_loader, device, args)
    print("Pre-training Evaluation Results:")
    print(f"HR MAE: {pre_results['hr_mae_mean']:.4f} ± {pre_results.get('hr_mae_std', 0.0):.4f}")
    print(f"HR RMSE: {pre_results['hr_rmse_mean']:.4f} ± {pre_results.get('hr_rmse_std', 0.0):.4f}")
    print(f"HR Correlation: {pre_results['hr_corr_mean']:.4f}")
    print(f"HR Error Std: {pre_results.get('hr_error_std', 0.0):.4f}")
    print(f"SNR: {pre_results['snr_mean']:.4f} ± {pre_results.get('snr_std', 0.0):.4f}")
    print("Starting training...")
    student, train_summary = train_model(student, teacher, policy, train_loader, val_loader, device, args)
    print("Training completed!")
    print("Evaluating student...")
    results = evaluate_student(student, test_loader, device, args)
    print("Student Evaluation Results:")
    print(f"HR MAE: {results['hr_mae_mean']:.4f} ± {results.get('hr_mae_std', 0.0):.4f}")
    print(f"HR RMSE: {results['hr_rmse_mean']:.4f} ± {results.get('hr_rmse_std', 0.0):.4f}")
    print(f"HR Correlation: {results['hr_corr_mean']:.4f}")
    print(f"HR Error Std: {results.get('hr_error_std', 0.0):.4f}")
    print(f"SNR: {results['snr_mean']:.4f} ± {results.get('snr_std', 0.0):.4f}")
    print(f"Peak Freq MAE: {results.get('peak_freq_mae', 0.0):.4f}")
    print(f"Band Ratio MAE: {results.get('band_ratio_mae', 0.0):.4f}")
    print(f"Number of samples: {results['num_samples']}")
    eval_type = "intra" if args.source_domain == args.target_domain else "cross"
    results_dir = (
        f"eval_results/{eval_type}/{args.target_domain}/RPM-Distill/"
        f"{args.student_model}/frame{args.frame_length}_fold{args.fold}_step{args.step}"
    )
    os.makedirs(results_dir, exist_ok=True)
    row = {
        "Model": args.student_model, "Teacher_Model": args.teacher_model,
        "Source_Domain": args.source_domain, "Target_Domain": args.target_domain,
        "Fold": args.fold, "Frame_Length": args.frame_length, "Step": args.step,
        "Timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"), "Eval_Type": eval_type,
        "TRAIN_MODE": args.train_mode, "META_LR": args.meta_lr, "META_INTERVAL": args.meta_interval,
        "VIRTUAL_LR": args.virtual_lr, "LAMBDA_DISTILL": args.lambda_distill,
        "POLICY_TYPE": args.policy_type, "POLICY_BASE_CH": args.policy_base_ch,
        "POLICY_DEPTH": args.policy_depth, "POLICY_MD_RANK": args.policy_md_rank,
        "POLICY_MD_HIDDEN": args.policy_md_hidden, "POLICY_K": args.policy_k,
        "GATE_REG": args.gate_reg,
        "G_MEAN": train_summary.get("g_mean", 0.0),
        "POLICY_ENTROPY_ALPHA": train_summary.get("policy_entropy_alpha", 0.0),
        "RGB_HR_MAE_mean": results["hr_mae_mean"], "RGB_HR_MAE_std": results.get("hr_mae_std", 0),
        "RGB_HR_RMSE_mean": results["hr_rmse_mean"], "RGB_HR_RMSE_std": results.get("hr_rmse_std", 0),
        "RGB_HR_Correlation_mean": results["hr_corr_mean"],
        "RGB_HR_Error_Std": results.get("hr_error_std", 0),
        "RGB_SNR_mean": results["snr_mean"], "RGB_SNR_std": results.get("snr_std", 0),
        "RGB_Samples": results["num_samples"],
        "Peak_Freq_MAE": results.get("peak_freq_mae", 0.0), "Band_Ratio_MAE": results.get("band_ratio_mae", 0.0),
    }
    alpha_means = train_summary.get("alpha_means", [0.0] * args.policy_k)
    for k_idx in range(args.policy_k):
        row[f"ALPHA_MEAN_{k_idx}"] = alpha_means[k_idx] if k_idx < len(alpha_means) else 0.0
    summary_filename = os.path.join(results_dir, "result.csv")
    df_new = pd.DataFrame([row])
    if os.path.exists(summary_filename):
        df_old = pd.read_csv(summary_filename)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(summary_filename, index=False)
    print(f"[INFO] Results saved to: {summary_filename}")


if __name__ == "__main__":
    main()