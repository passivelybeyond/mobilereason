"""
Training Loop for MobileReason-MoE
- Distributed Data Parallel (DDP) for 2×T4 on Kaggle
- Gradient checkpointing to save VRAM
- bf16 mixed precision (T4 supports bf16 via emulation; fp16 is faster on T4)
- Cosine LR schedule with warmup
- Checkpoint save/resume across Kaggle sessions
- WandB logging (optional)

HOW TO RUN ON KAGGLE:
  torchrun --nproc_per_node=2 training/train.py --config training/config.json
"""

import os
import sys
import json
import math
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.architecture import MobileReasonMoE, MobileReasonConfig
from data.tokenizer_train import MobileReasonTokenizer
from data.dataset import make_dataloader


# ──────────────────────────────────────────────
# Training config
# ──────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Paths
    tokenizer_path: str = "tokenizer/mobilereason.model"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Training scale
    total_tokens: int = 10_000_000_000   # 10B tokens
    seq_len: int = 2048
    batch_size: int = 4                  # per GPU (T4 16GB)
    grad_accumulation: int = 8           # effective batch = 4*2*8 = 64 seqs = 131K tokens

    # Optimizer
    lr: float = 3e-4
    lr_min: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 2000

    # Regularization
    aux_loss_weight: float = 0.01        # MoE load balancing

    # Precision
    dtype: str = "fp16"                  # T4 is faster with fp16 than bf16

    # Logging & saving
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 1000
    use_wandb: bool = False

    # Gradient checkpointing
    gradient_checkpointing: bool = True

    @property
    def total_steps(self) -> int:
        tokens_per_step = self.seq_len * self.batch_size * self.grad_accumulation
        # Assume 2 GPUs
        return self.total_tokens // (tokens_per_step * 2)


# ──────────────────────────────────────────────
# LR schedule: cosine with warmup
# ──────────────────────────────────────────────

def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.lr_min + cosine * (cfg.lr - cfg.lr_min)


# ──────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    cfg: TrainConfig,
    model_cfg: MobileReasonConfig,
    rank: int,
):
    if rank != 0:
        return
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:07d}.pt"
    raw_model = model.module if hasattr(model, "module") else model
    torch.save({
        "step": step,
        "loss": loss,
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_config": asdict(cfg),
        "model_config": asdict(model_cfg),
    }, path)
    # Keep only last 3 checkpoints
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    for old in ckpts[:-3]:
        old.unlink()
    print(f"[rank 0] Saved checkpoint: {path}")


def load_checkpoint(path: str, model: nn.Module, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"Resumed from step {ckpt['step']}, loss={ckpt['loss']:.4f}")
    return ckpt["step"]


# ──────────────────────────────────────────────
# Gradient checkpointing wrapper
# ──────────────────────────────────────────────

def enable_gradient_checkpointing(model: nn.Module):
    """Enable gradient checkpointing on transformer blocks."""
    from torch.utils.checkpoint import checkpoint

    for layer in model.layers:
        original_forward = layer.forward

        def make_ckpt_forward(orig_fwd):
            def ckpt_forward(x, freqs_cis, mask=None):
                return checkpoint(orig_fwd, x, freqs_cis, mask, use_reentrant=False)
            return ckpt_forward

        layer.forward = make_ckpt_forward(original_forward)


# ──────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────

def train(train_cfg: TrainConfig, model_cfg: MobileReasonConfig, resume_from: str = None):
    # DDP setup
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"Training with {world_size} GPUs")
        print(f"Total steps: {train_cfg.total_steps:,}")
        effective_batch_tokens = train_cfg.seq_len * train_cfg.batch_size * train_cfg.grad_accumulation * world_size
        print(f"Effective batch: {effective_batch_tokens:,} tokens/step")

    # Model
    model = MobileReasonMoE(model_cfg).to(device)

    if train_cfg.gradient_checkpointing:
        enable_gradient_checkpointing(model)

    # Wrap with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # Optimizer: separate weight decay groups
    decay_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": train_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=train_cfg.lr,
        betas=(train_cfg.beta1, train_cfg.beta2),
        eps=1e-8,
        fused=True,  # faster fused AdamW on CUDA
    )

    scaler = GradScaler(enabled=(train_cfg.dtype == "fp16"))
    amp_dtype = torch.float16 if train_cfg.dtype == "fp16" else torch.bfloat16

    # Resume
    start_step = 0
    if resume_from:
        start_step = load_checkpoint(resume_from, model, optimizer)

    # Tokenizer & dataloader
    tokenizer = MobileReasonTokenizer(train_cfg.tokenizer_path)
    loader = make_dataloader(
        tokenizer=tokenizer,
        seq_len=train_cfg.seq_len,
        batch_size=train_cfg.batch_size,
        num_workers=2,
    )
    data_iter = iter(loader)

    # Optional WandB
    if train_cfg.use_wandb and rank == 0:
        import wandb
        wandb.init(project="mobilereason-moe", config={**asdict(train_cfg), **asdict(model_cfg)})

    # ──────────────────────────────────────────
    # Training loop
    # ──────────────────────────────────────────

    model.train()
    optimizer.zero_grad()
    t0 = time.time()
    tokens_seen = start_step * train_cfg.seq_len * train_cfg.batch_size * train_cfg.grad_accumulation * world_size

    for step in range(start_step, train_cfg.total_steps):
        # Update LR
        lr = get_lr(step, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation
        total_loss = 0.0
        for micro_step in range(train_cfg.grad_accumulation):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # Forward
            with autocast(device_type="cuda", dtype=amp_dtype):
                out = model(input_ids, labels=labels, aux_loss_weight=train_cfg.aux_loss_weight)
                loss = out["loss"] / train_cfg.grad_accumulation

            # Backward
            scaler.scale(loss).backward()
            total_loss += loss.item()

        # Gradient clip + optimizer step
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        tokens_seen += train_cfg.seq_len * train_cfg.batch_size * train_cfg.grad_accumulation * world_size

        # Logging
        if rank == 0 and step % train_cfg.log_every == 0:
            dt = time.time() - t0
            tok_per_sec = (train_cfg.seq_len * train_cfg.batch_size * train_cfg.grad_accumulation * world_size * train_cfg.log_every) / dt
            print(
                f"step {step:6d}/{train_cfg.total_steps} | "
                f"loss {total_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"grad_norm {grad_norm:.3f} | "
                f"tok/s {tok_per_sec:,.0f} | "
                f"tokens {tokens_seen/1e9:.2f}B"
            )
            t0 = time.time()

            if train_cfg.use_wandb:
                import wandb
                wandb.log({"loss": total_loss, "lr": lr, "grad_norm": grad_norm, "step": step})

        # Checkpoint
        if rank == 0 and step > 0 and step % train_cfg.save_every == 0:
            save_checkpoint(model, optimizer, step, total_loss, train_cfg, model_cfg, rank)

    # Final save
    save_checkpoint(model, optimizer, train_cfg.total_steps, total_loss, train_cfg, model_cfg, rank)

    if rank == 0:
        print("Training complete!")
    dist.destroy_process_group()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    model_cfg = MobileReasonConfig(
        vocab_size=32000,
        hidden_dim=1024,
        num_layers=24,
        num_heads=16,
        num_kv_heads=4,
        head_dim=64,
        num_experts=8,
        num_active_experts=2,
        expert_hidden_dim=1024,
        shared_expert_hidden_dim=512,
        ffn_hidden_dim=2752,
        max_seq_len=2048,
    )

    train_cfg = TrainConfig(use_wandb=args.wandb)

    train(train_cfg, model_cfg, resume_from=args.resume)
