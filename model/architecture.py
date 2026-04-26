"""
MobileReason-MoE Architecture
- Grouped Query Attention (GQA)
- Rotary Position Embeddings (RoPE)
- RMSNorm (Pre-LN)
- Sparse MoE (every 2nd FFN layer, 8 experts, top-2 routing)
- Shared expert (always-on)
Target: ~600M–1.2B total params, ~160–300M active params
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

@dataclass
class MobileReasonConfig:
    vocab_size: int = 32000
    hidden_dim: int = 1024
    num_layers: int = 24
    num_heads: int = 16          # query heads
    num_kv_heads: int = 4        # key/value heads (GQA)
    head_dim: int = 64
    max_seq_len: int = 2048
    rope_theta: float = 10000.0

    # MoE settings
    moe_every_n_layers: int = 2  # apply MoE every 2nd FFN layer
    num_experts: int = 8
    num_active_experts: int = 2  # top-k
    expert_hidden_dim: int = 1024
    shared_expert_hidden_dim: int = 512  # always-active shared expert

    # Dense FFN (non-MoE layers)
    ffn_hidden_dim: int = 2752   # ~2.7x hidden, SwiGLU friendly

    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    tie_embeddings: bool = True

    def __post_init__(self):
        assert self.hidden_dim == self.num_heads * self.head_dim, \
            "hidden_dim must equal num_heads * head_dim"


# ──────────────────────────────────────────────
# RMSNorm
# ──────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# ──────────────────────────────────────────────
# RoPE
# ──────────────────────────────────────────────

def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # x: (B, T, H, D)
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:x.shape[1]].unsqueeze(0).unsqueeze(2)
    xr = torch.view_as_real(xc * freqs_cis).flatten(3)
    return xr.type_as(x)


# ──────────────────────────────────────────────
# Grouped Query Attention
# ──────────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: MobileReasonConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.groups = cfg.num_heads // cfg.num_kv_heads

        self.q_proj = nn.Linear(cfg.hidden_dim, cfg.num_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_dim, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_dim, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_dim, bias=False)
        self.dropout = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # Expand KV heads to match Q heads (GQA)
        k = k.repeat_interleave(self.groups, dim=2)
        v = v.repeat_interleave(self.groups, dim=2)

        # (B, H, T, D)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]

        # Flash attention (torch 2.0+) or manual
        if hasattr(F, "scaled_dot_product_attention"):
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=(mask is None),
            )
        else:
            scale = math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) / scale
            if mask is not None:
                scores = scores + mask
            else:
                causal = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
                scores = scores + causal
            attn_out = F.softmax(scores, dim=-1) @ v

        out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


# ──────────────────────────────────────────────
# SwiGLU FFN (dense layers)
# ──────────────────────────────────────────────

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up   = nn.Linear(dim, hidden_dim, bias=False)
        self.down  = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ──────────────────────────────────────────────
# MoE Expert + Router
# ──────────────────────────────────────────────

class Expert(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up   = nn.Linear(dim, hidden_dim, bias=False)
        self.down  = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SparseMoE(nn.Module):
    """
    Sparse MoE with top-k routing + always-on shared expert.
    Includes load-balancing auxiliary loss.
    """
    def __init__(self, cfg: MobileReasonConfig):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_active_experts
        self.hidden_dim = cfg.hidden_dim

        self.router = nn.Linear(cfg.hidden_dim, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList([
            Expert(cfg.hidden_dim, cfg.expert_hidden_dim)
            for _ in range(cfg.num_experts)
        ])
        # Shared expert: always active
        self.shared_expert = Expert(cfg.hidden_dim, cfg.shared_expert_hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        flat = x.view(-1, D)  # (B*T, D)
        N = flat.size(0)

        # Router
        logits = self.router(flat)                    # (N, E)
        probs = F.softmax(logits, dim=-1)             # (N, E)
        topk_vals, topk_idx = probs.topk(self.top_k, dim=-1)  # (N, k)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)  # renormalize

        # Load balancing loss (auxiliary)
        # Encourage uniform expert usage
        avg_prob = probs.mean(0)                      # (E,)
        avg_frac = (topk_idx == torch.arange(self.num_experts, device=x.device).unsqueeze(0)).float().mean(0)
        aux_loss = (avg_prob * avg_frac).sum() * self.num_experts

        # Dispatch to experts
        out = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)                   # (N, k)
            token_mask = mask.any(dim=-1)            # (N,)
            if not token_mask.any():
                continue
            token_input = flat[token_mask]
            expert_out = expert(token_input)
            # weighted sum
            weights = topk_vals[token_mask][mask[token_mask]].unsqueeze(-1)
            # scatter back
            weighted = (expert_out * weights)
            out[token_mask] += weighted

        # Add shared expert output
        out = out + self.shared_expert(flat)

        return out.view(B, T, D), aux_loss


# ──────────────────────────────────────────────
# Transformer Block
# ──────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, cfg: MobileReasonConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_dim, cfg.rms_norm_eps)
        self.ffn_norm  = RMSNorm(cfg.hidden_dim, cfg.rms_norm_eps)
        self.attn = GroupedQueryAttention(cfg)

        self.use_moe = (layer_idx % cfg.moe_every_n_layers == 1)
        if self.use_moe:
            self.ffn = SparseMoE(cfg)
        else:
            self.ffn = SwiGLU(cfg.hidden_dim, cfg.ffn_hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm attention
        x = x + self.attn(self.attn_norm(x), freqs_cis, mask)
        # Pre-norm FFN
        ffn_in = self.ffn_norm(x)
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(ffn_in)
        else:
            ffn_out = self.ffn(ffn_in)
            aux_loss = torch.tensor(0.0, device=x.device)
        x = x + ffn_out
        return x, aux_loss


# ──────────────────────────────────────────────
# Full Model
# ──────────────────────────────────────────────

class MobileReasonMoE(nn.Module):
    def __init__(self, cfg: MobileReasonConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(cfg, i) for i in range(cfg.num_layers)
        ])
        self.norm = RMSNorm(cfg.hidden_dim, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # Precompute RoPE frequencies
        freqs = precompute_rope_freqs(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("freqs_cis", freqs)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=self.cfg.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=self.cfg.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        aux_loss_weight: float = 0.01,
    ):
        B, T = input_ids.shape
        x = self.embed(input_ids)

        freqs = self.freqs_cis[:T]
        total_aux = torch.tensor(0.0, device=x.device)

        for layer in self.layers:
            x, aux = layer(x, freqs)
            total_aux = total_aux + aux

        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, V)

        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = ce_loss + aux_loss_weight * total_aux

        return {"loss": loss, "logits": logits, "aux_loss": total_aux}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.cfg.max_seq_len:]
            out = self(ctx)
            logits = out["logits"][:, -1, :] / temperature
            # Top-p (nucleus) sampling
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_logits[cum_probs - F.softmax(sorted_logits, dim=-1) > top_p] = float("-inf")
            probs = F.softmax(sorted_logits, dim=-1)
            next_tok = torch.gather(sorted_idx, -1, torch.multinomial(probs, 1))
            input_ids = torch.cat([input_ids, next_tok], dim=-1)
        return input_ids


# ──────────────────────────────────────────────
# Param count utility
# ──────────────────────────────────────────────

def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    active = 0
    for name, p in model.named_parameters():
        # Shared expert + dense layers + attention = always active
        # Router experts: only top_k / num_experts fraction is active
        if "experts." in name and "shared" not in name:
            cfg = model.cfg
            active += p.numel() * cfg.num_active_experts // cfg.num_experts
        else:
            active += p.numel()
    return {
        "total_params": f"{total/1e6:.1f}M",
        "active_params": f"{active/1e6:.1f}M",
        "total_bytes_fp16": f"{total*2/1e9:.2f}GB",
    }


if __name__ == "__main__":
    cfg = MobileReasonConfig()
    model = MobileReasonMoE(cfg)
    info = count_params(model)
    print("Model size:", info)

    # Quick forward test
    ids = torch.randint(0, cfg.vocab_size, (2, 64))
    out = model(ids, labels=ids)
    print("Loss:", out["loss"].item())
    print("Aux loss:", out["aux_loss"].item())
