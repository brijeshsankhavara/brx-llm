import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # trainable gain
        self.eps = eps

    def forward(self, hidden_states):
        # hidden_states: [B, T, hs]. Normalize in float32 for stability, then cast back.
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)  # (1/n) * sum(x_i^2)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoFormer, Su et al. 2021)."""

    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.base = base

    @torch.no_grad()
    def forward(self, position_ids, dtype):
        # theta_i = base^(-2i/d) for i in [0, d/2). Computed per call rather than stored as a buffer:
        # it is tiny, and non-persistent buffers are not restored by from_pretrained.
        exponent = torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=position_ids.device) / self.head_dim
        inv_freq = 1.0 / (self.base**exponent)  # [hd/2]
        # position_ids: [B, T] -> cos, sin: [B, T, hd]
        freqs = position_ids[..., None].float() * inv_freq  # [B, T, hd/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B, T, hd]
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]  # first half of the head dim
    x2 = x[..., x.shape[-1] // 2 :]  # second half
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x, cos, sin):
    # x: [B, n_heads, T, hd], cos/sin: [B, T, hd] -> broadcast over heads
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(hidden_states, n_repeats):
    # [B, n_kv_heads, T, hd] -> [B, n_kv_heads * n_repeats, T, hd] so each query head has a kv head
    if n_repeats == 1:
        return hidden_states
    batch, n_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, n_kv_heads, n_repeats, seq_len, head_dim)
    return hidden_states.reshape(batch, n_kv_heads * n_repeats, seq_len, head_dim)


class SwiGLU(nn.Module):
    """Gated feed-forward block: down(silu(gate(x)) * up(x))."""

    def __init__(self, hidden_size, intermediate_size, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act = nn.SiLU()

    def forward(self, hidden_states):
        return self.down_proj(self.act(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
