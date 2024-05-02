"""
Per-token workflow through one BRX block:

  input_ids -> token embedding (hidden_size)
    -> RMSNorm -> q, k, v projections -> rotary positions on q, k
    -> repeat k, v for grouped-query attention -> causal attention -> o_proj -> + residual
    -> RMSNorm -> SwiGLU feed-forward -> + residual
  -> final RMSNorm -> lm_head -> logits over the vocabulary
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .config import BRXConfig
from .layers import RMSNorm, RotaryEmbedding, SwiGLU, apply_rotary, repeat_kv


def build_attention_mask(attention_mask):
    # attention_mask: [B, T] with 1 for real tokens, 0 for padding -> boolean [B, 1, T, T] (True = attend)
    seq_len = attention_mask.shape[-1]
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=attention_mask.device).tril()
    keep = causal[None, None, :, :] & attention_mask.bool()[:, None, None, :]
    # A query that only sees padding would produce NaNs, so always let a token attend to itself.
    return keep | torch.eye(seq_len, dtype=torch.bool, device=attention_mask.device)


class BRXAttention(nn.Module):
    def __init__(self, config: BRXConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_repeats = self.n_heads // self.n_kv_heads  # query heads per kv head

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=config.use_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=config.use_bias)

    def forward(self, hidden_states, cos, sin, mask=None):
        B, T, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, nh, T, hd]
        k = self.k_proj(hidden_states).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, nkv, T, hd]
        v = self.v_proj(hidden_states).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, nkv, T, hd]

        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        k = repeat_kv(k, self.n_repeats)  # [B, nh, T, hd]
        v = repeat_kv(v, self.n_repeats)

        if mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # mask already includes causality

        y = y.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)  # [B, T, hidden]
        return self.o_proj(y)


class BRXBlock(nn.Module):
    def __init__(self, config: BRXConfig):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = BRXAttention(config)
        self.post_attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = SwiGLU(config.hidden_size, config.intermediate_size, bias=config.use_bias)

    def forward(self, hidden_states, cos, sin, mask=None):
        h = hidden_states + self.attn(self.input_norm(hidden_states), cos, sin, mask)  # pre-norm residual
        return h + self.ffn(self.post_attention_norm(h))


class BRXPreTrainedModel(PreTrainedModel):
    config_class = BRXConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BRXBlock"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)


class BRXModel(BRXPreTrainedModel):
    def __init__(self, config: BRXConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(config.head_dim, base=config.rope_theta)
        self.blocks = nn.ModuleList([BRXBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,  # [B, T]
        attention_mask: torch.Tensor | None = None,  # [B, T]
        position_ids: torch.LongTensor | None = None,  # [B, T]
    ):
        B, T = input_ids.shape
        if T > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence length {T} exceeds max_position_embeddings {self.config.max_position_embeddings}"
            )

        x = self.embed_tokens(input_ids)  # [B, T, hidden]

        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        cos, sin = self.rotary_emb(position_ids, dtype=x.dtype)  # computed once, shared by every block

        # Without padding, SDPA's built-in causal kernel is faster than an explicit mask.
        mask = None
        if attention_mask is not None and not bool(attention_mask.all()):
            mask = build_attention_mask(attention_mask)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = self._gradient_checkpointing_func(block.__call__, x, cos, sin, mask)
            else:
                x = block(x, cos, sin, mask)

        return self.norm(x)  # [B, T, hidden]


class BRXForCausalLM(BRXPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: BRXConfig):
        super().__init__(config)
        self.model = BRXModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.lm_head_bias)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,  # [B, T], -100 marks positions to ignore
        **kwargs,
    ):
        hidden_states = self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        logits = self.lm_head(hidden_states).float()  # [B, T, vocab]

        loss = None
        if labels is not None:
            # shift so that tokens < n predict token n
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            shift_labels = labels[:, 1:].reshape(-1).to(shift_logits.device)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        return CausalLMOutput(loss=loss, logits=logits)
