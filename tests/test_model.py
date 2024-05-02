import pytest
import torch

from brx import BRXConfig, BRXForCausalLM
from brx.generate import generate
from brx.layers import RotaryEmbedding, apply_rotary, repeat_kv


@pytest.fixture
def config():
    return BRXConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )


@pytest.fixture
def model(config):
    torch.manual_seed(0)
    return BRXForCausalLM(config).eval()


def test_logits_shape_and_initial_loss(model, config):
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    out = model(input_ids, labels=input_ids)
    assert out.logits.shape == (2, 16, config.vocab_size)
    # an untrained model should be close to uniform: loss ~ ln(vocab_size)
    assert abs(out.loss.item() - torch.log(torch.tensor(config.vocab_size)).item()) < 0.5


def test_attention_is_causal(model, config):
    a = torch.randint(0, config.vocab_size, (1, 16))
    b = a.clone()
    b[:, 8:] = torch.randint(0, config.vocab_size, (1, 8))  # change only future tokens
    torch.testing.assert_close(model(a).logits[:, :8], model(b).logits[:, :8])


def test_right_padding_matches_unpadded(model, config):
    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, 12:] = 0
    padded = model(input_ids, attention_mask=attention_mask).logits[:, :12]
    unpadded = model(input_ids[:, :12]).logits
    torch.testing.assert_close(padded, unpadded)


def test_repeat_kv_shares_heads():
    kv = torch.randn(1, 2, 5, 8)
    out = repeat_kv(kv, 3)
    assert out.shape == (1, 6, 5, 8)
    torch.testing.assert_close(out[:, 0], out[:, 2])  # heads 0..2 share kv head 0
    torch.testing.assert_close(out[:, 3], kv[:, 1])


def test_rotary_preserves_norm_and_encodes_relative_position():
    rope = RotaryEmbedding(head_dim=16)
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)

    def rotate(x, pos):
        cos, sin = rope(torch.tensor([[pos]]), dtype=x.dtype)
        return apply_rotary(x, cos, sin)

    torch.testing.assert_close(rotate(q, 7).norm(), q.norm())
    # q.k depends only on the distance between positions, not their absolute values
    dot_a = (rotate(q, 3) * rotate(k, 1)).sum()
    dot_b = (rotate(q, 12) * rotate(k, 10)).sum()
    torch.testing.assert_close(dot_a, dot_b, rtol=1e-4, atol=1e-4)


def test_grouped_query_attention_has_fewer_kv_params(config):
    gqa = BRXForCausalLM(config)
    mha = BRXForCausalLM(BRXConfig(**{**config.to_dict(), "num_key_value_heads": config.num_attention_heads}))
    attn = gqa.model.blocks[0].attn
    assert attn.k_proj.out_features == config.num_key_value_heads * config.head_dim
    assert sum(p.numel() for p in gqa.parameters()) < sum(p.numel() for p in mha.parameters())


def test_tied_embeddings(config):
    model = BRXForCausalLM(BRXConfig(**{**config.to_dict(), "tie_word_embeddings": True}))
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_save_and_load_roundtrip(model, tmp_path):
    input_ids = torch.randint(0, model.config.vocab_size, (1, 10))
    model.save_pretrained(tmp_path)
    loaded = BRXForCausalLM.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(model(input_ids).logits, loaded(input_ids).logits)


def test_overfits_a_single_batch(config):
    torch.manual_seed(0)
    model = BRXForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (4, 16))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(60):
        loss = model(input_ids, labels=input_ids).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.5


def test_gradient_checkpointing_runs(model, config):
    model.train()
    model.gradient_checkpointing_enable()
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    model(input_ids, labels=input_ids).loss.backward()
    assert model.lm_head.weight.grad is not None


@pytest.mark.parametrize("kwargs", [{"temperature": 0}, {"top_k": 5}, {"top_p": 0.9}])
def test_generate(model, config, kwargs):
    prompt = torch.randint(0, config.vocab_size, (1, 4))
    out = generate(model, prompt, max_new_tokens=6, **kwargs)
    assert out.shape == (1, 10)
    assert torch.equal(out[:, :4], prompt)


def test_greedy_generation_is_deterministic(model, config):
    prompt = torch.randint(0, config.vocab_size, (1, 4))
    a = generate(model, prompt, max_new_tokens=6, temperature=0)
    b = generate(model, prompt, max_new_tokens=6, temperature=0)
    assert torch.equal(a, b)


def test_rejects_sequences_longer_than_context(model, config):
    with pytest.raises(ValueError):
        model(torch.zeros(1, config.max_position_embeddings + 1, dtype=torch.long))
