"""
Sample text from a trained BRX checkpoint.

  python -m brx.generate --model runs/brx-tiny --prompt "Once upon a time"
"""

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .model import BRXForCausalLM


@torch.no_grad()
def generate(model, input_ids, max_new_tokens=100, temperature=1.0, top_k=None, top_p=None, eos_token_id=None):
    """Autoregressive sampling. temperature=0 means greedy decoding."""
    model.eval()
    max_len = model.config.max_position_embeddings

    for _ in range(max_new_tokens):
        context = input_ids[:, -max_len:]  # crop to the context window
        logits = model(context).logits[:, -1, :]  # [B, vocab], logits for the next token only

        if temperature == 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature

            if top_k is not None:
                # keep the k most likely tokens
                kth_value = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                logits = logits.masked_fill(logits < kth_value, float("-inf"))

            if top_p is not None:
                # keep the smallest set of tokens whose cumulative probability exceeds top_p
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_to_remove = cumulative_probs > top_p
                sorted_to_remove[:, 1:] = sorted_to_remove[:, :-1].clone()  # always keep the first token
                sorted_to_remove[:, 0] = False
                to_remove = sorted_to_remove.scatter(1, sorted_indices, sorted_to_remove)
                logits = logits.masked_fill(to_remove, float("-inf"))

            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)  # [B, 1]

        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if eos_token_id is not None and bool((next_token == eos_token_id).all()):
            break

    return input_ids


def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained BRX model")
    parser.add_argument("--model", required=True, help="directory written by brx.train")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = BRXForCausalLM.from_pretrained(args.model).to(device)

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
