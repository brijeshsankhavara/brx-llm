"""
Pretrain BRX from scratch with the Hugging Face Trainer.

  python -m brx.train --config configs/brx-tiny.yaml

The YAML file has three sections: `model` (BRXConfig fields), `data` (dataset and tokenizer)
and `training` (passed straight to transformers.TrainingArguments).
"""

import argparse
import inspect
import json
import os
from itertools import chain

import yaml
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, default_data_collator

from .config import BRXConfig
from .model import BRXForCausalLM


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_split(data_cfg, split, max_samples):
    # Stream so only the first `max_samples` documents are downloaded, not the whole dataset.
    stream = load_dataset(data_cfg["dataset"], data_cfg.get("subset"), split=split, streaming=True)
    if max_samples:
        stream = stream.take(max_samples)
    return Dataset.from_list(list(stream))


def tokenize_and_pack(dataset, tokenizer, text_column, context_length):
    """Tokenize every document, join them with EOS and cut the stream into blocks of context_length."""

    def tokenize(batch):
        ids = tokenizer(batch[text_column], add_special_tokens=False)["input_ids"]
        return {"input_ids": [doc + [tokenizer.eos_token_id] for doc in ids]}

    def pack(batch):
        tokens = list(chain.from_iterable(batch["input_ids"]))
        total = (len(tokens) // context_length) * context_length  # drop the ragged tail
        blocks = [tokens[i : i + context_length] for i in range(0, total, context_length)]
        return {"input_ids": blocks, "labels": [b.copy() for b in blocks]}  # model shifts labels itself

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    return tokenized.map(pack, batched=True)


def main():
    parser = argparse.ArgumentParser(description="Pretrain a BRX model")
    parser.add_argument("--config", required=True, help="path to a YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg, data_cfg, train_cfg = cfg["model"], cfg["data"], cfg["training"]

    tokenizer = AutoTokenizer.from_pretrained(data_cfg["tokenizer"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    context_length = data_cfg["context_length"]
    train_raw = load_split(data_cfg, data_cfg.get("train_split", "train"), data_cfg.get("max_train_samples"))
    train_dataset = tokenize_and_pack(train_raw, tokenizer, data_cfg.get("text_column", "text"), context_length)

    eval_dataset = None
    if data_cfg.get("eval_split"):
        eval_raw = load_split(data_cfg, data_cfg["eval_split"], data_cfg.get("max_eval_samples"))
        eval_dataset = tokenize_and_pack(eval_raw, tokenizer, data_cfg.get("text_column", "text"), context_length)

    # vocab and special tokens always come from the tokenizer; the context window must fit a block
    config = BRXConfig(
        **{
            **model_cfg,
            "vocab_size": len(tokenizer),
            "max_position_embeddings": max(model_cfg.get("max_position_embeddings", 0), context_length),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
    )
    model = BRXForCausalLM(config)
    print(f"BRX: {count_parameters(model) / 1e6:.2f}M parameters")
    print(f"train blocks: {len(train_dataset)} x {context_length} tokens")

    # transformers < 4.41 names this argument evaluation_strategy
    if "eval_strategy" in train_cfg and "eval_strategy" not in inspect.signature(TrainingArguments).parameters:
        train_cfg["evaluation_strategy"] = train_cfg.pop("eval_strategy")

    training_args = TrainingArguments(**train_cfg)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()

    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    with open(os.path.join(training_args.output_dir, "log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)


if __name__ == "__main__":
    main()
