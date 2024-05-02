from transformers import PretrainedConfig


class BRXConfig(PretrainedConfig):
    model_type = "brx"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=1024,  # 2048 in TinyLlama
        intermediate_size=2048,
        num_hidden_layers=6,
        num_attention_heads=16,  # 32 in TinyLlama
        num_key_value_heads=2,  # < num_attention_heads -> grouped-query attention
        max_position_embeddings=2048,  # max context length
        initializer_range=0.02,  # std used to initialize the weights
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        use_bias=False,  # bias in the attention and feed-forward projections
        lm_head_bias=False,
        tie_word_embeddings=False,  # share the input embedding matrix with the output head
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
            )
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({num_key_value_heads})"
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.use_bias = use_bias
        self.lm_head_bias = lm_head_bias

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads
