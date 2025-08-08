"""
Sparse GPT model using Longformer's sparse attention mechanism.
This file defines a helper to create a Longformer-based causal language model with sliding-window sparse attention.
"""
from transformers import LongformerConfig, LongformerForMaskedLM


def get_sparse_model(
    num_layers: int = 12,
    num_heads: int = 12,
    hidden_size: int = 768,
    attention_window: int = 512,
    vocab_size: int = 200019,  # Default to o200k_base
    max_position_embeddings: int = 4096,
    dropout: float = 0.1,
):
    """
    Builds a Longformer-based causal language model with sparse attention.

    Args:
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        hidden_size: Hidden dimension size.
        attention_window: Sliding window size for local attention.
        vocab_size: Vocabulary size (default: 200019 for o200k_base).
        max_position_embeddings: Maximum sequence length.
        dropout: Dropout probability.

    Returns:
        A LongformerForMaskedLM model configured for causal LM.
    """
    config = LongformerConfig(
        attention_window=[attention_window] * num_layers,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,  
        attention_probs_dropout_prob=dropout,
        hidden_dropout_prob=dropout,
        pad_token_id=0,  # ensure padding token
        bos_token_id=1,
        eos_token_id=2,
        # Make it more suitable for causal LM
        is_decoder=True,
        add_cross_attention=False,
    )
    # instantiate Longformer for masked LM (we'll use it for causal LM)
    model = LongformerForMaskedLM(config)
    return model
