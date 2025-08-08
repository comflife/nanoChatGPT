#!/usr/bin/env python3
"""
Test script for sparse model consistency
"""

import os
import pickle
import numpy as np
import torch
from sparse_model import get_sparse_model

def test_sparse_model():
    print("=== Testing Sparse Model Consistency ===")
    
    # Check if data files exist
    data_dir = 'data/Chat'
    meta_file = os.path.join(data_dir, 'meta.pkl')
    
    if not os.path.exists(meta_file):
        print("❌ Missing meta.pkl file!")
        return False
    
    # Load meta info
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f)
    
    meta_vocab_size = meta['vocab_size']
    print(f"Meta vocab_size: {meta_vocab_size}")
    
    # Test sparse model creation
    try:
        model = get_sparse_model(
            num_layers=12,
            num_heads=12,
            hidden_size=768,
            attention_window=512,
            vocab_size=meta_vocab_size,
            max_position_embeddings=1024,
        )
        print(f"✅ Sparse model created successfully")
        print(f"Model vocab_size: {model.config.vocab_size}")
        
        # Test forward pass
        batch_size = 2
        seq_len = 32
        
        input_ids = torch.randint(0, meta_vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, meta_vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss
            logits = outputs.logits
        
        print(f"✅ Forward pass works! Loss: {loss.item():.4f}")
        print(f"Logits shape: {logits.shape}")
        print(f"Expected shape: [{batch_size}, {seq_len}, {meta_vocab_size}]")
        
        if logits.shape != (batch_size, seq_len, meta_vocab_size):
            print(f"❌ Logits shape mismatch!")
            return False
            
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"❌ Sparse model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✅ All sparse model tests passed!")
    return True

if __name__ == "__main__":
    success = test_sparse_model()
    if success:
        print("\n🎉 Sparse model is ready for training!")
    else:
        print("\n💥 Sparse model tests failed.")
