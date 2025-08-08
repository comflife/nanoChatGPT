#!/usr/bin/env python3
"""
Test script to check data consistency between prepare4.py and train2.py
"""

import os
import pickle
import numpy as np
import tiktoken
import torch

def test_data_consistency():
    print("=== Testing Data Consistency ===")
    
    # 1. Check if data files exist
    data_dir = 'data/Chat'
    train_file = os.path.join(data_dir, 'train4.bin')
    val_file = os.path.join(data_dir, 'val4.bin')
    meta_file = os.path.join(data_dir, 'meta.pkl')
    
    print(f"Train file exists: {os.path.exists(train_file)}")
    print(f"Val file exists: {os.path.exists(val_file)}")
    print(f"Meta file exists: {os.path.exists(meta_file)}")
    
    if not all([os.path.exists(f) for f in [train_file, val_file, meta_file]]):
        print("❌ Missing data files! Run prepare4.py first.")
        return False
    
    # 2. Check meta.pkl content
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f)
    
    meta_vocab_size = meta['vocab_size']
    print(f"Meta vocab_size: {meta_vocab_size}")
    
    # 3. Check o200k_base tokenizer
    enc = tiktoken.get_encoding("o200k_base")
    actual_vocab_size = enc.n_vocab
    print(f"o200k_base actual vocab_size: {actual_vocab_size}")
    
    if meta_vocab_size != actual_vocab_size:
        print(f"❌ Vocab size mismatch! Meta: {meta_vocab_size}, Actual: {actual_vocab_size}")
        return False
    else:
        print("✅ Vocab sizes match!")
    
    # 4. Check data files content
    train_data = np.memmap(train_file, dtype=np.uint32, mode='r')
    val_data = np.memmap(val_file, dtype=np.uint32, mode='r')
    
    print(f"Train data size: {len(train_data):,} tokens")
    print(f"Val data size: {len(val_data):,} tokens")
    print(f"Train data max token: {train_data.max()}")
    print(f"Train data min token: {train_data.min()}")
    print(f"Val data max token: {val_data.max()}")
    print(f"Val data min token: {val_data.min()}")
    
    # 5. Check if tokens are within vocab range
    if train_data.max() >= meta_vocab_size:
        print(f"❌ Train data has tokens >= vocab_size! Max: {train_data.max()}, Vocab: {meta_vocab_size}")
        return False
    
    if val_data.max() >= meta_vocab_size:
        print(f"❌ Val data has tokens >= vocab_size! Max: {val_data.max()}, Vocab: {meta_vocab_size}")
        return False
    
    print("✅ All tokens are within vocab range!")
    
    # 6. Test tokenizer decoding
    sample_tokens = train_data[:100].astype(np.int64)
    try:
        decoded_text = enc.decode(sample_tokens.tolist())
        print(f"Sample decoded text (first 200 chars): {decoded_text[:200]}...")
        print("✅ Tokenizer decoding works!")
    except Exception as e:
        print(f"❌ Tokenizer decoding failed: {e}")
        return False
    
    # 7. Test model loading
    try:
        from model import GPTConfig, GPT
        
        # Test with correct vocab_size
        model_args = {
            'n_layer': 12,
            'n_head': 12, 
            'n_embd': 768,
            'block_size': 1024,
            'bias': False,
            'vocab_size': meta_vocab_size,
            'dropout': 0.0
        }
        
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        print(f"✅ Model created successfully with vocab_size: {gptconf.vocab_size}")
        
        # Test forward pass with sample data
        x = torch.tensor(sample_tokens[:32].reshape(1, -1), dtype=torch.long)
        with torch.no_grad():
            logits, loss = model(x, x)
        
        print(f"✅ Model forward pass works! Logits shape: {logits.shape}")
        print(f"Expected logits shape: [1, 32, {meta_vocab_size}]")
        
        if logits.shape[-1] != meta_vocab_size:
            print(f"❌ Logits vocab dimension mismatch! Got: {logits.shape[-1]}, Expected: {meta_vocab_size}")
            return False
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 8. Check checkpoint if exists
    ckpt_path = 'out/ckpt.pt'
    if os.path.exists(ckpt_path):
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model_vocab_size = checkpoint['model_args']['vocab_size']
            print(f"Checkpoint vocab_size: {model_vocab_size}")
            
            if model_vocab_size != meta_vocab_size:
                print(f"❌ Checkpoint vocab_size mismatch! Checkpoint: {model_vocab_size}, Data: {meta_vocab_size}")
                return False
            else:
                print("✅ Checkpoint vocab_size matches data!")
                
        except Exception as e:
            print(f"⚠️ Could not load checkpoint: {e}")
    else:
        print("ℹ️ No checkpoint found (this is OK if training hasn't started)")
    
    print("\n=== Test Summary ===")
    print("✅ All tests passed! Data consistency is good.")
    return True

def test_tokenizer_compatibility():
    print("\n=== Testing Tokenizer Compatibility ===")
    
    # Test different chat formats
    test_texts = [
        "<human>Hello<endOfText><bot>Hi there!<endOfText>",
        "<human>How are you?<endOfText><bot>I'm good, thanks!<endOfText>",
        "Hello world",
        "<|endoftext|>",
    ]
    
    enc = tiktoken.get_encoding("o200k_base")
    
    for text in test_texts:
        try:
            tokens = enc.encode(text, allowed_special={"<|endoftext|>"}, disallowed_special=())
            decoded = enc.decode(tokens)
            
            print(f"Text: '{text[:50]}...'")
            print(f"Tokens: {len(tokens)} tokens")
            print(f"Max token: {max(tokens) if tokens else 0}")
            print(f"Decoded matches: {text == decoded}")
            print("---")
            
        except Exception as e:
            print(f"❌ Failed for text '{text}': {e}")
            return False
    
    print("✅ Tokenizer compatibility tests passed!")
    return True

if __name__ == "__main__":
    success1 = test_data_consistency()
    success2 = test_tokenizer_compatibility()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Your setup should work correctly.")
    else:
        print("\n💥 Some tests failed. Please check the issues above.")
