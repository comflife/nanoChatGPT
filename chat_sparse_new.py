"""
Chat with a trained sparse Longformer model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from sparse_model import get_sparse_model
import requests

# -----------------------------------------------------------------------------
init_from = 'resume'
out_dir = 'out_sparse' # where finetuned model lives
num_samples = 1 # no samples. 1 for 1 chat at a time
max_new_tokens = 100
temperature = 0.8 
top_k = 5 # retain only the top_k most likely tokens, clamp others to have 0 probability
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Auto-detect device
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # disable compile for transformers models
context="<human>Hello, how are you?<endOfText><bot>Thanks, Im good, what about you?<endOfText><human>Im great thanks, My names James, and I'm from the UK, wbu?<endOfText><bot>Hi James, I'm Conner, and im from america. <endOftext>" # a little context for better chat responses

# Model parameters (should match training config)
n_layer = 18
n_head = 16
n_embd = 1024
attention_window = 512
block_size = 1024
vocab_size = 50257

exec(open('configurator.py').read()) # overrides from command line
# -----------------------------------------------------------------------------

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load sparse model
print("Loading sparse model...")
model = get_sparse_model(
    num_layers=n_layer,
    num_heads=n_head,
    hidden_size=n_embd,
    attention_window=attention_window,
    vocab_size=vocab_size,
    max_position_embeddings=block_size,
)

if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'model_final.pt')
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint)
    else:
        print(f"No checkpoint found at {ckpt_path}, using untrained model")

model.eval()
model.to(device)

# gpt-2 encodings
print("loading GPT-2 encodings...")
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"}, disallowed_special=())
decode = lambda l: enc.decode(l)

def generate_response(input_text, max_tokens=100):
    """Generate response using sparse model"""
    input_ids = torch.tensor(encode(input_text), dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        with ctx:
            # Simple greedy generation
            generated_ids = input_ids.clone()
            
            for _ in range(max_tokens):
                # Limit sequence length to model's max
                if generated_ids.shape[1] >= block_size:
                    generated_ids = generated_ids[:, -block_size+1:]
                    attention_mask = torch.ones_like(generated_ids)
                
                outputs = model(input_ids=generated_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, _ = torch.topk(logits, top_k)
                    logits[logits < top_k_logits[:, [-1]]] = float('-inf')
                
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                attention_mask = torch.ones_like(generated_ids)
                
                # Stop if we hit end of text or specific tokens
                if next_token.item() in [enc.encode_single_token('<|endoftext|>')]:
                    break
            
            output = decode(generated_ids[0].tolist())
            return output

def respond(input_text, samples):
    """Generation function compatible with original interface"""
    full_response = generate_response(input_text, max_new_tokens)
    
    # Clean up response similar to original
    output = full_response.replace(input_text, '')
    output = output.partition('<human>')[0]
    output_text = output.rpartition('<endOftext>')[0]
    
    # Clean labels
    output_text = output_text.replace('<human>', ' ')
    output_text = output_text.replace('<bot>', ' ')
    output_text = output_text.replace('<endOfText>', ' ')
    output_text = output_text.replace('<|endoftext|>', ' ')
    
    return output_text.strip()

print("Sparse model loaded! Starting chat...")
print("Type 'quit' to exit")

# chat loop
while True:
    # get input from user
    start_input = input('User: ')
    if start_input.lower() == 'quit':
        break
        
    start = '<human>' + start_input + '<endOfText><bot>'
    
    # context
    context = context + start
    
    try:
        out = respond(context, num_samples)
        context = context + out + '<endOfText>'
        print('Bot: ' + out)
    except Exception as e:
        print(f"Error generating response: {e}")
        print("Bot: Sorry, I encountered an error. Please try again.")
