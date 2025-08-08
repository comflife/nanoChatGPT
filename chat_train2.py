"""
Chat with a trained model (train2.py version with o200k_base tokenizer)
python chat_train2.py --out_dir=out --context="hello how are you"
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import requests

# -----------------------------------------------------------------------------
init_from = 'resume'
out_dir = 'out' # where finetuned model lives (train2 output)
num_samples = 1 # no samples. 1 for 1 chat at a time
max_new_tokens = 100
temperature = 0.8 
top_k = 5 # retain only the top_k most likely tokens, clamp others to have 0 probability
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
context="<human>Hello, how are you?<endOfText><bot>Thanks, Im good, what about you?<endOfText><human>Im great thanks, My names James, and I'm from the UK, wbu?<endOfText><bot>Hi James, I'm Conner, and im from america. <endOftext>" # a little context for better chat responses
exec(open('configurator.py').read()) # overrides from command line, only for out_dir location
# -----------------------------------------------------------------------------

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def download_ckpt(url):
  response = requests.get(url)
  if response.status_code == 200:
    with open('ckpt.pt', 'wb') as f:
      f.write(response.content)
  else:
    print('Error downloading file:', response.status_code)

# Load model trained with train2.py
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}")
        exit(1)
    
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    # Print model info
    print(f"Model vocab_size: {gptconf.vocab_size}")
    print(f"Model layers: {gptconf.n_layer}")
    print(f"Model heads: {gptconf.n_head}")
    print(f"Model embedding: {gptconf.n_embd}")

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# Use the same tokenizer as train2.py (o200k_base)
print("Using o200k_base tokenizer (same as train2.py)...")

# Check model vocab_size first to determine correct tokenizer
model_vocab_size = gptconf.vocab_size
print(f"Model vocab_size: {model_vocab_size}")

if model_vocab_size == 200019:
    # This matches o200k_base tokenizer
    enc = tiktoken.get_encoding("o200k_base")
    tokenizer_name = "o200k_base"
    print("✓ Using o200k_base tokenizer (matches train2.py)")
elif model_vocab_size in [50257, 50304]:
    # This matches GPT-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    tokenizer_name = "gpt2"
    print("Using GPT-2 tokenizer")
elif model_vocab_size == 100256:
    # This matches cl100k_base tokenizer
    enc = tiktoken.get_encoding("cl100k_base")
    tokenizer_name = "cl100k_base"
    print("Using cl100k_base tokenizer")
else:
    # Unknown vocab size, default to o200k_base
    print(f"Warning: Unknown vocab_size {model_vocab_size}, defaulting to o200k_base")
    enc = tiktoken.get_encoding("o200k_base")
    tokenizer_name = "o200k_base"

# Check if vocab sizes match
model_vocab_size = gptconf.vocab_size
tokenizer_vocab_size = enc.n_vocab
print(f"Model vocab_size: {model_vocab_size}")
print(f"Tokenizer vocab_size: {tokenizer_vocab_size}")

if model_vocab_size != tokenizer_vocab_size:
    print(f"WARNING: Vocab size mismatch! Model expects {model_vocab_size}, tokenizer has {tokenizer_vocab_size}")
    print("This will cause garbled output. Please use the correct tokenizer.")
    
    # Try to find the right tokenizer
    if model_vocab_size == 50257 or model_vocab_size == 50304:
        print("Model seems to be trained with GPT-2 tokenizer")
        enc = tiktoken.get_encoding("gpt2")
        tokenizer_name = "gpt2"
    elif model_vocab_size >= 200000:
        print("Model seems to be trained with o200k_base tokenizer")
        try:
            enc = tiktoken.get_encoding("o200k_base")
            tokenizer_name = "o200k_base"
        except:
            enc = tiktoken.get_encoding("cl100k_base")
            tokenizer_name = "cl100k_base"

print(f"Final tokenizer: {tokenizer_name}, vocab_size: {enc.n_vocab}")

# Allow special tokens for better compatibility
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"}, disallowed_special=())
decode = lambda l: enc.decode(l)

def respond(input, samples): # generation function
    x = (torch.tensor(encode(input), dtype=torch.long, device=device)[None, ...]) 
    with torch.no_grad():
        with ctx:
            for k in range(samples):
                generated = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                output = decode(generated[0].tolist())

                # replace context
                output = output.replace(input,'')
                # remove any human response
                output = output.partition('<human>')
                # if the bot has anything left afterwards, the endOfText token is put to use
                output_text = output[0].rpartition('<endOftext>')
                output_text = output[0] + output[1]
                # label removing
                output_text = output_text.replace('<human>',' ')
                output_text = output_text.replace('<bot>',' ')
                output_text = output_text.replace('<endOfText>',' ')
                output_text = output_text.replace('<endOftext>',' ')
                output_text = output_text.replace('<|endoftext|>',' ')
                return output_text.strip()

print("Chat with train2 model loaded! (o200k_base tokenizer)")
print("Type 'quit' to exit")

# chat loop
while True:
    # get input from user
    start_input = input('User: ')
    if start_input.lower() == 'quit':
        break
    
    start = '<human>'+start_input+'<endOfText><bot>'

    # context
    context=context+start
    
    out = respond(context, num_samples)
    context=context+out+'<endOfText>'
    print('Bot: '+ out)
