"""
Training script for sparse Longformer-based GPT model with multi-GPU support.
"""
import os
import argparse
import time
import numpy as np  # noqa
import torch  # noqa
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sparse_model import get_sparse_model  # noqa


def get_batch(data, batch_size, block_size, device, vocab_size=200019):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    # Clamp tokens to be within vocab range
    x = torch.clamp(x, 0, vocab_size - 1)
    y = torch.clamp(y, 0, vocab_size - 1)
    # Create attention mask (all 1s since we don't have padding)
    attention_mask = torch.ones_like(x)
    return x.to(device), y.to(device), attention_mask.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Chat', help='subfolder in data/ for train/val bins')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--max_iters', type=int, default=10000)
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=6e-4)
    parser.add_argument('--n_layer', type=int, default=12)    # Match train2.py default
    parser.add_argument('--n_head', type=int, default=12)     # Match train2.py default  
    parser.add_argument('--n_embd', type=int, default=768)    # Match train2.py default
    parser.add_argument('--attention_window', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='out_sparse')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    args = parser.parse_args()

    # Setup distributed training
    if args.local_rank == -1:
        # Single GPU or multi-GPU without distributed
        # Detect available GPUs dynamically
        num_gpus = torch.cuda.device_count()
        device_ids = list(range(num_gpus))  # Use all available GPUs
        device = f'cuda:{device_ids[0]}' if device_ids else 'cpu'
        if device != 'cpu':
            torch.cuda.set_device(device)
        ddp = False
        world_size = 1
        rank = 0
        print(f"Detected {num_gpus} GPUs: {device_ids}")
    else:
        # Distributed training
        dist.init_process_group(backend='nccl')
        args.local_rank = int(os.environ['LOCAL_RANK'])
        # In DDP mode, local_rank directly maps to visible device
        gpu_id = args.local_rank  
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(device)
        ddp = True
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device_ids = []  # Not used in DDP mode
    
    print(f"Using device: {device}, rank: {rank}, world_size: {world_size}")

    # data memmap
    if rank == 0:
        print("Loading data...")
    start_time = time.time()
    data_dir = os.path.join('data', args.dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train4.bin'), dtype=np.uint32, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val4.bin'), dtype=np.uint32, mode='r')
    if rank == 0:
        print(f"Data loading took {time.time() - start_time:.2f} seconds")
        print(f"Train data size: {len(train_data):,} tokens")
        print(f"Val data size: {len(val_data):,} tokens")

    # infer vocab_size
    if args.vocab_size is None:
        meta_path = os.path.join(data_dir, 'meta.pkl')
        if os.path.exists(meta_path):
            import pickle
            meta = pickle.load(open(meta_path, 'rb'))
            args.vocab_size = meta.get('vocab_size', 200019)  # Default to o200k_base
        else:
            args.vocab_size = 200019  # Default to o200k_base
    if rank == 0:
        print(f"Using vocab_size={args.vocab_size}")
        
        # Debug: check data range
        print(f"Train data max value: {train_data.max()}")
        print(f"Train data min value: {train_data.min()}")
        if train_data.max() >= args.vocab_size:
            print(f"WARNING: Data contains tokens >= vocab_size ({train_data.max()} >= {args.vocab_size})")
            print("This will be clipped during training to prevent errors.")

    # build model
    if rank == 0:
        print("Building model...")
    start_time = time.time()
    model = get_sparse_model(
        num_layers=args.n_layer,
        num_heads=args.n_head,
        hidden_size=args.n_embd,
        attention_window=args.attention_window,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.block_size,
    )
    model.to(device)
    
    # Wrap model for multi-GPU
    if not ddp and len(device_ids) > 1:
        # DataParallel for single-node multi-GPU
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        if rank == 0:
            print(f"Using DataParallel with GPUs: {device_ids}")
    elif ddp:
        # DistributedDataParallel - use the correct GPU ID
        model = DDP(model, device_ids=[gpu_id])
        if rank == 0:
            print(f"Using DistributedDataParallel")
    
    model.train()
    if rank == 0:
        print(f"Model building took {time.time() - start_time:.2f} seconds")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    if rank == 0:
        print("Starting training...")
    start_time = time.time()
    
    for iter_num in range(1, args.max_iters + 1):
        iter_start = time.time()
        x, y, attention_mask = get_batch(train_data, args.batch_size, args.block_size, device, args.vocab_size)
        data_time = time.time() - iter_start
        
        forward_start = time.time()
        outputs = model(input_ids=x, labels=y, attention_mask=attention_mask)
        loss = outputs.loss
        # Handle DataParallel loss averaging
        if isinstance(model, torch.nn.DataParallel):
            loss = loss.mean()
        forward_time = time.time() - forward_start
        
        backward_start = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backward_time = time.time() - backward_start
        
        if rank == 0 and (iter_num <= 5 or iter_num % 100 == 0):
            print(f"iter {iter_num}: loss {loss.item():.4f}, data_time {data_time:.3f}s, forward_time {forward_time:.3f}s, backward_time {backward_time:.3f}s")

        if iter_num % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val, attention_mask_val = get_batch(val_data, args.batch_size, args.block_size, device, args.vocab_size)
                val_outputs = model(input_ids=x_val, labels=y_val, attention_mask=attention_mask_val)
                val_loss = val_outputs.loss
                if isinstance(model, torch.nn.DataParallel):
                    val_loss = val_loss.mean()
            if rank == 0:
                print(f"iter {iter_num}: train_loss {loss.item():.4f}, val_loss {val_loss.item():.4f}")
            model.train()

        if rank == 0 and iter_num % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f'model_{iter_num}.pt')
            # Save the underlying model, not the wrapper
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({'iter': iter_num, 'model_state': model_to_save.state_dict(), 'optimizer_state': optimizer.state_dict()}, ckpt_path)

    # final save
    if rank == 0:
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, 'model_final.pt'))
        
    if ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
