import os
import requests
import tiktoken
import numpy as np

train_ids = []
val_ids = []
enc = tiktoken.get_encoding("cl100k_base")

def download_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        with open('dataset.txt', 'wb') as f:
            f.write(response.content)
            print("downloaded dataset, tokenizing")
    else:
        print('Error downloading file:', response.status_code)

download_file('https://huggingface.co/VatsaDev/ChatGpt-nano/resolve/main/Dataset.txt')

def split_file(filename, output_dir, chunk_size):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(filename, 'r') as f:
        lines = f.readlines()

    n_chunks = (len(lines) + chunk_size - 1) // chunk_size  # Better handling for remainder
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(lines))

        chunk_lines = lines[start:end]

        output_filename = os.path.join(output_dir, f'{i}-dataset.txt')
        with open(output_filename, 'w') as f:
            f.writelines(chunk_lines)

split_file('dataset.txt', 'output', 100000)

def get_chunk_num(filename):
    try:
        return int(filename.split('-')[0])
    except ValueError:
        return -1  # Invalid, skip

for filename in os.listdir('output'):
    if filename.endswith('.txt'):
        chunk_num = get_chunk_num(filename)
        if chunk_num >= 0:
            with open(os.path.join('output', filename), 'r') as f:
                data = f.read()
            ids = enc.encode_ordinary(data)
            if chunk_num < 48:
                train_ids.extend(ids)
            else:
                val_ids.extend(ids)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

train_ids = np.array(train_ids, dtype=np.uint32)
val_ids = np.array(val_ids, dtype=np.uint32)

data_dir = os.path.dirname(__file__)
train_ids.tofile(os.path.join(data_dir, 'train3.bin'))
val_ids.tofile(os.path.join(data_dir, 'val3.bin'))