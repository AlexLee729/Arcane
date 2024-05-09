import os

def chunk_file(input_file, chunk_size_mb):
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    chunks = []
    start = 0
    while start < len(content):
        end = min(start + chunk_size_bytes, len(content))
        chunks.append(content[start:end])
        start = end
    
    return chunks

def save_chunks(chunks, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, chunk in enumerate(chunks):
        output_file = os.path.join(output_dir, f"chunk_{i}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(chunk)
        print(f"Chunk {i} saved to {output_file}")

if __name__ == "__main__":
    input_file = "Training Files/chats_dataset.txt"
    chunk_size_mb = 200 # in megabytes
    output_dir = "Training Files"

    chunks = chunk_file(input_file, chunk_size_mb)
    save_chunks(chunks, output_dir)
