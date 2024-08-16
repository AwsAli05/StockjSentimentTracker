import torch
import time

def benchmark_gpu(device, size=20000, iterations=20000):
    # Create random data for matrix multiplication
    data = torch.randn(size, size, device=device)

    # Warm up
    for _ in range(10):
        _ = torch.mm(data, data)

    # Timing
    start_time = time.time()
    for _ in range(iterations):
        _ = torch.mm(data, data)
    end_time = time.time()

    print(f"Total time to perform {iterations} matrix multiplications of size {size}x{size}: {end_time - start_time:.2f} seconds")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Benchmarking GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

benchmark_gpu(device)
