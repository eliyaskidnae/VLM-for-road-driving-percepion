import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA runtime:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(
            f"GPU {i}: {props.name} | "
            f"{props.total_memory / 1024**3:.1f} GB VRAM | "
            f"compute capability {props.major}.{props.minor}"
        )
else:
    print("\nNo CUDA GPU detected.")
    print("Check nvidia-smi and install a CUDA-enabled PyTorch build.")
